// gguf.js — ブラウザ / Node 両対応の最小 GGUF v3 リーダ
//
// llama.cpp の GGUF は汎用テンソル容器なので、ハエのコネクトーム（疎行列）も
// そのまま格納できる。ここでは Q8_0 の逆量子化まで含めて実装する。
//
// 使い方:
//   const gguf = await GGUF.fromArrayBuffer(buf);
//   gguf.kv['cx.neuron_count'];
//   gguf.tensor('cx.weights');   // Float32Array (Q8_0 を逆量子化)

const GGUF_MAGIC = 0x46554747; // 'G','G','U','F' をリトルエンディアンの u32 で読んだ値

const VT = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};

const GGML = {
  F32: 0, F16: 1, Q8_0: 8, I8: 24, I16: 25, I32: 26, I64: 27, F64: 28,
};

const QK8_0 = 32;
const Q8_0_BLOCK_BYTES = 34; // f16 スケール(2) + int8 x 32

// 型 -> [ブロック要素数, ブロックあたりバイト数]
const TYPE_LAYOUT = {
  [GGML.F32]: [1, 4],
  [GGML.F16]: [1, 2],
  [GGML.F64]: [1, 8],
  [GGML.I8]: [1, 1],
  [GGML.I16]: [1, 2],
  [GGML.I32]: [1, 4],
  [GGML.I64]: [1, 8],
  [GGML.Q8_0]: [QK8_0, Q8_0_BLOCK_BYTES],
};

class Cursor {
  constructor(buffer) {
    this.dv = new DataView(buffer);
    this.u8 = new Uint8Array(buffer);
    this.pos = 0;
    this.dec = new TextDecoder('utf-8');
  }
  u32() { const v = this.dv.getUint32(this.pos, true); this.pos += 4; return v; }
  i32() { const v = this.dv.getInt32(this.pos, true); this.pos += 4; return v; }
  // GGUF の u64 は現実的なサイズしか出てこないので Number で扱う
  u64() {
    const v = this.dv.getBigUint64(this.pos, true); this.pos += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('u64 が大きすぎます');
    return Number(v);
  }
  str() {
    const n = this.u64();
    const s = this.dec.decode(this.u8.subarray(this.pos, this.pos + n));
    this.pos += n;
    return s;
  }
  scalar(t) {
    const d = this.dv;
    let v;
    switch (t) {
      case VT.UINT8: v = d.getUint8(this.pos); this.pos += 1; break;
      case VT.INT8: v = d.getInt8(this.pos); this.pos += 1; break;
      case VT.UINT16: v = d.getUint16(this.pos, true); this.pos += 2; break;
      case VT.INT16: v = d.getInt16(this.pos, true); this.pos += 2; break;
      case VT.UINT32: v = d.getUint32(this.pos, true); this.pos += 4; break;
      case VT.INT32: v = d.getInt32(this.pos, true); this.pos += 4; break;
      case VT.FLOAT32: v = d.getFloat32(this.pos, true); this.pos += 4; break;
      case VT.BOOL: v = d.getUint8(this.pos) !== 0; this.pos += 1; break;
      case VT.UINT64: v = this.u64(); return v;
      case VT.INT64: { const b = d.getBigInt64(this.pos, true); this.pos += 8; return Number(b); }
      case VT.FLOAT64: v = d.getFloat64(this.pos, true); this.pos += 8; break;
      default: throw new Error('未知のスカラ型 ' + t);
    }
    return v;
  }
  value(t) {
    if (t === VT.STRING) return this.str();
    if (t === VT.ARRAY) {
      const et = this.u32();
      const n = this.u64();
      const out = new Array(n);
      for (let i = 0; i < n; i++) out[i] = this.value(et);
      return out;
    }
    return this.scalar(t);
  }
}

const alignUp = (n, a) => Math.ceil(n / a) * a;

/** Q8_0 ブロック列を Float32Array へ逆量子化する */
function dequantizeQ8_0(u8, byteOffset, nElements) {
  const nblocks = nElements / QK8_0;
  const out = new Float32Array(nElements);
  // f16 スケールの読み取りは DataView + 手動変換（getFloat16 は環境依存のため使わない）
  const dv = new DataView(u8.buffer, u8.byteOffset + byteOffset, nblocks * Q8_0_BLOCK_BYTES);
  for (let b = 0; b < nblocks; b++) {
    const base = b * Q8_0_BLOCK_BYTES;
    const d = f16ToF32(dv.getUint16(base, true));
    const o = b * QK8_0;
    for (let j = 0; j < QK8_0; j++) {
      out[o + j] = dv.getInt8(base + 2 + j) * d;
    }
  }
  return out;
}

/** IEEE754 half -> float */
function f16ToF32(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 5.960464477539063e-8;      // subnormal: 2^-24
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (frac + 1024) * Math.pow(2, exp - 25);            // (1+frac/1024)*2^(exp-15)
}

class GGUF {
  constructor(buffer) {
    const c = new Cursor(buffer);
    if (c.u32() !== GGUF_MAGIC) throw new Error('GGUF ファイルではありません');
    this.version = c.u32();
    if (this.version !== 3) throw new Error('未対応の GGUF バージョン: ' + this.version);
    const nTensors = c.u64();
    const nKv = c.u64();

    this.kv = Object.create(null);
    for (let i = 0; i < nKv; i++) {
      const key = c.str();
      this.kv[key] = c.value(c.u32());
    }
    this.alignment = this.kv['general.alignment'] ?? 32;

    this.tensorInfo = Object.create(null);
    for (let i = 0; i < nTensors; i++) {
      const name = c.str();
      const nd = c.u32();
      const dims = [];
      for (let k = 0; k < nd; k++) dims.push(c.u64());
      const type = c.u32();
      const offset = c.u64();
      this.tensorInfo[name] = { dims, type, offset };
    }
    this.dataStart = alignUp(c.pos, this.alignment);
    this.u8 = c.u8;
    this._cache = Object.create(null);
  }

  static fromArrayBuffer(buf) { return new GGUF(buf); }

  static async fromURL(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GGUF の取得に失敗: ${res.status} ${url}`);
    return new GGUF(await res.arrayBuffer());
  }

  has(name) { return name in this.tensorInfo; }

  /** テンソルを TypedArray で返す。Q8_0 は Float32Array に逆量子化される。 */
  tensor(name) {
    if (this._cache[name]) return this._cache[name];
    const info = this.tensorInfo[name];
    if (!info) throw new Error('テンソルがありません: ' + name);
    const n = info.dims.reduce((a, b) => a * b, 1);
    const off = this.dataStart + info.offset;
    let arr;
    switch (info.type) {
      case GGML.Q8_0: arr = dequantizeQ8_0(this.u8, off, n); break;
      case GGML.F32: arr = new Float32Array(sliceCopy(this.u8, off, n * 4).buffer); break;
      case GGML.F64: arr = new Float64Array(sliceCopy(this.u8, off, n * 8).buffer); break;
      case GGML.I8: arr = new Int8Array(sliceCopy(this.u8, off, n).buffer); break;
      case GGML.I16: arr = new Int16Array(sliceCopy(this.u8, off, n * 2).buffer); break;
      case GGML.I32: arr = new Int32Array(sliceCopy(this.u8, off, n * 4).buffer); break;
      case GGML.I64: arr = new BigInt64Array(sliceCopy(this.u8, off, n * 8).buffer); break;
      case GGML.F16: {
        const dv = new DataView(this.u8.buffer, this.u8.byteOffset + off, n * 2);
        arr = new Float32Array(n);
        for (let i = 0; i < n; i++) arr[i] = f16ToF32(dv.getUint16(i * 2, true));
        break;
      }
      default: throw new Error('未対応のテンソル型 ' + info.type);
    }
    this._cache[name] = arr;
    return arr;
  }

  /** デバッグ用の一覧 */
  summary() {
    return Object.entries(this.tensorInfo).map(([name, i]) =>
      `${name}\t[${i.dims.join('x')}]\ttype=${i.type}`).join('\n');
  }
}

// TypedArray のアライメント要件を満たすためコピーして返す
function sliceCopy(u8, off, len) {
  return new Uint8Array(u8.slice(off, off + len));
}

export { GGUF, dequantizeQ8_0, f16ToF32, GGML, VT };
export default GGUF;
