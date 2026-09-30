"""
最小限だが仕様準拠の GGUF v3 リーダ / ライタ。

GGUF は llama.cpp 由来の「メタデータ付きテンソル容器」フォーマットであり、
Transformer 専用ではない。ここではハエのコネクトーム（疎な結合行列）を
Q8_0 量子化して GGUF に格納するために使う。

参照仕様: ggml/docs/gguf.md (GGUF v3)

レイアウト:
    magic      : "GGUF" (4 bytes)
    version    : u32 = 3
    n_tensors  : u64
    n_kv       : u64
    kv[]       : key(gguf_string) + value_type(u32) + value
    tensor_info[]: name(gguf_string) + n_dims(u32) + dims(u64 * n_dims)
                   + ggml_type(u32) + offset(u64)
    <padding to general.alignment>
    tensor_data[]  (各テンソルは alignment 境界に配置)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# ---- gguf_metadata_value_type ----
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_FMT = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<?",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}

# ---- ggml_type (テンソル要素型) ----
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28

QK8_0 = 32  # Q8_0 のブロックサイズ
Q8_0_BLOCK_BYTES = 2 + QK8_0  # f16 スケール + int8 x 32

# (block_size, bytes_per_block)
_TYPE_LAYOUT = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_F64: (1, 8),
    GGML_TYPE_I8: (1, 1),
    GGML_TYPE_I16: (1, 2),
    GGML_TYPE_I32: (1, 4),
    GGML_TYPE_I64: (1, 8),
    GGML_TYPE_Q8_0: (QK8_0, Q8_0_BLOCK_BYTES),
}

TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_I8: "I8",
    GGML_TYPE_I16: "I16",
    GGML_TYPE_I32: "I32",
    GGML_TYPE_I64: "I64",
    GGML_TYPE_F64: "F64",
}


# --------------------------------------------------------------------------
# Q8_0 量子化
# --------------------------------------------------------------------------
def quantize_q8_0(x: np.ndarray) -> bytes:
    """float32 配列を Q8_0 (32要素ごとに f16 スケール + int8) へ量子化する。

    要素数は 32 の倍数でなければならない。
    """
    x = np.ascontiguousarray(x, dtype=np.float32).ravel()
    if x.size % QK8_0 != 0:
        raise ValueError(f"Q8_0 は 32 の倍数長が必要 (got {x.size})")
    blocks = x.reshape(-1, QK8_0)

    amax = np.abs(blocks).max(axis=1)
    # d を f16 に丸めてからスケールする。逆量子化側と完全に一致させるため。
    d = (amax / 127.0).astype(np.float16)
    d_f32 = d.astype(np.float32)
    inv = np.where(d_f32 > 0, 1.0 / np.where(d_f32 == 0, 1.0, d_f32), 0.0)

    q = np.rint(blocks * inv[:, None]).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)

    out = np.empty((blocks.shape[0], Q8_0_BLOCK_BYTES), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def dequantize_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    """quantize_q8_0 の逆変換。"""
    nblocks = n_elements // QK8_0
    buf = np.frombuffer(raw, dtype=np.uint8, count=nblocks * Q8_0_BLOCK_BYTES)
    buf = buf.reshape(nblocks, Q8_0_BLOCK_BYTES)
    d = buf[:, 0:2].copy().view(np.float16).reshape(-1).astype(np.float32)
    q = buf[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d[:, None]).reshape(-1)[:n_elements]


# --------------------------------------------------------------------------
# ライタ
# --------------------------------------------------------------------------
@dataclass
class _Tensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    data: bytes


@dataclass
class GGUFWriter:
    alignment: int = DEFAULT_ALIGNMENT
    _kv: list[tuple[str, int, Any]] = field(default_factory=list)
    _tensors: list[_Tensor] = field(default_factory=list)

    # ---- メタデータ ----
    def add_string(self, key: str, value: str) -> None:
        self._kv.append((key, GGUF_TYPE_STRING, value))

    def add_uint32(self, key: str, value: int) -> None:
        self._kv.append((key, GGUF_TYPE_UINT32, int(value)))

    def add_float32(self, key: str, value: float) -> None:
        self._kv.append((key, GGUF_TYPE_FLOAT32, float(value)))

    def add_bool(self, key: str, value: bool) -> None:
        self._kv.append((key, GGUF_TYPE_BOOL, bool(value)))

    def add_string_array(self, key: str, values: list[str]) -> None:
        self._kv.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, list(values))))

    def add_int32_array(self, key: str, values) -> None:
        vals = [int(v) for v in values]
        self._kv.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, vals)))

    def add_float32_array(self, key: str, values) -> None:
        vals = [float(v) for v in values]
        self._kv.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, vals)))

    # ---- テンソル ----
    def add_tensor(self, name: str, array: np.ndarray, ggml_type: int | None = None) -> None:
        arr = np.ascontiguousarray(array)
        if ggml_type is None:
            ggml_type = {
                np.dtype(np.float32): GGML_TYPE_F32,
                np.dtype(np.float16): GGML_TYPE_F16,
                np.dtype(np.int8): GGML_TYPE_I8,
                np.dtype(np.int16): GGML_TYPE_I16,
                np.dtype(np.int32): GGML_TYPE_I32,
                np.dtype(np.int64): GGML_TYPE_I64,
            }.get(arr.dtype)
            if ggml_type is None:
                raise ValueError(f"未対応の dtype: {arr.dtype}")

        if ggml_type == GGML_TYPE_Q8_0:
            data = quantize_q8_0(arr)
        else:
            data = arr.tobytes()

        # GGUF の dims は ne[] (dim0 が最内側)。numpy の shape とは逆順。
        dims = tuple(int(d) for d in reversed(arr.shape)) or (1,)
        self._tensors.append(_Tensor(name, dims, ggml_type, data))

    # ---- 書き出し ----
    @staticmethod
    def _pack_string(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _pack_value(self, vtype: int, value: Any) -> bytes:
        if vtype == GGUF_TYPE_STRING:
            return self._pack_string(value)
        if vtype == GGUF_TYPE_ARRAY:
            elem_type, items = value
            out = struct.pack("<IQ", elem_type, len(items))
            for it in items:
                out += self._pack_value(elem_type, it)
            return out
        return struct.pack(_SCALAR_FMT[vtype], value)

    def write(self, path: str) -> None:
        kv = list(self._kv)
        kv.append(("general.alignment", GGUF_TYPE_UINT32, self.alignment))

        header = bytearray()
        header += GGUF_MAGIC
        header += struct.pack("<I", GGUF_VERSION)
        header += struct.pack("<Q", len(self._tensors))
        header += struct.pack("<Q", len(kv))
        for key, vtype, value in kv:
            header += self._pack_string(key)
            header += struct.pack("<I", vtype)
            header += self._pack_value(vtype, value)

        # テンソルデータのオフセットを決める（データ領域先頭からの相対値）
        offsets: list[int] = []
        cursor = 0
        for t in self._tensors:
            offsets.append(cursor)
            cursor += len(t.data)
            cursor = _align_up(cursor, self.alignment)

        for t, off in zip(self._tensors, offsets):
            header += self._pack_string(t.name)
            header += struct.pack("<I", len(t.dims))
            for d in t.dims:
                header += struct.pack("<Q", d)
            header += struct.pack("<I", t.ggml_type)
            header += struct.pack("<Q", off)

        pad = _align_up(len(header), self.alignment) - len(header)
        header += b"\x00" * pad

        with open(path, "wb") as f:
            f.write(bytes(header))
            written = 0
            for t in self._tensors:
                f.write(t.data)
                written += len(t.data)
                target = _align_up(written, self.alignment)
                f.write(b"\x00" * (target - written))
                written = target


def _align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


# --------------------------------------------------------------------------
# リーダ（テスト・検証用）
# --------------------------------------------------------------------------
class GGUFReader:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.buf = f.read()
        self.pos = 0
        if self._raw(4) != GGUF_MAGIC:
            raise ValueError("GGUF マジックが不正")
        self.version = self._u32()
        if self.version != GGUF_VERSION:
            raise ValueError(f"未対応の GGUF バージョン: {self.version}")
        n_tensors = self._u64()
        n_kv = self._u64()

        self.kv: dict[str, Any] = {}
        for _ in range(n_kv):
            key = self._str()
            self.kv[key] = self._value(self._u32())

        self.alignment = self.kv.get("general.alignment", DEFAULT_ALIGNMENT)

        self.tensors: dict[str, dict[str, Any]] = {}
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u32()
            dims = tuple(self._u64() for _ in range(nd))
            ttype = self._u32()
            off = self._u64()
            self.tensors[name] = {"dims": dims, "type": ttype, "offset": off}

        self.data_start = _align_up(self.pos, self.alignment)

    # -- primitives --
    def _raw(self, n: int) -> bytes:
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def _u32(self) -> int:
        return struct.unpack("<I", self._raw(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self._raw(8))[0]

    def _str(self) -> str:
        return self._raw(self._u64()).decode("utf-8")

    def _value(self, vtype: int) -> Any:
        if vtype == GGUF_TYPE_STRING:
            return self._str()
        if vtype == GGUF_TYPE_ARRAY:
            et = self._u32()
            n = self._u64()
            return [self._value(et) for _ in range(n)]
        fmt = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, self._raw(struct.calcsize(fmt)))[0]

    # -- tensors --
    def tensor(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        dims = info["dims"]
        n = 1
        for d in dims:
            n *= d
        blk, nbytes = _TYPE_LAYOUT[info["type"]]
        raw = self.buf[
            self.data_start + info["offset"] : self.data_start + info["offset"] + (n // blk) * nbytes
        ]
        t = info["type"]
        if t == GGML_TYPE_Q8_0:
            arr = dequantize_q8_0(raw, n)
        else:
            np_dtype = {
                GGML_TYPE_F32: np.float32,
                GGML_TYPE_F16: np.float16,
                GGML_TYPE_F64: np.float64,
                GGML_TYPE_I8: np.int8,
                GGML_TYPE_I16: np.int16,
                GGML_TYPE_I32: np.int32,
                GGML_TYPE_I64: np.int64,
            }[t]
            arr = np.frombuffer(raw, dtype=np_dtype).copy()
        # numpy 形状は GGUF dims の逆順
        return arr.reshape(tuple(reversed(dims)))
