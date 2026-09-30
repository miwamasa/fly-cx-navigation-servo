// cxnet.js — GGUF に入ったハエ中心複合体コネクトームを回すレートベース推論エンジン
//
// モデル:
//   tau_i * dr_i/dt = -r_i + f( g_i * sum_j W_ij r_j + b_i + I_i )
//   f(x) = x <= 0 ? 0 : x / (1 + x)      (0..1 に飽和する半波整流)
//
// W_ij は「シナプス数 x 前シナプス神経伝達物質の符号」を行ごとに正規化したもので、
// Q8_0 量子化されて GGUF に格納されている。g_i (row_gain) が物理スケールを戻す。

export const ROLE = {
  OTHER: 0, EPG: 1, EL: 2, PEN: 3, PEG: 4, D7: 5, ER: 6, ExR: 7,
  PFL1: 8, PFL2: 9, PFL3: 10, PFN: 11, HDELTA: 12, VDELTA: 13,
  FC: 14, FR: 15, FS: 16, PFR: 17, LNO: 18, SPSP: 19,
};

export const ROLE_NAMES = Object.fromEntries(
  Object.entries(ROLE).map(([k, v]) => [v, k])
);

const TWO_PI = Math.PI * 2;

/** 閾値つき・0..1 に飽和する半波整流 */
function act(x, thr) {
  const y = x - thr;
  return y <= 0 ? 0 : y / (1 + y);
}

export class CXNetwork {
  constructor(gguf) {
    const t = (n) => gguf.tensor(n);
    this.gguf = gguf;
    this.meta = gguf.kv;

    this.N = gguf.kv['cx.neuron_count'];
    this.indptr = t('cx.indptr');
    this.indices = t('cx.indices');
    this.weights = t('cx.weights');       // Q8_0 -> Float32Array
    this.rowGain = t('cx.row_gain');
    this.tau = t('cx.tau');
    this.bias = t('cx.bias');
    this.role = t('cx.role');
    this.side = t('cx.side');
    this.phase = t('cx.phase');           // ラジアン。無効値は NaN ではなく -9 を使う
    this.typeId = t('cx.type_id');
    this.typeNames = gguf.kv['cx.type_names'] || [];
    this.bodyId = gguf.has('cx.body_id') ? t('cx.body_id') : null;

    // 量子化・剪定をやり直せるよう、元の重みを取っておく
    this._w0 = Float32Array.from(this.weights);
    this.alive = new Uint8Array(this.N).fill(1);
    this.bits = 8;
    this.pruneThreshold = 0;

    this.r = new Float32Array(this.N);
    this.drive = new Float32Array(this.N);
    this._syn = new Float32Array(this.N);

    // コネクトームには含まれない大域的な抑制（局所回路の未モデル分と
    // 全体の発火率制御）を 1 パラメータで補う。リングアトラクタが
    // 「局所興奮 + 大域抑制」で成り立つための必須項。
    this.excitability = this.meta['cx.excitability'] ?? 1.0;
    this.globalInhibition = this.meta['cx.global_inhibition'] ?? 0.0;
    this.threshold = this.meta['cx.threshold'] ?? 0.15;
    // 中心複合体を脳から切り出したことで失われた「外部からの入力」を
    // 定常バイアスとして戻す。cx.ext_coef は各細胞の全入力に占める外部入力の
    // 符号つき比率で、externalDrive はその相手側の平均発火率にあたる。
    this.externalDrive = this.meta['cx.external_drive'] ?? 0.3;
    this.extCoef = gguf.has('cx.ext_coef') ? t('cx.ext_coef') : new Float32Array(this.N);

    // 役割ごとのインデックス表を事前計算しておく（毎フレーム走査しないため）
    this.byRole = new Map();
    for (let i = 0; i < this.N; i++) {
      const rl = this.role[i];
      if (!this.byRole.has(rl)) this.byRole.set(rl, []);
      this.byRole.get(rl).push(i);
    }
    this.epg = this._roleWithPhase(ROLE.EPG);
    this.er = this._roleWithPhase(ROLE.ER);
    this.penL = this._roleSide(ROLE.PEN, -1);
    this.penR = this._roleSide(ROLE.PEN, +1);
    this.pfl3L = this._roleSide(ROLE.PFL3, -1);
    this.pfl3R = this._roleSide(ROLE.PFL3, +1);
    // 目標方位を保持する集団。FC2 と hDeltaB に限る（Westeinde et al. 2024）。
    // FC/hDelta 全体を混ぜると型ごとに位相基準が異なり、左右の対称性が壊れる。
    this.goalCells = this._byTypePrefix(['FC2', 'hDeltaB']);
    // コネクトーム実測の PFL3 位相オフセット（左 -73deg / 右 +73deg）。
    // null は左右が釣り合う＝直進する目標オフセット。
    this.pfl3OffsetLeft = this.meta['cx.pfl3_offset_left'] ?? 0;
    this.pfl3OffsetRight = this.meta['cx.pfl3_offset_right'] ?? 0;
    this.goalNullOffset = this.meta['cx.goal_null_offset'] ?? 0;

    // --- 入力層 ---
    // ER / ExR は中心複合体の *外* (視覚系) から情報を運び込む入力ニューロンで、
    // EPG への入力シナプスの約 6 割を占める最大の抑制源。回路内部のダイナミクスで
    // 動かすのではなく、外界の状態から直接クランプするのが正しい扱いになる。
    // SpsP / LNO は自己運動（角速度・並進）を運ぶ入力。
    this.clamped = new Uint8Array(this.N);
    this.clampVal = new Float32Array(this.N);
    for (const rl of [ROLE.ER, ROLE.ExR, ROLE.SPSP, ROLE.LNO]) {
      for (const i of this.byRole.get(rl) || []) this.clamped[i] = 1;
    }
    this.erBaseline = this.meta['cx.er_baseline'] ?? 0.08;  // ER の一様活動（EPG のゲイン制御）
    this.visualCueGain = this.meta['cx.visual_cue_gain'] ?? 0.8;  // 学習済み ER→EPG 地図の強さ
    this._cueTheta = 0;
    this._cueGain = 0;

    // 初期状態: EPG に小さなランダムノイズを入れてバンプの自発形成を促す
    this.reset();
  }

  static fromGGUF(gguf) { return new CXNetwork(gguf); }

  _roleWithPhase(role) {
    return (this.byRole.get(role) || []).filter((i) => this.phase[i] > -8);
  }
  _byTypePrefix(prefixes) {
    const out = [];
    for (let i = 0; i < this.N; i++) {
      if (this.phase[i] <= -8) continue;
      const t = this.typeNames[this.typeId[i]] || '';
      if (prefixes.some((p) => t.startsWith(p))) out.push(i);
    }
    return out;
  }
  _roleSide(role, side) {
    return (this.byRole.get(role) || []).filter((i) => this.side[i] === side);
  }

  reset(seed = 1) {
    let s = seed >>> 0 || 1;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    this.r.fill(0);
    for (const i of this.epg) if (this.alive[i]) this.r[i] = 0.05 + 0.05 * rnd();
    this.drive.fill(0);
  }

  // ------------------------------------------------------------------
  // 「最小の脳」用: 細胞の除去・重みの再量子化・サイズ計算
  // ------------------------------------------------------------------

  /** 生存マスクを設定する。死んだ細胞は発火率 0 に固定される。 */
  setAlive(mask) {
    this.alive.set(mask);
    for (let i = 0; i < this.N; i++) if (!this.alive[i]) this.r[i] = 0;
  }

  /** 型名の前方一致で生存マスクを作る（null なら全生存） */
  aliveFromTypes(keepPrefixes) {
    const m = new Uint8Array(this.N);
    for (let i = 0; i < this.N; i++) {
      const t = this.typeNames[this.typeId[i]] || '';
      m[i] = (keepPrefixes === null || keepPrefixes.some((p) => t.startsWith(p))) ? 1 : 0;
    }
    return m;
  }

  /**
   * 重みの情報量を削る。元の重みからやり直すので何度でも呼べる。
   * @param bits  1..8  行内で最大絶対値正規化したうえでの量子化ビット数
   * @param prune 0..1  行内最大値に対する比がこれ未満のシナプスを切る
   */
  applyWeightBudget(bits = 8, prune = 0) {
    this.bits = bits; this.pruneThreshold = prune;
    const w = this.weights, w0 = this._w0, ip = this.indptr;
    const lv = (1 << (bits - 1)) - 1;
    for (let i = 0; i < this.N; i++) {
      const e0 = ip[i], e1 = ip[i + 1];
      let m = 0;
      for (let e = e0; e < e1; e++) {
        const v = Math.abs(w0[e]) < prune ? 0 : w0[e];
        w[e] = v;
        if (Math.abs(v) > m) m = Math.abs(v);
      }
      if (m <= 0 || bits >= 8) continue;
      for (let e = e0; e < e1; e++) w[e] = Math.round(w[e] / m * lv) / lv * m;
    }
  }

  /** 今の脳の大きさ。kbit = 生存シナプス数 × ビット数。 */
  budget() {
    const ip = this.indptr, w = this.weights;
    let edges = 0, neurons = 0;
    for (let i = 0; i < this.N; i++) {
      if (!this.alive[i]) continue;
      neurons++;
      for (let e = ip[i]; e < ip[i + 1]; e++) {
        if (this.alive[this.indices[e]] && w[e] !== 0) edges++;
      }
    }
    return { neurons, edges, bits: this.bits, kbit: edges * this.bits / 1000 };
  }

  /** 外部入力をクリアする（毎フレーム最初に呼ぶ） */
  clearDrive() { this.drive.fill(0); }

  /**
   * 視覚ランドマーク入力。ER(リング神経)の活動を直接セットする。
   *
   * ER は GABA 性で EPG を一様に抑え込んでいる。ランドマークが見えている方位に
   * 対応する ER だけ活動が下がることで、その方位の EPG が「脱抑制」されて
   * バンプが立つ。ER→EPG の位相オフセットはハエでは可塑的に学習される量なので、
   * ここではバンプが実際の方位に一致する向きに合わせてある。
   *
   * @param theta   ランドマークから推定される現在の方位（ラジアン）
   * @param clarity 0 = 見えない（一様抑制）, 1 = はっきり見える
   */
  setVisualScene(theta, clarity = 1.0) {
    const c = Math.max(0, Math.min(1, clarity));
    // ER は EB 上ではほぼ無選択（コネクトーム実測で方位選択性 R の中央値 0.07）。
    // 一様な抑制＝ゲイン制御として働かせる。
    for (let i = 0; i < this.N; i++) {
      if (this.clamped[i]) this.clampVal[i] = this.erBaseline;
    }
    // ランドマーク方位 -> EPG の対応づけは、ハエでは ER→EPG シナプスの
    // 可塑性として学習される量で、配線図には書かれていない。ここだけは
    // モデル側のパラメータとして明示的に与える（学習済みの地図に相当）。
    this._cueTheta = theta;
    this._cueGain = this.visualCueGain * c;
  }

  /** 後方互換: 視覚キュー（内部では setVisualScene を呼ぶ） */
  injectHeadingCue(theta, gain = 1.0) { this.setVisualScene(theta, gain); }

  /** 角速度入力。PEN の左右非対称な駆動がバンプを回す（経路積分）。 */
  injectAngularVelocity(omega, gain = 1.0) {
    const base = 0.35;
    const l = base + gain * omega;
    const rr = base - gain * omega;
    for (const i of this.penL) this.drive[i] += l;
    for (const i of this.penR) this.drive[i] += rr;
  }

  /**
   * 目標方位（アロセントリック）を FC2 / hDeltaB のゴールバンプとして入れる。
   * goalNullOffset を足すことで「theta == 現在方位」のとき左右 PFL3 が釣り合う。
   */
  injectGoal(theta, gain = 1.0) {
    const th = theta + this.goalNullOffset;
    for (const i of this.goalCells) {
      const d = Math.cos(this.phase[i] - th);
      this.drive[i] += gain * Math.max(0, d) ** 2;
    }
  }

  /** 視覚手がかり（学習済みの地図）を EPG に位相選択的に加える。step() が毎回呼ぶ。 */
  _applyVisualCue() {
    if (this._cueGain <= 0) return;
    for (const i of this.epg) {
      const c = Math.max(0, Math.cos(this.phase[i] - this._cueTheta));
      this.drive[i] += this._cueGain * c * c;
    }
  }

  /** 一様な覚醒入力（全体のゲイン） */
  injectTonic(amount, role = null) {
    if (role === null) {
      for (let i = 0; i < this.N; i++) this.drive[i] += amount;
    } else {
      for (const i of this.byRole.get(role) || []) this.drive[i] += amount;
    }
  }

  /** 1 ステップ進める。dt は秒。 */
  step(dt) {
    this._applyVisualCue();
    const { N, indptr, indices, weights, rowGain, bias, tau, r, drive } = this;
    const syn = this._syn;
    let mean = 0;
    for (let i = 0; i < N; i++) mean += r[i];
    mean /= N;
    const gi = this.globalInhibition * mean;
    const ex = this.excitability;
    for (let i = 0; i < N; i++) {
      let s = 0;
      const e0 = indptr[i], e1 = indptr[i + 1];
      for (let e = e0; e < e1; e++) s += weights[e] * r[indices[e]];
      syn[i] = s * rowGain[i] * ex + this.extCoef[i] * this.externalDrive
             + bias[i] + drive[i] - gi;
    }
    const thr = this.threshold;
    const clamped = this.clamped, clampVal = this.clampVal, alive = this.alive;
    for (let i = 0; i < N; i++) {
      if (!alive[i]) { r[i] = 0; continue; }
      if (clamped[i]) { r[i] = clampVal[i]; continue; }
      const target = act(syn[i], thr);
      const k = Math.min(1, dt / tau[i]);
      r[i] += k * (target - r[i]);
    }
  }

  /** dt を細かく刻んで複数ステップ回す */
  advance(dtTotal, subSteps = 2) {
    const dt = dtTotal / subSteps;
    for (let k = 0; k < subSteps; k++) this.step(dt);
  }

  /** 集団ベクトルで EPG バンプの位相を読む。{ theta, strength } */
  readHeading() {
    let x = 0, y = 0, sum = 0;
    for (const i of this.epg) {
      const a = this.r[i];
      x += a * Math.cos(this.phase[i]);
      y += a * Math.sin(this.phase[i]);
      sum += a;
    }
    const mag = Math.hypot(x, y);
    return { theta: Math.atan2(y, x), strength: sum > 1e-6 ? mag / sum : 0, sum };
  }

  /** PFL3 の左右差から操舵指令を読む。正=右旋回。 */
  readSteering() {
    let l = 0, rr = 0;
    for (const i of this.pfl3L) l += this.r[i];
    for (const i of this.pfl3R) rr += this.r[i];
    const tot = l + rr;
    return { left: l, right: rr, turn: tot > 1e-6 ? (rr - l) / tot : 0 };
  }

  /** 可視化用: 方位ビンごとの EPG 活動 */
  ringProfile(bins = 16) {
    const acc = new Float32Array(bins);
    const cnt = new Float32Array(bins);
    for (const i of this.epg) {
      let p = this.phase[i] % TWO_PI;
      if (p < 0) p += TWO_PI;
      const b = Math.min(bins - 1, Math.floor((p / TWO_PI) * bins));
      acc[b] += this.r[i];
      cnt[b] += 1;
    }
    for (let b = 0; b < bins; b++) if (cnt[b] > 0) acc[b] /= cnt[b];
    return acc;
  }

  stats() {
    let active = 0, sum = 0;
    for (let i = 0; i < this.N; i++) { sum += this.r[i]; if (this.r[i] > 0.05) active++; }
    return { neurons: this.N, edges: this.indptr[this.N], active, meanRate: sum / this.N };
  }
}

export default CXNetwork;
