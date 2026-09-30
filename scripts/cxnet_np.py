#!/usr/bin/env python3
"""web/cxnet.js と同じ力学の numpy 実装。

ブラウザ側は推論だけを担当し、学習・解析は Python でやる、という分担のため。
**JS 版と数値が一致していることは tests/test_cxnet_np.py で検証している**
（食い違うと、学習した読み出しをブラウザへ持っていった瞬間に壊れる）。

  tau_i dr_i/dt = -r_i + f( g_i Σ_j W_ij r_j + e_i E + b_i + I_i )
  f(x) = (x-θ)/(1+(x-θ))  for x>θ else 0
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf import GGUFReader  # noqa: E402

TAU = 2 * np.pi

ROLE = {
    'OTHER': 0, 'EPG': 1, 'EL': 2, 'PEN': 3, 'PEG': 4, 'D7': 5, 'ER': 6, 'ExR': 7,
    'PFL1': 8, 'PFL2': 9, 'PFL3': 10, 'PFN': 11, 'HDELTA': 12, 'VDELTA': 13,
    'FC': 14, 'FR': 15, 'FS': 16, 'PFR': 17, 'LNO': 18, 'SPSP': 19,
}
ROLE_NAMES = {v: k for k, v in ROLE.items()}


def act(x, thr, kind='sat', scale=1.0):
    """発火率関数。既定 (kind='sat', scale=1) は従来どおり y/(1+y)。

    kind='sat'  : f(y) = y / (1 + y/s)     f'(y) = 1 / (1 + y/s)²
        s は飽和の大きさ。s=1 が従来のモデル、s→∞ で ReLU に近づく。
        **s を上げるほど、強く駆動したときの微分の落ち方が緩くなる。**
        再帰ループの利得 |λ| が 0.43 で頭打ちになるのは s=1 のときの性質なので、
        ここを動かせば積分器が成立するかどうかを直接試せる。
    kind='clip' : f(y) = min(y, s)         f'(y) = 1 (y<s), 0 (y>s)
        発火率の上限を保ったまま、上限までは微分が 1 のままになる版。
    """
    y = np.maximum(x - thr, 0.0)
    if kind == 'clip':
        return np.minimum(y, scale)
    return y / (1.0 + y / scale)


def act_prime(x, thr, kind='sat', scale=1.0):
    """act の導関数。線形化して再帰利得 |λ| を測るのに使う。"""
    y = x - thr
    if kind == 'clip':
        return np.where((y > 0) & (y < scale), 1.0, 0.0)
    return np.where(y > 0, 1.0 / (1.0 + y / scale) ** 2, 0.0)


class CXNetworkNP:
    def __init__(self, path):
        g = GGUFReader(path)
        self.kv = g.kv
        self.N = int(g.kv['cx.neuron_count'])
        indptr = np.asarray(g.tensor('cx.indptr'), dtype=np.int64)
        indices = np.asarray(g.tensor('cx.indices'), dtype=np.int64)
        weights = np.asarray(g.tensor('cx.weights'), dtype=np.float64)
        nnz = int(indptr[self.N])
        # GGUF では nnz を 32 の倍数に詰めてあるので、実データぶんだけ使う
        self.W = sp.csr_matrix(
            (weights[:nnz], indices[:nnz], indptr), shape=(self.N, self.N))
        self.row_gain = np.asarray(g.tensor('cx.row_gain'), dtype=np.float64)
        self.tau = np.asarray(g.tensor('cx.tau'), dtype=np.float64)
        self.bias = np.asarray(g.tensor('cx.bias'), dtype=np.float64)
        self.ext_coef = np.asarray(g.tensor('cx.ext_coef'), dtype=np.float64)
        self.role = np.asarray(g.tensor('cx.role'), dtype=np.int64)
        self.side = np.asarray(g.tensor('cx.side'), dtype=np.int64)
        self.pb_side = (np.asarray(g.tensor('cx.pb_side'), dtype=np.int64)
                        if 'cx.pb_side' in g.tensors else self.side.copy())
        self.phase = np.asarray(g.tensor('cx.phase'), dtype=np.float64)
        self.type_id = np.asarray(g.tensor('cx.type_id'), dtype=np.int64)
        self.type_names = list(g.kv.get('cx.type_names', []))
        self.body_id = (np.asarray(g.tensor('cx.body_id'), dtype=np.int64)
                        if 'cx.body_id' in g.tensors else None)

        self.excitability = float(g.kv.get('cx.excitability', 1.0))
        self.global_inhibition = float(g.kv.get('cx.global_inhibition', 0.0))
        self.threshold = float(g.kv.get('cx.threshold', 0.15))
        self.external_drive = float(g.kv.get('cx.external_drive', 0.0))
        self.er_baseline = float(g.kv.get('cx.er_baseline', 0.08))
        self.visual_cue_gain = float(g.kv.get('cx.visual_cue_gain', 0.8))

        # 発火率関数。既定は従来モデル。scripts/activation_sweep.py で振る。
        self.act_kind = 'sat'
        self.act_scale = 1.0
        # ゲイン安定化のフック。None なら何もしない（既定の力学は変わらない）。
        # 呼び出しは stabilizer(net, syn, dt) → syn。scripts/pi_benchmark.py で使う。
        self.stabilizer = None

        # 分流（シャント）抑制。6.10 節で宣言して入れる機構。
        #   shunt_frac = 0 なら抑制はすべて減算（＝これまでのモデルと完全に同じ）
        #   shunt_frac = 1 なら抑制はすべて除算（塩化物チャネルの分流抑制）
        # 途中の値は減算と除算の混合。shunt_gain は除算の強さ。
        # 既定は 0.0 で、力学はビット単位で従来と一致する（tests/test_pi.py で固定）。
        self.shunt_frac = 0.0
        self.shunt_gain = 1.0
        self._Wpos = None
        self._Wneg = None

        self.r = np.zeros(self.N)
        self.drive = np.zeros(self.N)
        self.alive = np.ones(self.N, dtype=bool)

        # 入力層（JS と同じ）: ER / ExR / SpsP / LNO はクランプする
        self.clamped = np.zeros(self.N, dtype=bool)
        for rl in (ROLE['ER'], ROLE['ExR'], ROLE['SPSP'], ROLE['LNO']):
            self.clamped |= (self.role == rl)
        self.clamp_val = np.zeros(self.N)

        self._cue_theta = 0.0
        self._cue_gain = 0.0

        # よく使う集団
        self.by_role = {k: np.where(self.role == v)[0] for k, v in ROLE.items()}
        self.epg = self._with_phase(ROLE['EPG'])
        self.hdelta = self.by_role['HDELTA']
        self.vdelta = self.by_role['VDELTA']
        self.pfn = self.by_role['PFN']
        self.pfl3_l = self._role_side(ROLE['PFL3'], -1)
        self.pfl3_r = self._role_side(ROLE['PFL3'], +1)
        self.lno_left = self._role_side(ROLE['LNO'], -1)
        self.lno_right = self._role_side(ROLE['LNO'], +1)
        self.spsp_left = self._role_side(ROLE['SPSP'], -1)
        self.spsp_right = self._role_side(ROLE['SPSP'], +1)
        # 目標を注入する先。**既定は従来どおり FC2 と hDeltaB の両方**で、
        # 力学は 1 ビットも変わらない。研究計画の実験 0（docs/RESEARCH_PLAN_SERVO.md）で
        # 「hΔB にも目標を書き込んでいた」ことが交絡として見つかったので、
        # set_goal_types() で差し替えられるようにだけしてある。
        self.goal_types = ['FC2', 'hDeltaB']
        self.goal_cells = self._by_type_prefix(self.goal_types)
        self.reset()

    # --------------------------------------------------------------
    def _with_phase(self, role):
        idx = np.where(self.role == role)[0]
        return idx[self.phase[idx] > -8]

    def _role_side(self, role, sd):
        idx = np.where(self.role == role)[0]
        return idx[self.side[idx] == sd]

    def _by_type_prefix(self, prefixes):
        out = []
        for i in range(self.N):
            if self.phase[i] <= -8:
                continue
            t = self.type_names[self.type_id[i]] if self.type_id[i] < len(self.type_names) else ''
            if any(t.startswith(p) for p in prefixes):
                out.append(i)
        return np.array(out, dtype=np.int64)

    def type_of(self, i):
        return self.type_names[self.type_id[i]] if self.type_id[i] < len(self.type_names) else ''

    def types_matching(self, pattern):
        rx = re.compile(pattern)
        return np.array([i for i in range(self.N) if rx.match(self.type_of(i))], dtype=np.int64)

    # --------------------------------------------------------------
    def reset(self, seed=1):
        s = np.uint32(seed if seed else 1)
        self.r[:] = 0.0
        # JS 版と同じ線形合同法でノイズを作る（初期値まで一致させる）
        for i in self.epg:
            s = np.uint32((np.uint64(s) * np.uint64(1664525) + np.uint64(1013904223)) % np.uint64(2**32))
            self.r[i] = 0.05 + 0.05 * (float(s) / 4294967296.0)
        self.drive[:] = 0.0

    def clear_drive(self):
        self.drive[:] = 0.0

    def set_visual_scene(self, theta, clarity=1.0):
        c = min(max(clarity, 0.0), 1.0)
        self.clamp_val[self.clamped] = self.er_baseline
        self._cue_theta = theta
        self._cue_gain = self.visual_cue_gain * c

    def set_goal_types(self, prefixes):
        """目標を注入する細胞型を差し替える（実験 0 用）。

        既定の ['FC2', 'hDeltaB'] を渡せば、何も変わらない。
        """
        self.goal_types = list(prefixes)
        self.goal_cells = self._by_type_prefix(self.goal_types)
        return len(self.goal_cells)

    def inject_goal(self, theta, gain=1.0):
        th = theta + float(self.kv.get('cx.goal_null_offset', 0.0))
        d = np.cos(self.phase[self.goal_cells] - th)
        self.drive[self.goal_cells] += gain * np.maximum(0, d) ** 2

    def _apply_visual_cue(self):
        if self._cue_gain <= 0:
            return
        c = np.maximum(0.0, np.cos(self.phase[self.epg] - self._cue_theta))
        self.drive[self.epg] += self._cue_gain * c * c

    def _split_weights(self):
        """興奮性と抑制性を分けた行列を用意する（分流抑制でのみ使う）。"""
        if self._Wpos is None or self._Wpos.shape != self.W.shape:
            W = self.W.tocsr()
            pos = W.copy(); pos.data = np.maximum(pos.data, 0.0); pos.eliminate_zeros()
            neg = W.copy(); neg.data = np.maximum(-neg.data, 0.0); neg.eliminate_zeros()
            self._Wpos, self._Wneg = pos, neg
        return self._Wpos, self._Wneg

    def invalidate_weight_split(self):
        """W を差し替えたら呼ぶ（分流抑制用のキャッシュを捨てる）。"""
        self._Wpos = self._Wneg = None

    def step(self, dt):
        self._apply_visual_cue()
        gi = self.global_inhibition * self.r.mean()
        if self.shunt_frac <= 0.0:
            syn = (self.W @ self.r) * self.row_gain * self.excitability \
                + self.ext_coef * self.external_drive + self.bias + self.drive - gi
        else:
            # 抑制を減算と除算に分ける。f = 0 のとき上の式と厳密に一致する。
            Wpos, Wneg = self._split_weights()
            scale = self.row_gain * self.excitability
            exc = (Wpos @ self.r) * scale
            inh = (Wneg @ self.r) * scale
            ext = self.ext_coef * self.external_drive
            exc_total = exc + np.maximum(ext, 0.0) + self.bias + self.drive
            inh_total = inh + np.maximum(-ext, 0.0) + gi
            f = self.shunt_frac
            syn = (exc_total - (1.0 - f) * inh_total) \
                / (1.0 + f * self.shunt_gain * np.maximum(inh_total, 0.0))
        if self.stabilizer is not None:
            syn = self.stabilizer(self, syn, dt)
        target = act(syn, self.threshold, self.act_kind, self.act_scale)
        k = np.minimum(1.0, dt / self.tau)
        new = self.r + k * (target - self.r)
        new[self.clamped] = self.clamp_val[self.clamped]
        new[~self.alive] = 0.0
        self.r = new

    # --------------------------------------------------------------
    def read_heading(self):
        a = self.r[self.epg]
        x = float((a * np.cos(self.phase[self.epg])).sum())
        y = float((a * np.sin(self.phase[self.epg])).sum())
        s = float(a.sum())
        mag = np.hypot(x, y)
        return {'theta': float(np.arctan2(y, x)),
                'strength': float(mag / s) if s > 1e-6 else 0.0}

    def read_steering(self):
        l = float(self.r[self.pfl3_l].sum())
        rr = float(self.r[self.pfl3_r].sum())
        tot = l + rr
        return {'left': l, 'right': rr, 'turn': (rr - l) / tot if tot > 1e-6 else 0.0}
