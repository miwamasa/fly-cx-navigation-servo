#!/usr/bin/env python3
"""B2/B3（第 5 試験）: hΔB は R(θ)·v_body を計算しているか。

  python3 scripts/coord_transform.py
  → data/coord_transform.json

問いの立て直し
--------------
6.7〜6.8 節は経路積分（時間積分）を測って陰性だった。しかし文献が
PFNd/PFNv → hΔB に与えている役割は積分ではなく **座標変換** である。

    身体座標の速度 v_body（速さ |v|、体軸から見た進行方向 φ）
  ＋ 世界に対する頭方位 θ
  → 世界座標の速度 R(θ)·v_body（向き θ+φ、大きさ |v|）

積分器として期待する前に、**瞬時の座標変換として成立しているか**を測る。
これは時間積分を含まないので、6.7 節の 4 試験とは独立な試験になる。

試験前に決めた合格基準
----------------------
θ と φ を独立に振り、hΔB 集団ベクトルの位相 ψ を円環回帰
`ψ = a·θ + b·φ + c` で当てはめる。

  a（方位の利得）      [0.8, 1.2]   … 世界座標になっている
  b（進行方向の利得）  [0.8, 1.2]   … 身体座標の向きを取り込んでいる
  残差の集中度 R       ≥ 0.8        … 当てはまりが良い
  振幅の線形性 R²      ≥ 0.9        … 長さが速さに比例する

入力モデル（B2）
----------------
6.7 節までの自己運動入力は左右の視覚流 `v·cos(β∓45°)` を LNO と SpsP に
同じ形で入れており、PFNd と PFNv を区別していなかった。ここでは配線の実測
（scripts/pfn_basis.py）に合わせて 4 集団を別々に駆動する。

    PFNd-R ← LNO2-L, SpsP-R      PFNd-L ← LNO2-R, SpsP-L
    PFNv-R ← LNO1-L              PFNv-L ← LNO1-R

各集団に身体座標の選好方向 ψ_g を与え、`max(0, cos(φ − ψ_g))·|v|` で駆動する。
ψ_g は **文献から取ったモデルパラメータ**であって配線から導いたものではない
（PFNd は前方 ±45°、PFNv は後方 ±135°）。配線から導くと循環論法になる。

符号の仮定（scripts/pfn_operating.py で分かったこと）
----------------------------------------------------
自己運動の入口は実測ではすべて抑制性で、その動作点では PFNd は歩行中に沈黙し
PFNv は一切発火しない。そこで 2 条件を併走させる。

  measured    実測の符号のまま（＝変換段が動いていない状態）
  excitatory  自己運動の入口を興奮性にする（PFNd R=0.99 / PFNv R=0.95 で
              方位同調と歩行中の活動が両立する唯一の設定。ただし LNO1 の
              ground truth は GABA なので、**実測に反する仮定**である）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from cxnet_np import CXNetworkNP                       # noqa: E402
from pfn_operating import flip_signs                   # noqa: E402
from pi_task import bootstrap_ci                       # noqa: E402

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
OUT = os.path.join(ROOT, 'data', 'coord_transform.json')
DT = 0.02
TAU = 2 * np.pi

# 試験前に決めた合格基準
CRITERIA = {'heading_gain': [0.8, 1.2], 'travel_gain': [0.8, 1.2],
            'residual_R_min': 0.8, 'amplitude_r2_min': 0.9}

# 身体座標の選好方向（文献から。モデルパラメータであって配線由来ではない）
PREF_DEG = {'PFNd-L': +45.0, 'PFNd-R': -45.0, 'PFNv-L': -135.0, 'PFNv-R': +135.0}


def wrap(a):
    return (a + np.pi) % TAU - np.pi


class BodyVelocityInput:
    """身体座標の速度 (|v|, φ) から 4 集団を別々に駆動する。

    配線の実測に合わせて入口を割り当てる。LNO1 は PFNv（57%）と PFNd（43%）の
    両方へ入るので、PFNv を狙った駆動が PFNd へ漏れる。この漏れは実測の配線に
    あるものなので、そのまま残して測る。
    """

    def __init__(self, net, gain=0.08, baseline=0.02, excitatory=False,
                 pref_on='input'):
        """pref_on: 選好方向 ψ_g を誰の同調として与えるか。

          'input'  入口ニューロン（LNO/SpsP）の同調として与える。既定。
                   入口が抑制性なら、PFN の実効的な選好は **その 180° 反転**になる。
          'pfn'    PFN 自身の同調として与える。入口が抑制性なら駆動を 180° 回して
                   打ち消し、PFN が ψ_g に同調するようにする。

        6.10 節で分かったこと: 'input' で ±45/±135 を与えると、入口が 4 型とも
        抑制性なので PFN の実効選好は 4 集団とも前方寄り（+41/−39/+22/−24）に
        潰れ、4 基底が同位相で変調される。だから和が回らない。
        """
        self.net, self.gain, self.baseline = net, gain, baseline
        self.excitatory = excitatory
        self.pref_on = pref_on
        # 入口 → 駆動したい集団
        self.routes = [(net.lno_left, 'PFNd-R'), (net.lno_right, 'PFNd-L'),
                       (net.spsp_right, 'PFNd-R'), (net.spsp_left, 'PFNd-L')]
        # LNO1 と LNO2 を分ける
        lno1 = {i for i in range(net.N) if net.type_of(i).startswith('LNO1')}
        self.lno1_left = np.array([i for i in net.lno_left if i in lno1], dtype=np.int64)
        self.lno1_right = np.array([i for i in net.lno_right if i in lno1], dtype=np.int64)
        self.lno2_left = np.array([i for i in net.lno_left if i not in lno1], dtype=np.int64)
        self.lno2_right = np.array([i for i in net.lno_right if i not in lno1], dtype=np.int64)

    def drives(self, speed, phi):
        """各集団の駆動量 = max(0, cos(φ − ψ'_g)) · |v|

        ψ'_g は pref_on による。'pfn' かつ入口が抑制性なら ψ_g + 180°。
        """
        flip = 180.0 if (self.pref_on == 'pfn' and not self.excitatory) else 0.0
        return {g: float(max(0.0, np.cos(phi - np.radians(p + flip))) * speed)
                for g, p in PREF_DEG.items()}

    def apply(self, clamp, speed, phi):
        d = self.drives(speed, phi)
        b, g = self.baseline, self.gain
        def put(idx, val):
            for i in idx:
                clamp[i] = max(0.0, b + g * val)
        put(self.lno2_left, d['PFNd-R']);  put(self.lno2_right, d['PFNd-L'])
        put(self.net.spsp_right, d['PFNd-R']); put(self.net.spsp_left, d['PFNd-L'])
        put(self.lno1_left, d['PFNv-R']);  put(self.lno1_right, d['PFNv-L'])
        return clamp


def hdb_readout(net):
    """hΔB の読み出し。位相は 6.8 節の結線由来（出力側、R ≥ 0.3）。"""
    hp = json.load(open(os.path.join(ROOT, 'data', 'hdelta_phase.json'), encoding='utf-8'))
    by_body = {c['body_id']: c for c in hp['cells']
               if c['type'] == 'hDeltaB' and c['R_out'] >= 0.3}
    idx, ph = [], []
    for i in net.hdelta:
        c = by_body.get(int(net.body_id[i]))
        if c:
            idx.append(int(i)); ph.append(c['phase_out'])
    return np.array(idx, dtype=np.int64), np.array(ph)


def build(sign_variant, pfn_gain, hd_gain, pfnv_gain=1.0,
          shunt_frac=0.0, shunt_gain=1.0):
    net = CXNetworkNP(MODEL)
    if sign_variant != 'measured':
        flip_signs(net, sign_variant)
        net.invalidate_weight_split()
    net.shunt_frac = shunt_frac
    net.shunt_gain = shunt_gain
    if pfn_gain != 1.0:
        sel = np.array([i for i in range(net.N)
                        if re.sub(r'_.*', '', net.type_of(i)) in ('PFNd', 'PFNv')])
        net.row_gain[sel] *= pfn_gain
    if pfnv_gain != 1.0:
        # PFNv だけの行ゲイン。'selfmotion' 条件では PFNv が飽和して φ 変調を失うので、
        # 下げれば変調が戻るか（＝4 基底が揃うか）を見るための自由パラメータ
        sel = np.array([i for i in range(net.N)
                        if re.sub(r'_.*', '', net.type_of(i)) == 'PFNv'])
        net.row_gain[sel] *= pfnv_gain
    if hd_gain != 1.0:
        net.row_gain[net.hdelta] *= hd_gain
    return net


def sweep(sign_variant='measured', pfn_gain=1.0, hd_gain=1.0, pfnv_gain=1.0,
          shunt_frac=0.0, shunt_gain=1.0, pref_on='input',
          thetas=range(0, 360, 45), phis=range(0, 360, 45), speeds=(0.5, 1.0),
          steps=400):
    net = build(sign_variant, pfn_gain, hd_gain, pfnv_gain, shunt_frac, shunt_gain)
    smi = BodyVelocityInput(net, excitatory=(sign_variant != 'measured'),
                            pref_on=pref_on)
    idx, ph = hdb_readout(net)
    cos_, sin_ = np.cos(ph), np.sin(ph)
    rows = []
    for tdeg in thetas:
        th = np.radians(tdeg)
        for pdeg in phis:
            phi = np.radians(pdeg)
            for sp in speeds:
                net.reset(1)
                for _ in range(steps):
                    net.clear_drive(); net.set_visual_scene(th, 1.0)
                    smi.apply(net.clamp_val, sp, phi); net.step(DT)
                a = net.r[idx]
                x = float((a * cos_).sum()); y = float((a * sin_).sum())
                mag = float(np.hypot(x, y))
                rows.append({'theta': tdeg, 'phi': pdeg, 'speed': sp,
                             'psi': float(np.arctan2(y, x)) if mag > 1e-12 else None,
                             'mag': mag, 'mean_rate': float(a.mean())})
    return rows, len(idx)


def circular_fit(rows):
    """ψ = a·θ + b·φ + c を円環で当てはめる（a, b を格子探索して残差の集中度を最大化）。"""
    ok = [r for r in rows if r['psi'] is not None and r['mag'] > 1e-9]
    if len(ok) < 8:
        return None
    th = np.radians([r['theta'] for r in ok]); phi = np.radians([r['phi'] for r in ok])
    psi = np.array([r['psi'] for r in ok])
    best = None
    grid = np.arange(-1.5, 1.501, 0.05)
    for a in grid:
        for b in grid:
            d = wrap(psi - a * th - b * phi)
            R = float(np.hypot(np.cos(d).mean(), np.sin(d).mean()))
            if best is None or R > best[0]:
                best = (R, float(a), float(b),
                        float(np.arctan2(np.sin(d).mean(), np.cos(d).mean())))
    R, a, b, c = best
    return {'heading_gain': a, 'travel_gain': b, 'offset_deg': float(np.degrees(c)),
            'residual_R': R, 'n': len(ok)}


def amplitude_fit(rows):
    """長さが速さに比例するか（原点を通る線形回帰の R²）。"""
    ok = [r for r in rows if r['mag'] > 1e-12]
    if len(ok) < 4:
        return None
    x = np.array([r['speed'] for r in ok]); y = np.array([r['mag'] for r in ok])
    k = float((x * y).sum() / (x * x).sum())
    ss_res = float(((y - k * x) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {'slope': k, 'r2': float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            'n': len(ok)}


def verdict(fit, amp):
    if fit is None:
        return {'pass': False, 'why': '読み出しが沈黙'}
    c = CRITERIA
    checks = {
        'heading_gain': c['heading_gain'][0] <= fit['heading_gain'] <= c['heading_gain'][1],
        'travel_gain': c['travel_gain'][0] <= fit['travel_gain'] <= c['travel_gain'][1],
        'residual_R': fit['residual_R'] >= c['residual_R_min'],
        'amplitude_r2': bool(amp and amp['r2'] is not None
                             and amp['r2'] >= c['amplitude_r2_min']),
    }
    return {'pass': all(checks.values()), 'checks': checks}


def run(name, **kw):
    rows, n = sweep(**kw)
    fit, amp = circular_fit(rows), amplitude_fit(rows)
    v = verdict(fit, amp)
    res = {'name': name, 'params': kw, 'n_readout_cells': n,
           'fit': fit, 'amplitude': amp, 'verdict': v, 'rows': rows}
    if fit:
        print(f"[{name}] 方位の利得 {fit['heading_gain']:+.2f}  "
              f"進行方向の利得 {fit['travel_gain']:+.2f}  "
              f"残差の集中度 {fit['residual_R']:.2f}  "
              f"振幅 R² {amp['r2'] if amp and amp['r2'] is not None else float('nan'):.2f}  "
              f"→ {'合格' if v['pass'] else '不合格'}", flush=True)
    else:
        print(f"[{name}] {v['why']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()

    results = []
    # 陽性対照: 理想の位相（θ+φ）を hΔB に直接書いたら合格するか＝採点系の検定
    print('=== 採点系の検定（合成データ）')
    synth = []
    rng = np.random.default_rng(0)
    for tdeg in range(0, 360, 45):
        for pdeg in range(0, 360, 45):
            for sp in (0.5, 1.0):
                psi = np.radians(tdeg + pdeg) + rng.normal(0, 0.05)
                synth.append({'theta': tdeg, 'phi': pdeg, 'speed': sp,
                              'psi': float(wrap(psi)), 'mag': sp * 1.0, 'mean_rate': sp})
    f, am = circular_fit(synth), amplitude_fit(synth)
    print(f"  理想の変換器: 方位 {f['heading_gain']:+.2f} 進行方向 {f['travel_gain']:+.2f} "
          f"R {f['residual_R']:.2f} 振幅 R² {am['r2']:.2f} → "
          f"{'合格' if verdict(f, am)['pass'] else '不合格'}")
    # 陰性対照: 方位だけ（進行方向を無視）
    syn2 = [dict(r, psi=float(wrap(np.radians(r['theta'])))) for r in synth]
    f2, am2 = circular_fit(syn2), amplitude_fit(syn2)
    print(f"  方位のみ:     方位 {f2['heading_gain']:+.2f} 進行方向 {f2['travel_gain']:+.2f} "
          f"R {f2['residual_R']:.2f} → "
          f"{'合格' if verdict(f2, am2)['pass'] else '不合格'}")
    controls = {'ideal': {'fit': f, 'amplitude': am, 'verdict': verdict(f, am)},
                'heading_only': {'fit': f2, 'amplitude': am2, 'verdict': verdict(f2, am2)}}

    print('\n=== ネットワーク')
    results.append(run('実測の符号', sign_variant='measured', pfn_gain=1.0, hd_gain=1.0))
    results.append(run('実測の符号・hΔ ゲイン 10', sign_variant='measured',
                       pfn_gain=1.0, hd_gain=10.0))
    results.append(run('自己運動を興奮性に', sign_variant='selfmotion',
                       pfn_gain=1.0, hd_gain=1.0))
    results.append(run('自己運動を興奮性に・hΔ ゲイン 10', sign_variant='selfmotion',
                       pfn_gain=1.0, hd_gain=10.0))
    results.append(run('グルタミン酸だけ興奮性に', sign_variant='glut',
                       pfn_gain=1.0, hd_gain=1.0))

    # 6.10 節: 分流（シャント）抑制。C2（scripts/shunting.py）で
    # PFNd/PFNv が「生きて・同調して・φ で変調される」ことを確認した条件で通す。
    print('\n=== 分流抑制を入れる（6.10 節）')
    results.append(run('分流抑制 f=1.0', sign_variant='measured',
                       pfn_gain=1.0, hd_gain=1.0, shunt_frac=1.0))
    results.append(run('分流抑制 f=1.0 強さ 0.3', sign_variant='measured',
                       pfn_gain=1.0, hd_gain=1.0, shunt_frac=1.0, shunt_gain=0.3))
    results.append(run('分流抑制 f=1.0 ＋ glut 興奮性', sign_variant='glut',
                       pfn_gain=1.0, hd_gain=1.0, shunt_frac=1.0))
    results.append(run('分流抑制 f=0.5（迷路が保たれる側）', sign_variant='measured',
                       pfn_gain=1.0, hd_gain=1.0, shunt_frac=0.5))

    # 選好方向を PFN 自身の同調として与える（入口が抑制性なので駆動は 180° 回る）。
    # 'input' で与えると 4 集団の実効選好が前方に潰れることが分かったので、その対照。
    print('\n=== 選好を PFN 自身の同調として与える')
    results.append(run('分流 f=1.0・選好は PFN 側', sign_variant='measured',
                       pfn_gain=1.0, hd_gain=1.0, shunt_frac=1.0, pref_on='pfn'))
    results.append(run('分流 f=1.0 ＋ glut・選好は PFN 側', sign_variant='glut',
                       pfn_gain=1.0, hd_gain=1.0, shunt_frac=1.0, pref_on='pfn'))
    results.append(run('従来・選好は PFN 側', sign_variant='measured',
                       pfn_gain=1.0, hd_gain=10.0, pref_on='pfn'))

    # 4 基底を揃える最後の試み: PFNv の飽和を下げる
    print('\n=== PFNv の飽和を下げる（自己運動を興奮性にした上で）')
    for gv in (0.3, 0.1, 0.03, 0.01):
        results.append(run(f'自己運動を興奮性に・PFNv ゲイン {gv}', sign_variant='selfmotion',
                           pfn_gain=1.0, hd_gain=1.0, pfnv_gain=gv))

    with open(a.out, 'w', encoding='utf-8') as f_:
        json.dump({'criteria': CRITERIA, 'pref_deg': PREF_DEG,
                   'controls': controls, 'results': results}, f_,
                  ensure_ascii=False, indent=1)
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
