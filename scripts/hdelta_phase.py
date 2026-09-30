#!/usr/bin/env python3
"""hΔ の位相座標を、ラベルではなく結線から決め直す。

  python3 scripts/hdelta_phase.py
  → data/hdelta_phase.json

なぜ必要か
----------
extract_cx.py は hΔ の位相を細胞名の列ラベル（_C1…_C12）から付けている。
ところが列→角度の対応は PFN/PFL の **9 列**で学習したもので、hΔ は **12 列**ある。
C10〜C12 は `2π(c−1)/9` の線形な代替値が入り、C1〜C9 も 9 列の座標で読まれている。
つまり hΔ の位相座標は、hΔ 自身の結線が示す位相と一致していない可能性がある。

hΔ は扇状体の離れた列を結ぶ細胞で、樹状突起の列と軸索の列が違う。
読み出し（集団ベクトル）に使うべきは「その細胞が **どこへ書くか**」なので、
ここでは各 hΔ について

  phase_in   前シナプス相手（PB 糸球体で位相が決まる細胞のみ）の重み付き円周平均
  phase_out  後シナプス相手（同上）の重み付き円周平均

を計算し、集中度 R とともに保存する。glomerulus 由来の位相しか使わないのは、
列ラベル由来の位相（hΔ 自身や FC など）を使うと循環参照になるから。
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NPZ = os.path.join(ROOT, 'data', 'cx_network.npz')
OUT = os.path.join(ROOT, 'data', 'hdelta_phase.json')
TAU = 2 * np.pi


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def circ_mean(ph, wt):
    x = float((wt * np.cos(ph)).sum()); y = float((wt * np.sin(ph)).sum())
    s = float(wt.sum())
    return (float(np.arctan2(y, x)) % TAU, (np.hypot(x, y) / s) if s > 0 else 0.0, s)


def compute(npz_path=NPZ):
    d = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in d['names']]
    phase = d['phase'].astype(float)
    pre, post, w = d['pre'], d['post'], d['weight'].astype(float)
    body = d['body_id']
    glom_derived = np.array(['(PB' in nm for nm in names])   # 糸球体ラベル由来の位相を持つ細胞
    has = phase > -8
    hd = [i for i, nm in enumerate(names) if nm.startswith('hDelta')]

    cells = []
    for i in hd:
        m_in = (post == i) & glom_derived[pre] & has[pre] & (w > 0)
        m_out = (pre == i) & glom_derived[post] & has[post] & (w > 0)
        pin, Rin, win = circ_mean(phase[pre[m_in]], w[m_in])
        pout, Rout, wout = circ_mean(phase[post[m_out]], w[m_out])
        cells.append({'index': int(i), 'body_id': int(body[i]), 'name': names[i],
                      'type': names[i].split('_')[0],
                      'phase_assigned': float(phase[i]),
                      'phase_in': pin, 'R_in': Rin, 'w_in': win,
                      'phase_out': pout, 'R_out': Rout, 'w_out': wout})

    def summary(key_a, key_b, Rkey_a=None, Rkey_b=None, thr=0.3):
        sel = [c for c in cells if (Rkey_a is None or c[Rkey_a] >= thr)
               and (Rkey_b is None or c[Rkey_b] >= thr)]
        dd = np.array([wrap(c[key_a] - c[key_b]) for c in sel])
        x, y = np.cos(dd).mean(), np.sin(dd).mean()
        return {'n': len(sel), 'mean_deg': float(np.degrees(np.arctan2(y, x))),
                'R': float(np.hypot(x, y))}

    # 型ごとの in−out（180° シフトが型に依存するか）
    by_type = {}
    for t in sorted({c['type'] for c in cells}):
        sel = [c for c in cells if c['type'] == t and c['R_in'] >= 0.3 and c['R_out'] >= 0.3]
        if len(sel) < 3:
            continue
        dd = np.array([wrap(c['phase_in'] - c['phase_out']) for c in sel])
        x, y = np.cos(dd).mean(), np.sin(dd).mean()
        by_type[t] = {'n': len(sel), 'in_minus_out_deg': float(np.degrees(np.arctan2(y, x))),
                      'R': float(np.hypot(x, y))}

    out = {
        'n_cells': len(cells),
        'assigned_vs_in': summary('phase_assigned', 'phase_in', None, 'R_in'),
        'assigned_vs_out': summary('phase_assigned', 'phase_out', None, 'R_out'),
        'in_vs_out': summary('phase_in', 'phase_out', 'R_in', 'R_out'),
        'in_vs_out_by_type': by_type,
        'R_in_median': float(np.median([c['R_in'] for c in cells])),
        'R_out_median': float(np.median([c['R_out'] for c in cells])),
        'n_R_in_ge_0.3': int(sum(c['R_in'] >= 0.3 for c in cells)),
        'n_R_out_ge_0.3': int(sum(c['R_out'] >= 0.3 for c in cells)),
        'cells': cells,
    }
    return out


def main():
    out = compute()
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"hΔ {out['n_cells']} 細胞")
    for k in ('assigned_vs_in', 'assigned_vs_out', 'in_vs_out'):
        s = out[k]
        print(f"  {k:16s}: n={s['n']:3d}  円周平均 {s['mean_deg']:+6.0f}°  集中度 R={s['R']:.2f}")
    print('  型ごとの in − out:')
    for t, s in out['in_vs_out_by_type'].items():
        print(f"    {t:8s} n={s['n']:2d}  {s['in_minus_out_deg']:+6.0f}°  R={s['R']:.2f}")
    print(f"  R_in 中央値 {out['R_in_median']:.2f}（≥0.3: {out['n_R_in_ge_0.3']}）  "
          f"R_out 中央値 {out['R_out_median']:.2f}（≥0.3: {out['n_R_out_ge_0.3']}）")
    print(f'→ {OUT}')


if __name__ == '__main__':
    main()
