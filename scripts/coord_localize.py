#!/usr/bin/env python3
"""C3 の続き: 座標変換の信号がどこで消えるかを段ごとに測る。

  python3 scripts/coord_localize.py
  → data/coord_localize.json

分流抑制（6.10 節）を入れると PFNd/PFNv は初めて「生きて・方位に同調して・
φ で変調される」状態になった（scripts/shunting.py）。それでも hΔB の
進行方向の利得は b ≈ 0 のままだった。つまり **φ は PFN にあるのに hΔB に
渡っていない** か、**PFN の段で既に集団ベクトルとして回っていない** かの
どちらかである。同じ円環回帰を段ごとに当てはめて切り分ける。

  PFN-4基底   4 集団の駆動量 A_g を重みとした Σ A_g·e^{i(θ+Δ_g)}
              （Δ_g は実測の書き込みオフセット。理論上ここで θ+φ になるはず）
  PFN集団     PFNd+PFNv の実際の活動から作った集団ベクトル（各細胞の糸球体位相）
  hΔB集団     hΔB の集団ベクトル（6.8 節の結線由来の位相）

あわせて hΔB の入力に占める PFN の割合も出す。PFN が少数派なら、
φ を運んでいても hΔB では薄まる。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from coord_transform import (BodyVelocityInput, PREF_DEG, build,  # noqa: E402
                             circular_fit, hdb_readout, wrap)

DT = 0.02
OUT = os.path.join(ROOT, 'data', 'coord_localize.json')


def groups_of(net):
    out = {}
    for g in PREF_DEG:
        fam, side = g.split('-')
        out[g] = np.array([i for i in net.pfn
                           if re.sub(r'_.*', '', net.type_of(i)) == fam
                           and {-1: 'L', 1: 'R'}.get(int(net.pb_side[i])) == side
                           and net.phase[i] > -8], dtype=np.int64)
    return out


def hdb_input_share(net):
    """hΔB の総入力に占める前シナプス型ごとの割合（興奮性のみ）。"""
    idx, _ = hdb_readout(net)
    hs = set(int(i) for i in idx)
    W = net.W.tocoo()
    tot = collections.Counter()
    for i, j, w in zip(W.row, W.col, W.data):
        if int(i) in hs and w > 0:
            tot[re.sub(r'_.*', '', net.type_of(int(j)))] += float(w)
    s = sum(tot.values())
    return {k: v / s for k, v in tot.most_common(12)}, s


def sweep_stages(shunt_frac, shunt_gain, sign_variant, steps=400):
    net = build(sign_variant, 1.0, 1.0, 1.0, shunt_frac, shunt_gain)
    bvi = BodyVelocityInput(net)
    grp = groups_of(net)
    hdb_idx, hdb_ph = hdb_readout(net)
    pfn_idx = np.concatenate([grp[g] for g in PREF_DEG])
    pfn_ph = net.phase[pfn_idx]
    # 実測の書き込みオフセット（6.9.1 節）
    pb = json.load(open(os.path.join(ROOT, 'data', 'pfn_basis.json'), encoding='utf-8'))
    delta = {g: pb['groups'][g]['write_offset'] for g in PREF_DEG}

    rows = {'basis': [], 'pfn': [], 'hdb': []}
    for tdeg in range(0, 360, 45):
        th = np.radians(tdeg)
        for pdeg in range(0, 360, 45):
            phi = np.radians(pdeg)
            for sp in (0.5, 1.0):
                net.reset(1)
                for _ in range(steps):
                    net.clear_drive(); net.set_visual_scene(th, 1.0)
                    bvi.apply(net.clamp_val, sp, phi); net.step(DT)
                base = {'theta': tdeg, 'phi': pdeg, 'speed': sp}

                # ① 4 基底の理論和（実際の集団活動を振幅に使う）
                x = y = 0.0
                for g in PREF_DEG:
                    A = float(net.r[grp[g]].mean())
                    x += A * np.cos(th + delta[g]); y += A * np.sin(th + delta[g])
                m = float(np.hypot(x, y))
                rows['basis'].append({**base, 'mag': m,
                                      'psi': float(np.arctan2(y, x)) if m > 1e-12 else None})

                # ② PFN の実際の集団ベクトル
                a = net.r[pfn_idx]
                x = float((a * np.cos(pfn_ph)).sum()); y = float((a * np.sin(pfn_ph)).sum())
                m = float(np.hypot(x, y))
                rows['pfn'].append({**base, 'mag': m,
                                    'psi': float(np.arctan2(y, x)) if m > 1e-12 else None})

                # ③ hΔB
                a = net.r[hdb_idx]
                x = float((a * np.cos(hdb_ph)).sum()); y = float((a * np.sin(hdb_ph)).sum())
                m = float(np.hypot(x, y))
                rows['hdb'].append({**base, 'mag': m,
                                    'psi': float(np.arctan2(y, x)) if m > 1e-12 else None})
    share, tot = hdb_input_share(net)
    return rows, share, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()

    conds = [('分流 f=1.0', 1.0, 1.0, 'measured'),
             ('分流 f=1.0 ＋ glut 興奮性', 1.0, 1.0, 'glut'),
             ('従来（減算のみ）', 0.0, 1.0, 'measured')]
    out = []
    label = {'basis': '4 基底の理論和', 'pfn': 'PFN 集団ベクトル', 'hdb': 'hΔB 集団ベクトル'}
    for name, f, g, sv in conds:
        rows, share, tot = sweep_stages(f, g, sv)
        rec = {'name': name, 'shunt_frac': f, 'shunt_gain': g, 'sign_variant': sv,
               'stages': {}, 'hdb_input_share': share}
        print(f'\n=== {name}')
        for stage in ('basis', 'pfn', 'hdb'):
            fit = circular_fit(rows[stage])
            rec['stages'][stage] = fit
            if fit:
                print(f"  {label[stage]:18s} 方位 {fit['heading_gain']:+.2f}  "
                      f"進行方向 {fit['travel_gain']:+.2f}  残差 R {fit['residual_R']:.2f}")
            else:
                print(f'  {label[stage]:18s} 沈黙')
        out.append(rec)
        if name.startswith('分流 f=1.0') and 'glut' not in name:
            print('  hΔB の興奮性入力の内訳:',
                  ', '.join(f'{k} {v*100:.0f}%' for k, v in list(share.items())[:6]))

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump({'conditions': out}, fh, ensure_ascii=False, indent=1)
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
