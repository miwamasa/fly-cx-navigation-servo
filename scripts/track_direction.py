#!/usr/bin/env python3
"""横滑り課題で、集団ベクトルが進行方向をどれだけ拾っているかを測る。

本プロジェクトはここで 3 回目の「誤差棒なしで点推定を比べる」をやった。
最初の版は軌跡 6 本・単一シードで PFN +8.7°、配線シャッフルで −3.2° という
結果を出し、これを配線特異的な陽性結果として報告してしまった。
軌跡セットを変えると −19.9°〜+8.7° まで動く量だった。

そこでこのスクリプトでは 2 つを変えている。

1. **指標を変えた。** 「方位と進行方向のどちらに近いか」ではなく、
   進行方向の取り込み率 k を回帰で直接測る:

       集団ベクトルの角度 − θ  =  k · β + c

   k = 0 なら純粋に方位、k = 1 なら純粋に進行方向。
   「どちらに近いか」は k < 0.5 のとき負を返すので、
   部分的な取り込みを「追っていない」と誤読してしまう。

2. **独立な軌跡セットを複数回して信頼区間を付けた。** 条件間は
   同じ軌跡セットで対応をとって比べる。

対照は 3 つ:
  実測コネクトーム / PB→FB 鏡像破壊（狙い撃ち）/ 配線シャッフル（粗い）

  python3 scripts/track_direction.py           # 既定 20 本 × 8 セット
  python3 scripts/track_direction.py --quick   # 短縮版
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from cxnet_np import CXNetworkNP                                    # noqa: E402
from pi_task import SelfMotionInput                                 # noqa: E402
import train_pi                                                     # noqa: E402
from train_pi import (                                              # noqa: E402
    MODEL, BURN_IN, DT, SUBSAMPLE, apply_integrator, break_pb_fb_mirror,
    pfn_lr_write_offset, shuffle_connectome,
)

OUT = os.path.join(ROOT, 'data', 'pi_tracking.json')
POPS = ('PFN', 'hDelta', 'EPG')


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def travel_gain(net, smi, args, n_traj, seed0, drive_mode='normal', phase=None):
    """1 セット分。各集団について k（進行方向の取り込み率）を返す。

    drive_mode  自己運動入力の対照（'equalized' / 'swapped'、pi_task 参照）
    phase       読み出しに使う位相。既定は net.phase。
                位相ラベルだけを差し替えた対照を回すために外から渡せる。
    """
    ph = net.phase if phase is None else phase
    groups = {'PFN': net.pfn, 'hDelta': net.hdelta, 'EPG': net.epg}
    dev = {k: [] for k in groups}
    beta = []
    for k in range(n_traj):
        traj = train_pi.make_traj(args, seed0 + k)
        net.reset(1 + k)
        for t in range(args.steps):
            net.clear_drive()
            net.set_visual_scene(traj['th'][t], 1.0)
            smi.apply(net.clamp_val, traj['v'][t], traj['beta'][t], drive_mode)
            net.step(DT)
            # 止まっている間は進行方向が定義できないので使わない
            if t < BURN_IN or t % SUBSAMPLE or traj['v'][t] <= 0:
                continue
            vals, ok = {}, True
            for name, idx in groups.items():
                sel = idx[ph[idx] > -8]
                a = net.r[sel]
                if a.sum() <= 1e-9:
                    ok = False
                    break
                vals[name] = np.arctan2((a * np.sin(ph[sel])).sum(),
                                        (a * np.cos(ph[sel])).sum())
            if not ok:
                continue
            for name in groups:
                dev[name].append(wrap(vals[name] - traj['th'][t]))
            beta.append(traj['beta'][t])
    b = np.asarray(beta)
    b = b - b.mean()
    out = {}
    for name in groups:
        d = np.asarray(dev[name])
        # 各集団は方位に対して一定のオフセットを持つので、円周平均を除く
        d = wrap(d - np.arctan2(np.mean(np.sin(d)), np.mean(np.cos(d))))
        out[name] = float((b @ d) / (b @ b))
    return out, len(b)


def summarize(vals):
    v = np.asarray(vals, dtype=float)
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan')
    return {'k': float(v.mean()), 'ci95': float(ci),
            'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            'sets': int(len(v)), 'per_set': [float(x) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=900)
    ap.add_argument('--sideslip', type=float, default=55.0)
    ap.add_argument('--sideslip-tau', type=float, default=4.0)
    ap.add_argument('--hd-gain', type=float, default=10.0)
    ap.add_argument('--hd-tau', type=float, default=10.0)
    ap.add_argument('--traj', type=int, default=20, help='1 セットあたりの軌跡本数')
    ap.add_argument('--sets', type=int, default=8, help='独立な軌跡セットの数')
    ap.add_argument('--seed0', type=int, default=50000)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()
    if a.quick:
        a.traj, a.sets = 6, 3
    args = argparse.Namespace(steps=a.steps, sideslip=a.sideslip,
                              sideslip_tau=a.sideslip_tau)

    conds = [
        ('実測コネクトーム', None),
        ('鏡像破壊', lambda n: break_pb_fb_mirror(n, verbose=False)),
        ('配線シャッフル', lambda n: shuffle_connectome(n, seed=0)),
    ]

    raw = {c: {p: [] for p in POPS} for c, _ in conds}
    offsets, n_samples = {}, None
    print(f'軌跡 {a.traj} 本 × 独立 {a.sets} セット、横滑り σ={a.sideslip:.0f}°')
    t0 = time.time()
    for s in range(a.sets):
        seed0 = a.seed0 + s * a.traj * 2
        for name, fn in conds:
            net = CXNetworkNP(MODEL)
            apply_integrator(net, a.hd_gain, a.hd_tau)
            if fn:
                fn(net)
            if name not in offsets:
                offsets[name] = pfn_lr_write_offset(net, 'vdelta')
            smi = SelfMotionInput(net, lno_gain=0.08, spsp_gain=0.08)
            g, n_samples = travel_gain(net, smi, args, a.traj, seed0)
            for p in POPS:
                raw[name][p].append(g[p])
        print(f'  セット {s+1}/{a.sets} ({time.time()-t0:.0f}s)', flush=True)

    out = {'params': vars(a), 'samples_per_set': n_samples,
           'metric': 'k: 集団ベクトルの角度 − θ = k·β + c。0=方位, 1=進行方向',
           'conditions': {}}
    for name, _ in conds:
        out['conditions'][name] = {
            'pfn_lr_write_offset': offsets[name],
            **{p: summarize(raw[name][p]) for p in POPS}}
    # 条件間は同じ軌跡セットどうしで対応をとって比べる
    out['paired_差'] = {}
    for a_, b_ in (('実測コネクトーム', '配線シャッフル'), ('実測コネクトーム', '鏡像破壊')):
        out['paired_差'][f'{a_} − {b_}'] = {
            p: summarize(np.asarray(raw[a_][p]) - np.asarray(raw[b_][p]))
            for p in POPS}

    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'\n進行方向の取り込み率 k（0 = 方位のみ / 1 = 進行方向のみ）')
    print(f'{"条件":16s} ' + ' '.join(f'{p:>18s}' for p in POPS))
    for name, _ in conds:
        c = out['conditions'][name]
        print(f'{name:16s} ' + ' '.join(
            f'{c[p]["k"]:+.3f} ± {c[p]["ci95"]:.3f}'.rjust(18) for p in POPS)
            + f'   左右のずれ {c["pfn_lr_write_offset"]["offset_deg"]:+6.1f}°')
    print('\n対応のある差')
    for k, v in out['paired_差'].items():
        print(f'  {k}')
        for p in POPS:
            sig = '有意' if abs(v[p]['k']) > v[p]['ci95'] else '有意でない'
            print(f'    {p:8s} {v[p]["k"]:+.3f} ± {v[p]["ci95"]:.3f}   {sig}')
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
