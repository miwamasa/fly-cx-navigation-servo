#!/usr/bin/env python3
"""PFN の進行方向取り込みは、どの経路で作られているのか。

7.2 節で分かったこと: PFN は進行方向を部分的に取り込んでおり（k ≈ 0.12）、
それは配線依存だが、**PB→FB の鏡像写像を壊しても消えない**。
つまり「左右 PFN の ±66° ベクトル基底」では説明できない。

代わりに立てた仮説:
  β は「左右 PFN のどちらを強く駆動するか」としてのみ回路に入る。
  PFN の集団ベクトルは各細胞の**扁豆体の糸球体**から付けた位相で読むので、
  左右の駆動比が偏るだけで、その集団ベクトルは回る。
  PB→FB の写像は下流（hΔ/vΔ が何を受け取るか）を決めるだけで、ここには効かない。

この仮説は 3 つの対照で検証できる。いずれも予測がはっきりしている。

  ① 駆動を左右均等にする（和＝速さは保ち、差＝向きだけ消す） → k → 0
  ② 左右の駆動を入れ替える                                    → k の符号が反転
  ③ 読み出しに使う PFN の位相ラベルだけをシャッフル（配線も力学も無傷） → k → 0

③ が効くなら、「比」を「角度」に変換しているのは糸球体の位相地図だと言える。

  python3 scripts/mechanism_control.py
  python3 scripts/mechanism_control.py --quick
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
from track_direction import POPS, summarize, travel_gain            # noqa: E402
from train_pi import MODEL, apply_integrator                        # noqa: E402

OUT = os.path.join(ROOT, 'data', 'pi_mechanism.json')


def shuffled_pfn_phase(net, seed=0):
    """読み出し用の位相配列。PFN の位相ラベルだけを入れ替える。

    ネットワークの重みも力学も一切変えない。変わるのは
    「集団ベクトルを計算するとき、どの細胞にどの角度を割り当てるか」だけ。
    """
    ph = net.phase.copy()
    idx = net.pfn[ph[net.pfn] > -8]
    ph[idx] = np.random.default_rng(seed).permutation(ph[idx])
    return ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=900)
    ap.add_argument('--sideslip', type=float, default=55.0)
    ap.add_argument('--sideslip-tau', type=float, default=4.0)
    ap.add_argument('--hd-gain', type=float, default=10.0)
    ap.add_argument('--hd-tau', type=float, default=10.0)
    ap.add_argument('--traj', type=int, default=20)
    ap.add_argument('--sets', type=int, default=20)
    ap.add_argument('--seed0', type=int, default=50000)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()
    if a.quick:
        a.traj, a.sets = 6, 3
    args = argparse.Namespace(steps=a.steps, sideslip=a.sideslip,
                              sideslip_tau=a.sideslip_tau)

    # (表示名, 自己運動の入れ方, 読み出し位相を作る関数, 予測)
    conds = [
        ('基準（実測のまま）', 'normal', None, 'k > 0'),
        ('① 駆動を左右均等に', 'equalized', None, 'k → 0'),
        ('② 左右の駆動を入替', 'swapped', None, 'k の符号が反転'),
        ('③ PFN 位相をシャッフル', 'normal', shuffled_pfn_phase, 'k → 0'),
    ]

    raw = {c[0]: {p: [] for p in POPS} for c in conds}
    print(f'軌跡 {a.traj} 本 × 独立 {a.sets} セット、横滑り σ={a.sideslip:.0f}°')
    t0 = time.time()
    for s in range(a.sets):
        seed0 = a.seed0 + s * a.traj * 2
        for name, mode, phfn, _pred in conds:
            net = CXNetworkNP(MODEL)
            apply_integrator(net, a.hd_gain, a.hd_tau)
            smi = SelfMotionInput(net, lno_gain=0.08, spsp_gain=0.08)
            ph = phfn(net, seed=s) if phfn else None
            g, _ = travel_gain(net, smi, args, a.traj, seed0,
                               drive_mode=mode, phase=ph)
            for p in POPS:
                raw[name][p].append(g[p])
        print(f'  セット {s+1}/{a.sets} ({time.time()-t0:.0f}s)', flush=True)

    out = {'params': vars(a),
           'metric': 'k: 集団ベクトルの角度 − θ = k·β + c。0=方位, 1=進行方向',
           'conditions': {}}
    for name, mode, phfn, pred in conds:
        out['conditions'][name] = {
            'drive_mode': mode,
            'readout_phase': 'PFN shuffled' if phfn else 'as measured',
            'prediction': pred,
            **{p: summarize(raw[name][p]) for p in POPS}}
    base = np.asarray(raw['基準（実測のまま）']['PFN'])
    out['paired_差'] = {
        f'基準 − {name}': summarize(base - np.asarray(raw[name]['PFN']))
        for name, _m, _f, _p in conds[1:]}
    # ② は符号反転が予測なので、和がゼロになるかも見る
    out['② との和'] = summarize(base + np.asarray(raw['② 左右の駆動を入替']['PFN']))

    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('\n進行方向の取り込み率 k')
    print(f'{"条件":24s} {"PFN":>18s} {"予測":>16s}  判定')
    b = out['conditions']['基準（実測のまま）']['PFN']
    for name, _m, _f, pred in conds:
        c = out['conditions'][name]['PFN']
        if name.startswith('基準'):
            verdict = '—'
        elif pred == 'k → 0':
            verdict = '予測どおり' if abs(c['k']) < c['ci95'] else '予測はずれ'
        else:
            verdict = '予測どおり' if c['k'] < -c['ci95'] else '予測はずれ'
        print(f'{name:24s} {c["k"]:+.3f} ± {c["ci95"]:.3f}'.ljust(46)
              + f'{pred:>16s}  {verdict}')
    print('\n対応のある差（同じ軌跡セットどうし。こちらのほうが検出力が高い）')
    for k, v in out['paired_差'].items():
        sig = '有意' if abs(v['k']) > v['ci95'] else '有意でない'
        print(f'  {k:28s} {v["k"]:+.3f} ± {v["ci95"]:.3f}   {sig}')
    w = out['② との和']
    print(f'  {"基準 + ②（反転なら 0）":28s} {w["k"]:+.3f} ± {w["ci95"]:.3f}   '
          + ('0 と区別できない＝きれいに反転' if abs(w['k']) < w['ci95'] else '完全な反転ではない'))
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
