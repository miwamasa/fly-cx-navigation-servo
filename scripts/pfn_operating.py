#!/usr/bin/env python3
"""B1 の続き: PFNd/PFNv が動作する動作点があるかを探す。

  python3 scripts/pfn_operating.py
  → data/pfn_operating.json

なぜこれが要るのか
------------------
6.7〜6.8 節は「PFN→hΔ の変換段が逆向きの変位を表現できない」で止まっていた。
ところが B1（scripts/pfn_basis.py）で配線を見直すと、変換段そのものが
**動いていない**ことが分かった。

  ・自己運動の入口（LNO1/LNO2/LNOa/SpsP → PFN）は **すべて抑制性**
    （LNO1 は GABA、他は glutamate = GluCl 経由で抑制性とみなしている）
  ・PFNd の入力収支は CX 内で 興奮 +13,723 / 抑制 −19,464 = **純 −5,741**、
    さらに外部入力の復元が 1 細胞あたり −283
  ・EPG からの興奮は弱い（PFNd では上位 5 位に入らず、PFNv で +450）。
    PFN の方位同調は EPG の興奮ではなく **Δ7 の抑制**が作っている
  ・その結果、既定の動作点では **PFNv は速度・利得によらず一切発火せず、
    PFNd は自己運動を入れた瞬間に沈黙する**

つまり 7.2 節で測った PFN 集団ベクトル（k = +0.117）は PFNd/PFNv ではなく
PFNp/PFNm/PFNa が作っていた。文献が座標変換の担い手とする 2 つの細胞型は
最初から蚊帳の外だった。

ここで探すのは、hΔ に対して 6.7 節でやったのと同じこと — **行ゲインという
細胞側の自由パラメータで、PFNd と PFNv が閾値の上に乗る動作点があるか**。
無ければ「このモデル階層では変換段を動かせない」が結論になる。

測る量（すべて自己運動を切った状態と入れた状態で）
  alive     発火している細胞の割合
  tuning    方位 θ を回したときの集団ベクトルの集中度（方位同調があるか）
  modulation 自己運動を入れたときの活動の変化（符号つき）
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
from pi_task import SelfMotionInput                    # noqa: E402

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
OUT = os.path.join(ROOT, 'data', 'pfn_operating.json')
DT = 0.02
TAU = 2 * np.pi


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def build(pfn_gain=1.0, types=('PFNd', 'PFNv'), sign_variant='measured'):
    net = CXNetworkNP(MODEL)
    sel = np.array([i for i in range(net.N)
                    if re.sub(r'_.*', '', net.type_of(i)) in types], dtype=np.int64)
    if sign_variant != 'measured':
        flip_signs(net, sign_variant)
    if pfn_gain != 1.0:
        net.row_gain[sel] *= pfn_gain
    return net, sel


def flip_signs(net, variant):
    """符号の仮定を変える対照。

    グルタミン酸を抑制性（GluCl 経由）とみなすのはショウジョウバエでは標準的だが、
    予測値であり ground truth の無い型もある（SpsP, Δ7）。この仮定が
    陰性結果の原因かどうかを分けて確かめる。

      selfmotion  LNO*/SpsP すべて興奮性（GABA の LNO1 も含む = 実測に反する）
      glut        グルタミン酸の入口だけ（LNO2/LNOa/SpsP）。LNO1 は GABA なので抑制のまま
      spsp        SpsP だけ（ground truth が無く、予測信頼度 0.61 で最も弱い）
      lno1        LNO1 だけ（ground truth が GABA。反証用の対照）
      delta7      Δ7 → * だけ興奮性にする（方位同調を作っている相手）
      both        selfmotion + delta7
    """
    pre_types = {i: re.sub(r'_.*', '', net.type_of(i)) for i in range(net.N)}
    flip_src = set()
    if variant in ('selfmotion', 'both'):
        flip_src |= {i for i, t in pre_types.items()
                     if t.startswith('LNO') or t.startswith('SpsP')}
    if variant == 'glut':
        flip_src |= {i for i, t in pre_types.items()
                     if t in ('LNO2', 'LNOa') or t.startswith('SpsP')}
    if variant == 'spsp':
        flip_src |= {i for i, t in pre_types.items() if t.startswith('SpsP')}
    if variant == 'lno1':
        flip_src |= {i for i, t in pre_types.items() if t == 'LNO1'}
    if variant in ('delta7', 'both'):
        flip_src |= {i for i, t in pre_types.items() if t.startswith('Delta7')}
    W = net.W.tocsc(copy=True)
    for j in sorted(flip_src):
        seg = slice(W.indptr[j], W.indptr[j + 1])
        W.data[seg] = np.abs(W.data[seg])
    net.W = W.tocsr()
    return len(flip_src)


def settle(net, smi, theta, v, steps=400, mode='inhibit'):
    net.reset(1)
    for _ in range(steps):
        net.clear_drive()
        net.set_visual_scene(theta, 1.0)
        smi.apply(net.clamp_val, v)
        net.step(DT)


def pop_vector(net, idx):
    ph = net.phase[idx]
    ok = ph > -8
    a = net.r[idx][ok]; ph = ph[ok]
    if a.sum() <= 1e-12:
        return None, 0.0, 0.0
    x = float((a * np.cos(ph)).sum()); y = float((a * np.sin(ph)).sum())
    return float(np.arctan2(y, x)), float(np.hypot(x, y) / a.sum()), float(a.mean())


def probe(pfn_gain, thetas=(0, 90, 180, 270), speeds=(0.0, 1.0), sign_variant='measured'):
    net, _ = build(pfn_gain, sign_variant=sign_variant)
    smi = SelfMotionInput(net, lno_gain=0.08, spsp_gain=0.08)
    groups = {}
    for name in ('PFNd', 'PFNv'):
        groups[name] = np.array([i for i in net.pfn
                                 if re.sub(r'_.*', '', net.type_of(i)) == name], dtype=np.int64)
    out = {'pfn_gain': pfn_gain, 'sign_variant': sign_variant, 'groups': {}}
    for name, idx in groups.items():
        rows = []
        for v in speeds:
            angs, mags, means, alive = [], [], [], []
            for tdeg in thetas:
                settle(net, smi, np.radians(tdeg), v)
                ang, mag, mean = pop_vector(net, idx)
                alive.append(float(np.mean(net.r[idx] > 1e-6)))
                means.append(mean)
                if ang is not None:
                    angs.append(wrap(ang - np.radians(tdeg))); mags.append(mag)
            # 方位同調: 集団ベクトルが θ と一緒に回るか（差の集中度）
            if angs:
                x, y = np.cos(angs).mean(), np.sin(angs).mean()
                follow = float(np.hypot(x, y))
                offset = float(np.degrees(np.arctan2(y, x)))
            else:
                follow, offset = 0.0, float('nan')
            rows.append({'v': v, 'alive': float(np.mean(alive)),
                         'mean_rate': float(np.mean(means)),
                         'bump_R': float(np.mean(mags)) if mags else 0.0,
                         'heading_follow': follow, 'offset_deg': offset})
        m0, m1 = rows[0]['mean_rate'], rows[-1]['mean_rate']
        out['groups'][name] = {
            'by_speed': rows,
            'modulation': float((m1 - m0) / m0) if m0 > 1e-12 else None,
            'alive_still': rows[0]['alive'], 'alive_moving': rows[-1]['alive'],
            'heading_follow_still': rows[0]['heading_follow'],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()
    gains = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0]
    variants = ['measured', 'selfmotion', 'glut', 'spsp', 'lno1', 'delta7', 'both']
    all_res, chosen = {}, {}
    for var in variants:
        res = []
        print(f"\n=== 符号の仮定: {var}")
        print(f"{'PFN行ゲイン':>10s}  {'PFNd 静止/歩行':>16s} {'同調':>5s}  "
              f"{'PFNv 静止/歩行':>16s} {'同調':>5s}")
        for g in gains:
            r = probe(g, sign_variant=var)
            res.append(r)
            d, v_ = r['groups']['PFNd'], r['groups']['PFNv']
            print(f"{g:10.0f}  {d['alive_still']*100:6.0f}% /{d['alive_moving']*100:5.0f}% "
                  f"{d['heading_follow_still']:9.2f}  "
                  f"{v_['alive_still']*100:6.0f}% /{v_['alive_moving']*100:5.0f}% "
                  f"{v_['heading_follow_still']:9.2f}", flush=True)
        ok = [r for r in res
              if r['groups']['PFNd']['alive_moving'] > 0.2
              and r['groups']['PFNv']['alive_moving'] > 0.2
              and r['groups']['PFNd']['heading_follow_still'] > 0.5
              and r['groups']['PFNv']['heading_follow_still'] > 0.5]
        chosen[var] = ok[0]['pfn_gain'] if ok else None
        all_res[var] = res
        print(f"  → 両型が歩行中に活動し方位同調もある最小ゲイン: "
              f"{chosen[var] if chosen[var] else '存在しない'}")
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump({'scan': all_res, 'chosen_gain': chosen,
                   'criterion': 'PFNd/PFNv とも歩行中の活動率 > 20% かつ静止時の方位同調 R > 0.5'},
                  f, ensure_ascii=False, indent=1)
    print(f'→ {a.out}')


if __name__ == '__main__':
    main()
