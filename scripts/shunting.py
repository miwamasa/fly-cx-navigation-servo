#!/usr/bin/env python3
"""C2: 分流（シャント）抑制を宣言して入れ、変換段が動くかを判定する。

  python3 scripts/shunting.py
  → data/shunting.json

なぜこの機構か
--------------
6.9.5 節で、変換段が動かない原因は 2 つに分解された。

  ・PFNd を黙らせている入力はすべて「グルタミン酸は抑制性」という 1 つの仮定に載っている
  ・PFNv は主な抑制が **GABA 作動性の LNO1（ground truth）** で、EPG からの興奮が
    +22/細胞 しか無い。**符号の読み替えではどの組み合わせでも救えない**

残る候補が分流抑制である。GABA-A も GluCl も塩化物チャネルで、生理としては
減算より **除算**（分流）に近い。6.9.2 節の壁 —「方位同調（Δ7 の抑制）と自己運動
（LNO/SpsP の抑制）がどちらも減算で入るので、片方に耐える利得がもう片方を潰す」—
は、抑制が除算なら原理的に消える。除算は**同調の形を保ったまま利得だけを下げる**からである。

宣言するもの
------------
`cxnet_np.CXNetworkNP.shunt_frac`（0 = 全部減算 = 従来、1 = 全部除算）と
`shunt_gain`（除算の強さ）。**既定は 0 で、従来の力学とビット単位で一致する**
（tests/test_pi.py の TestShuntingDefaultIsInert で固定）。
これは配線から導いた値ではなく、**モデルパラメータとして宣言して足す機構**である。

測るもの（すべて 6.9 節と同じ量。合格基準も変えない）
  alive       歩行中に発火している割合
  tuning      静止時の方位同調 R
  phi_mod     φ を振ったときの活動の変調（集団ごと。0 なら進行方向を運んでいない）
  maze        閉ループの方位誤差（機構を足した代償を測る。6.5 節と同じ作法）
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
from coord_transform import BodyVelocityInput, PREF_DEG  # noqa: E402
from pfn_operating import flip_signs                   # noqa: E402
from pi_task import SelfMotionInput                    # noqa: E402

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
OUT = os.path.join(ROOT, 'data', 'shunting.json')
DT = 0.02
TAU = 2 * np.pi

# 6.9 節と同じ判定基準（変えない）
CRITERION = {'alive_moving_min': 0.2, 'heading_tuning_min': 0.5, 'phi_mod_min': 0.2}


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def build(shunt_frac=0.0, shunt_gain=1.0, sign_variant='measured'):
    net = CXNetworkNP(MODEL)
    if sign_variant != 'measured':
        flip_signs(net, sign_variant)
        net.invalidate_weight_split()
    net.shunt_frac = shunt_frac
    net.shunt_gain = shunt_gain
    return net


def groups_of(net):
    out = {}
    for g, _ in PREF_DEG.items():
        fam, side = g.split('-')
        out[g] = np.array([i for i in net.pfn
                           if re.sub(r'_.*', '', net.type_of(i)) == fam
                           and {-1: 'L', 1: 'R'}.get(int(net.pb_side[i])) == side],
                          dtype=np.int64)
    for fam in ('PFNd', 'PFNv'):
        out[fam] = np.array([i for i in net.pfn
                             if re.sub(r'_.*', '', net.type_of(i)) == fam], dtype=np.int64)
    return out


def pop_vector(net, idx):
    ph = net.phase[idx]
    ok = ph > -8
    a = net.r[idx][ok]; ph = ph[ok]
    if a.sum() <= 1e-12:
        return None, 0.0
    x = float((a * np.cos(ph)).sum()); y = float((a * np.sin(ph)).sum())
    return float(np.arctan2(y, x)), float(np.hypot(x, y) / a.sum())


def probe(net, steps=400):
    """歩行中の活動率・静止時の方位同調・φ 変調を測る。"""
    grp = groups_of(net)
    smi = SelfMotionInput(net, lno_gain=0.08, spsp_gain=0.08)
    bvi = BodyVelocityInput(net)

    # 方位同調（静止時）と歩行中の活動率: 6.9 節と同じ作法
    res = {}
    for label, v in (('still', 0.0), ('moving', 1.0)):
        angs, alive, rates = {g: [] for g in grp}, {g: [] for g in grp}, {g: [] for g in grp}
        for tdeg in (0, 90, 180, 270):
            net.reset(1)
            for _ in range(steps):
                net.clear_drive(); net.set_visual_scene(np.radians(tdeg), 1.0)
                smi.apply(net.clamp_val, v); net.step(DT)
            for g, idx in grp.items():
                ang, _ = pop_vector(net, idx)
                if ang is not None:
                    angs[g].append(wrap(ang - np.radians(tdeg)))
                alive[g].append(float(np.mean(net.r[idx] > 1e-6)))
                rates[g].append(float(net.r[idx].mean()))
        for g in grp:
            a = angs[g]
            follow = (float(np.hypot(np.cos(a).mean(), np.sin(a).mean())) if a else 0.0)
            res.setdefault(g, {})[f'alive_{label}'] = float(np.mean(alive[g]))
            res[g][f'tuning_{label}'] = follow
            res[g][f'rate_{label}'] = float(np.mean(rates[g]))

    # φ 変調: 身体座標の進行方向を振ったときの、各集団の活動の振れ幅
    for g in grp:
        res[g]['rate_by_phi'] = []
    for pdeg in (0, 45, 90, 135, 180, 225, 270, 315):
        net.reset(1)
        for _ in range(steps):
            net.clear_drive(); net.set_visual_scene(0.0, 1.0)
            bvi.apply(net.clamp_val, 1.0, np.radians(pdeg)); net.step(DT)
        for g, idx in grp.items():
            res[g]['rate_by_phi'].append(float(net.r[idx].mean()))
    for g in grp:
        r = np.array(res[g]['rate_by_phi'])
        m = float(r.mean())
        # 変調度 = (max − min) / mean。0 なら φ を運んでいない
        res[g]['phi_mod'] = float((r.max() - r.min()) / m) if m > 1e-12 else 0.0
        # φ に対する選好方向（変調が余弦なら位相が立つ）
        ph = np.radians(np.arange(0, 360, 45))
        x = float((r * np.cos(ph)).sum()); y = float((r * np.sin(ph)).sum())
        res[g]['phi_pref_deg'] = float(np.degrees(np.arctan2(y, x)))
        res[g]['phi_R'] = float(np.hypot(x, y) / r.sum()) if r.sum() > 1e-12 else 0.0
    return res


def maze_error(net, seeds=4, goals=4, steps=700):
    """閉ループの方位誤差。機構を足した代償を測る（6.5 節と同じ）。"""
    errs = []
    for s in range(seeds):
        for k in range(goals):
            mag = np.radians(30 + 140 * (k + 0.5) / goals)
            goal = float((1 if k % 2 else -1) * mag)
            net.reset(1 + s)
            th = 0.0
            for t in range(150 + steps):
                net.clear_drive(); net.set_visual_scene(th, 1.0)
                net.inject_goal(goal, 0.3); net.step(DT)
                if t >= 150:
                    th = wrap(th + 2.6 * net.read_steering()['turn'] * DT)
            errs.append(abs(np.degrees(wrap(th - goal))))
    return float(np.median(errs))


def verdict(res):
    c = CRITERION
    ok = {}
    for fam in ('PFNd', 'PFNv'):
        v = res[fam]
        ok[fam] = {'alive': v['alive_moving'] >= c['alive_moving_min'],
                   'tuning': v['tuning_still'] >= c['heading_tuning_min'],
                   'phi_mod': v['phi_mod'] >= c['phi_mod_min']}
        ok[fam]['pass'] = all(ok[fam].values())
    ok['both'] = ok['PFNd']['pass'] and ok['PFNv']['pass']
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()

    rows = []
    print(f"{'条件':34s} {'PFNd 生/同調/φ変調':>26s} {'PFNv 生/同調/φ変調':>26s}  迷路")
    conds = [('従来（減算のみ）', 0.0, 1.0, 'measured')]
    for f in (0.25, 0.5, 0.75, 1.0):
        conds.append((f'分流 f={f}', f, 1.0, 'measured'))
    for gain in (0.3, 3.0, 10.0):
        conds.append((f'分流 f=1.0 強さ {gain}', 1.0, gain, 'measured'))
    conds.append(('分流 f=1.0 ＋ glut 興奮性', 1.0, 1.0, 'glut'))

    for name, f, g, sv in conds:
        net = build(f, g, sv)
        res = probe(net)
        v = verdict(res)
        net2 = build(f, g, sv)
        mz = maze_error(net2)
        rows.append({'name': name, 'shunt_frac': f, 'shunt_gain': g,
                     'sign_variant': sv, 'probe': res, 'verdict': v, 'maze_deg': mz})
        d, w = res['PFNd'], res['PFNv']
        mark = '  ← 合格' if v['both'] else ''
        print(f"{name:34s} "
              f"{d['alive_moving']*100:5.0f}% {d['tuning_still']:5.2f} {d['phi_mod']:6.2f}  "
              f"{w['alive_moving']*100:5.0f}% {w['tuning_still']:5.2f} {w['phi_mod']:6.2f}  "
              f"{mz:5.1f}°{mark}", flush=True)

    best = [r for r in rows if r['verdict']['both']]
    print(f"\n両型が「生きて・同調して・φ で変調される」条件: "
          f"{[r['name'] for r in best] if best else 'なし'}")
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump({'criterion': CRITERION, 'rows': rows,
                   'passing': [r['name'] for r in best]}, fh, ensure_ascii=False, indent=1)
    print(f'→ {a.out}')


if __name__ == '__main__':
    main()
