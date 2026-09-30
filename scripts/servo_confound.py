#!/usr/bin/env python3
"""実験 0 — 交絡の除去。目標を hΔB にも注入していた件を切り分ける。

  python3 scripts/servo_confound.py
  → data/servo_confound.json

何が問題だったか
----------------
`cxnet_np.inject_goal` は目標を `goal_cells` へ注入する。その `goal_cells` が

    FC2A 18 + FC2B 27 + FC2C 47 + hDeltaB 19 = 111 細胞

で、**hΔB にも目標を書き込んでいた**。記事（docs/RESEARCH_PLAN_SERVO.md 1 章）の
枠組みでは hΔB はフィードバック信号（進行方向）の担い手なので、
そこへ目標を書き込むのは制御系として筋が通らない。
本リポジトリの閉ループ結果（4.3 節の迷路、6.5 節以降の代償、6.10 節の 19.3°→99.3°）は
すべてこの状態で走っていた。

事前登録した判定（docs/RESEARCH_PLAN_SERVO.md 5 章、走らせる前に固定）
--------------------------------------------------------------------
FC2 のみに注入したときの迷路の方位誤差を e とする。

  e ≤ 25°        目標は実質 FC2 に載っていた。既存結果は有効、注記のみ
  25° < e ≤ 60°  hΔB への注入が部分的に効いていた。既存の数値を再測定して差し替える
  e > 60°        操舵は hΔB への注入に依存していた。閉ループ結果の解釈を全面的に見直す

条件
----
  従来（FC2 ＋ hΔB）   これまでの既定。19.3° が再現するはず（回帰の確認にもなる）
  FC2 のみ             交絡を除いた本命
  hΔB のみ             診断用。hΔB だけでどれだけ操舵できるか

作法は 6.10 節と同じ閉ループ（視覚手がかりあり、PFL3 の左右差で旋回）。
試行は種 6 × 目標 8 = 48。信頼区間は試行を復元抽出したブートストラップ。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from cxnet_np import CXNetworkNP  # noqa: E402

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
OUT = os.path.join(ROOT, 'data', 'servo_confound.json')
DT = 0.02
TAU = 2 * np.pi

# 事前登録した判定の境界（度）。結果を見てから動かさない。
PREREG = {'ok_max': 25.0, 'partial_max': 60.0}


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def maze_trials(net, seeds=6, goals=8, steps=700, settle=150):
    """閉ループの方位誤差を試行ごとに返す（6.10 節と同じ作法）。"""
    errs = []
    for s in range(seeds):
        for k in range(goals):
            mag = np.radians(30 + 140 * (k + 0.5) / goals)
            goal = float((1 if k % 2 else -1) * mag)
            net.reset(1 + s)
            th = 0.0
            for t in range(settle + steps):
                net.clear_drive()
                net.set_visual_scene(th, 1.0)
                net.inject_goal(goal, 0.3)
                net.step(DT)
                if t >= settle:
                    th = wrap(th + 2.6 * net.read_steering()['turn'] * DT)
            errs.append(float(abs(np.degrees(wrap(th - goal)))))
    return errs


def boot_median(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    m = [float(np.median(rng.choice(x, size=len(x), replace=True))) for _ in range(n)]
    return float(np.median(x)), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def verdict(e):
    if e <= PREREG['ok_max']:
        return 'ok', '目標は実質 FC2 に載っていた。既存結果は有効（注記のみ）'
    if e <= PREREG['partial_max']:
        return 'partial', 'hΔB への注入が部分的に効いていた。既存の数値を再測定して差し替える'
    return 'depends', '操舵は hΔB への注入に依存していた。閉ループ結果の解釈を全面的に見直す'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--goals', type=int, default=8)
    a = ap.parse_args()

    conds = [('従来（FC2 ＋ hΔB）', ['FC2', 'hDeltaB']),
             ('FC2 のみ', ['FC2']),
             ('hΔB のみ', ['hDeltaB'])]

    rows = []
    print(f"{'条件':22s} {'細胞数':>6s} {'方位誤差の中央値':>18s} {'95% CI':>22s}")
    for name, types in conds:
        net = CXNetworkNP(MODEL)
        n = net.set_goal_types(types)
        errs = maze_trials(net, seeds=a.seeds, goals=a.goals)
        med, ci = boot_median(errs)
        rows.append({'name': name, 'goal_types': types, 'n_goal_cells': n,
                     'n_trials': len(errs), 'median_deg': med, 'ci95': ci,
                     'errors_deg': errs})
        print(f'{name:22s} {n:6d} {med:15.1f}° {f"[{ci[0]:.1f}, {ci[1]:.1f}]":>22s}',
              flush=True)

    fc2 = next(r for r in rows if r['name'] == 'FC2 のみ')
    base = next(r for r in rows if r['name'].startswith('従来'))
    v, why = verdict(fc2['median_deg'])
    print(f"\n事前登録の判定: [{v}] {why}")
    print(f"  従来 {base['median_deg']:.1f}° → FC2 のみ {fc2['median_deg']:.1f}° "
          f"（差 {fc2['median_deg'] - base['median_deg']:+.1f}°）")

    res = {'prereg': PREREG, 'protocol': {'seeds': a.seeds, 'goals': a.goals,
                                          'steps': 700, 'settle': 150, 'dt': DT},
           'rows': rows, 'verdict': v, 'verdict_why': why,
           'fc2_only_median_deg': fc2['median_deg'],
           'baseline_median_deg': base['median_deg']}
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f'→ {a.out}')


if __name__ == '__main__':
    main()
