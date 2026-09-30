#!/usr/bin/env python3
"""実験 4 — 方位追従（bearing following）と帰巣（homing）を分けて測る。

  python3 scripts/bearing_vs_homing.py
  → data/bearing_vs_homing.json

なぜ分けるのか（docs/RESEARCH_PLAN_SERVO.md 2.2 節）
---------------------------------------------------
記事の中心的な主張は「hΔB は瞬時の世界座標速度表現であって位置積分器ではない。
だから**経路積分は要らない**」である。ただしこの主張が成り立つ範囲は課題に依る。

  課題 A 方位追従「東へ進み続けろ」   瞬時の進行方向 T があれば足りる。**積分は不要**
  課題 B 帰巣「出発点へ戻れ」         変位の**累積**が要る。積分が必須

本リポジトリの 4 試験（6.7 節）は**課題 B を測っていた**。だから「経路積分が成立しない」と
「記事の主張が正しい」は両立しうる。ここで測るのは**課題 A のほう**である。

課題 A の作法
-------------
閉ループ。頭方位 H は PFL3 の左右差で更新し、横滑り φ は OU 過程で外から与える
（σ = 55°、τ = 4 秒、|φ| ≤ 60°。7 節で診断的だと分かった条件と同じ）。
実際の進行方向は T = H + φ。目標 G は FC2 のみに注入する（実験 0 の交絡を除いた条件）。

  誤差 = 整定後の T と G の円周差の中央値

**横滑りが無い対照（σ = 0）も測る。** そこでは H = T なので heading servo でも解けるはずで、
「この課題が横滑りのときだけ識別的である」ことの確認になる
（7.1 節で「前進のみの課題は識別力が無かった」と同じ論点）。

事前登録した判定（走らせる前に固定）
------------------------------------
  課題 A の合格: 進行方向の誤差の中央値 ≤ 20°（共通の許容誤差。6.7 節と同じ値）

採点系の検定を先にやる。合成モデル 3 つを同じ課題に通す。

  理想のベクトル帰還制御 u = K·sin(G − T)   → 解ける（≤ 20°）
  理想の heading servo   u = K·sin(G − H)   → **解けない**（誤差 ≈ 横滑りの大きさ）
  制御しない（u = 0）                        → 解けない

**この検定が予想どおりに出て初めて、実回路を通す意味がある。**

課題 B について
---------------
課題 B は 6.7 節の 4 試験（蓄積・保持・打ち消し・経路同等）と**同じものを測る**。
新しい採点系を作ると「基準を作り直して通した」ことになるので、ここでは走らせず、
既存の結果（`data/pi_benchmark.json` / `data/pi_benchmark_shunt.json`）を参照する。
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

from cxnet_np import CXNetworkNP            # noqa: E402
from coord_transform import BodyVelocityInput  # noqa: E402

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
OUT = os.path.join(ROOT, 'data', 'bearing_vs_homing.json')
DT = 0.02
TAU = 2 * np.pi

# 事前登録した合格基準。6.7 節の共通の許容誤差と同じ値を使う（新しい基準は作らない）。
PREREG = {'bearing_err_deg_max': 20.0,
          'sideslip_sigma_deg': 55.0, 'sideslip_tau_s': 4.0,
          'sideslip_max_deg': 60.0}


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def sideslip_track(n, dt, sigma_deg, tau_s, max_deg, rng):
    """横滑りの時系列（OU 過程）。pi_task.py と同じ作り方。"""
    if sigma_deg <= 0:
        return np.zeros(n)
    a = np.exp(-dt / tau_s)
    sig = np.radians(sigma_deg)
    phi = np.zeros(n)
    x = 0.0
    for i in range(n):
        x = a * x + np.sqrt(max(0.0, 1 - a * a)) * sig * rng.standard_normal()
        phi[i] = np.clip(x, -np.radians(max_deg), np.radians(max_deg))
    return phi


# ------------------------------------------------------------------ 合成モデル

def run_synthetic(kind, goal, phi_track, dt=DT, K=2.6, settle=150):
    """理想の制御則で閉ループを回す（回路を通さない）。"""
    H = 0.0
    errs = []
    for i, phi in enumerate(phi_track):
        T = H + phi
        if kind == 'bearing':
            u = np.sin(wrap(goal - T))
        elif kind == 'heading':
            u = np.sin(wrap(goal - H))
        else:
            u = 0.0
        H = wrap(H + K * u * dt)
        if i >= settle:
            errs.append(abs(np.degrees(wrap(T - goal))))
    return errs


# ------------------------------------------------------------------ 実回路

def run_network(net, bvi, goal, phi_track, dt=DT, K=2.6, settle=150, speed=1.0):
    net.reset(1)
    H = 0.0
    errs = []
    for i, phi in enumerate(phi_track):
        net.clear_drive()
        net.set_visual_scene(H, 1.0)
        net.inject_goal(goal, 0.3)
        bvi.apply(net.clamp_val, speed, phi)
        net.step(dt)
        T = H + phi
        H = wrap(H + K * net.read_steering()['turn'] * dt)
        if i >= settle:
            errs.append(abs(np.degrees(wrap(T - goal))))
    return errs


def boot_median(x, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    m = [float(np.median(rng.choice(x, size=len(x), replace=True))) for _ in range(n)]
    return float(np.median(x)), [float(np.percentile(m, 2.5)),
                                 float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    ap.add_argument('--trials', type=int, default=12)
    ap.add_argument('--steps', type=int, default=900)
    a = ap.parse_args()

    goals = [np.radians(g) for g in (30, 90, 150, 210, 270, 330)]
    conds = [('横滑りあり（σ=55°）', PREREG['sideslip_sigma_deg']),
             ('横滑りなし（σ=0°、対照）', 0.0)]

    out = {'prereg': PREREG,
           'protocol': {'trials': a.trials, 'steps': a.steps, 'dt': DT,
                        'settle': 150, 'goals_deg': [30, 90, 150, 210, 270, 330]},
           'note': '課題 B（帰巣）は 6.7 節の 4 試験と同じものなので、ここでは走らせず'
                   'data/pi_benchmark.json を参照する（基準を作り直さないため）。',
           'synthetic': {}, 'network': {}}

    # 横滑りの時系列は条件間で共有する（同じ外乱で比べるため）
    tracks = {}
    for name, sigma in conds:
        rng = np.random.default_rng(20260919)
        tracks[name] = [sideslip_track(a.steps, DT, sigma, PREREG['sideslip_tau_s'],
                                       PREREG['sideslip_max_deg'], rng)
                        for _ in range(a.trials)]

    # --- 採点系の対照を先に
    print('=== 採点系の対照（合成モデル）: 進行方向の誤差の中央値')
    print(f"{'モデル':22s} {'横滑りあり':>16s} {'横滑りなし':>16s}  判定")
    expect = {'bearing': (True, True), 'heading': (False, True), 'none': (False, False)}
    ok_all = True
    for kind in ('bearing', 'heading', 'none'):
        res = {}
        for name, _ in conds:
            errs = []
            for k, tr in enumerate(tracks[name]):
                errs += run_synthetic(kind, goals[k % len(goals)], tr)
            med, ci = boot_median(errs)
            res[name] = {'median_deg': med, 'ci95': ci,
                         'pass': med <= PREREG['bearing_err_deg_max']}
        got = tuple(res[n]['pass'] for n, _ in conds)
        good = got == expect[kind]
        ok_all &= good
        out['synthetic'][kind] = {'by_cond': res, 'expected': expect[kind],
                                  'passes_control': good}
        cells = [f"{res[n]['median_deg']:.1f}° {'✓' if res[n]['pass'] else '✗'}"
                 for n, _ in conds]
        print(f"{kind:22s} {cells[0]:>16s} {cells[1]:>16s}  "
              f"{'OK' if good else '期待と違う'}")

    if not ok_all:
        print('\n**採点系が合成モデルを区別できていない。実回路を通す前に直すこと。**')
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        return

    # --- 実回路
    print('\n=== 実回路（目標は FC2 のみに注入）')
    for name, _ in conds:
        net = CXNetworkNP(MODEL)
        net.set_goal_types(['FC2'])
        bvi = BodyVelocityInput(net)
        errs = []
        for k, tr in enumerate(tracks[name]):
            errs += run_network(net, bvi, goals[k % len(goals)], tr)
            print(f'    試行 {k + 1}/{len(tracks[name])}', flush=True)
        med, ci = boot_median(errs)
        passed = med <= PREREG['bearing_err_deg_max']
        out['network'][name] = {'median_deg': med, 'ci95': ci, 'pass': passed,
                                'n_samples': len(errs)}
        print(f"  {name:24s} {med:6.1f}°  CI [{ci[0]:.1f}, {ci[1]:.1f}]  "
              f"{'合格' if passed else '不合格'}")

    # --- 課題 B は既存の結果を参照する
    try:
        bench = json.load(open(os.path.join(ROOT, 'data', 'pi_benchmark.json'),
                               encoding='utf-8'))
        out['homing_reference'] = {
            'source': 'data/pi_benchmark.json（6.7 節の 4 試験）',
            'note': '新しい採点系を作らないため、ここでは再実行していない。'}
        del bench
    except Exception as e:  # noqa: BLE001
        out['homing_reference'] = {'error': str(e)}

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
