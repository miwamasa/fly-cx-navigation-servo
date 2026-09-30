#!/usr/bin/env python3
"""実験 1・2 — PFL3 の伝達関数を同定し、3 モデルを弁別する。

  python3 scripts/servo_identify.py
  → data/servo_identify.json

問い（docs/RESEARCH_PLAN_SERVO.md 5 章）
---------------------------------------
PFL3 の旋回指令 u は、頭方位の誤差で決まるのか、進行方向の誤差で決まるのか。

    Model A   u = K_H · sin(G − H)
    Model B'  u = K_T · sin(G − T)          T = H + φ
    Model C   u = K_H · sin(G − H) + K_T · sin(G − T)

手続き
------
θ（頭方位、視覚手がかりで固定）・G（目標、FC2 へ注入）・φ（身体座標の進行方向）を
独立に振り、定常状態で u = (右 PFL3 − 左 PFL3) / (右 + 左) を読む。そして

    u = K_H · sin(G − H) + K_T · sin(G − H − φ) + c

を最小二乗で当てはめる。**この式は制御則の記述であって、配線から導いた式ではない。**
ここでやるのは当てはめであって、実装ではない。

**目標は FC2 のみに注入する。** 実験 0 で「hΔB にも注入していた」交絡が見つかっており、
hΔB に目標が載ったままだと u が hΔB 経由で G に依存してしまい、K_T が偽陽性になる。
実験 0 の結果（FC2 のみで 16.4°、従来 17.9°、CI は重なる）から、この変更で
操舵性能は落ちないことを確認済み。

事前登録した判定（走らせる前に固定。docs/RESEARCH_PLAN_SERVO.md 5 章）
----------------------------------------------------------------------
  ρ = |K_T| / (|K_H| + |K_T|)

  当てはまり R² < 0.5         どのモデルでもない。そう報告して先へ進まない
  R² ≥ 0.5 かつ ρ < 0.2       Model A（heading servo）
  R² ≥ 0.5 かつ ρ > 0.8       Model B'（travel servo）
  R² ≥ 0.5 かつ 0.2 ≤ ρ ≤ 0.8 Model C（混合）

実験 2（横滑り弁別）も同じ掃引から出す。
  Δu = θ と G を固定して φ を振ったときの u の変動幅（θ, G について平均）
  U  = φ = 0 で θ と G を振ったときの u の全振幅
  Δu / U < 0.2 → 進行方向は操舵に効いていない

採点系の対照
------------
合成モデル（理想の heading servo / 理想の travel servo / 半々の混合）を
**同じ掃引と同じ当てはめ**に通し、採点系が三者を区別できることを先に確かめる。
理想の heading servo が ρ ≥ 0.2 を出すようなら、実験のほうが壊れている。
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
OUT = os.path.join(ROOT, 'data', 'servo_identify.json')
DT = 0.02
TAU = 2 * np.pi

# 事前登録した判定の境界。結果を見てから動かさない。
PREREG = {'r2_min': 0.5, 'rho_A_max': 0.2, 'rho_B_min': 0.8, 'du_over_u_min': 0.2}


def wrap(a):
    return (a + np.pi) % TAU - np.pi


# ------------------------------------------------------------------ 当てはめ

def fit_servo(rows):
    """u = K_H·sin(G−H) + K_T·sin(G−H−φ) + c を最小二乗で。"""
    if len(rows) < 4:
        return None
    g = np.radians([r['goal'] for r in rows])
    h = np.radians([r['theta'] for r in rows])
    p = np.radians([r['phi'] for r in rows])
    u = np.array([r['u'] for r in rows], dtype=float)
    X = np.column_stack([np.sin(g - h), np.sin(g - h - p), np.ones(len(u))])
    beta, *_ = np.linalg.lstsq(X, u, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((u - pred) ** 2))
    ss_tot = float(np.sum((u - u.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    kh, kt, c = (float(b) for b in beta)
    denom = abs(kh) + abs(kt)
    return {'K_H': kh, 'K_T': kt, 'c': c, 'r2': float(r2),
            'rho': float(abs(kt) / denom) if denom > 1e-12 else None,
            'u_range': float(u.max() - u.min()), 'n': len(u)}


def sideslip_index(rows):
    """実験 2 の Δu / U。"""
    by_tg = {}
    for r in rows:
        by_tg.setdefault((r['theta'], r['goal']), []).append(r['u'])
    dus = [max(v) - min(v) for v in by_tg.values() if len(v) > 1]
    at0 = [r['u'] for r in rows if r['phi'] == 0]
    U = (max(at0) - min(at0)) if len(at0) > 1 else 0.0
    du = float(np.mean(dus)) if dus else 0.0
    return {'delta_u': du, 'U': float(U),
            'ratio': float(du / U) if U > 1e-9 else None}


def verdict(fit, ss):
    if fit is None:
        return 'none', '当てはめができない'
    if fit['r2'] < PREREG['r2_min']:
        return 'none', f"当てはまり R² = {fit['r2']:.2f} < {PREREG['r2_min']} — どのモデルでもない"
    rho = fit['rho']
    if rho is None:
        return 'none', '利得がどちらも 0'
    if rho < PREREG['rho_A_max']:
        return 'A', 'Model A（heading servo）'
    if rho > PREREG['rho_B_min']:
        return "B'", "Model B'（travel servo）"
    return 'C', 'Model C（混合）'


# ------------------------------------------------------------------ 合成モデル

def synthetic(kind, thetas, goals, phis):
    """採点系の対照。理想の制御則から u を作る（回路を通さない）。"""
    rows = []
    for t in thetas:
        for g in goals:
            for p in phis:
                eh = np.radians(g - t)
                et = np.radians(g - t - p)
                u = {'heading': np.sin(eh), 'travel': np.sin(et),
                     'mixed': 0.5 * np.sin(eh) + 0.5 * np.sin(et)}[kind]
                rows.append({'theta': t, 'goal': g, 'phi': p, 'u': float(u)})
    return rows


# ------------------------------------------------------------------ 実回路

def sweep_network(thetas, goals, phis, steps=300, speed=1.0,
                  goal_types=('FC2',), goal_gain=0.3, shunt_frac=0.0):
    net = CXNetworkNP(MODEL)
    net.set_goal_types(list(goal_types))
    if shunt_frac:
        net.shunt_frac = shunt_frac
    bvi = BodyVelocityInput(net)
    rows = []
    n = len(thetas) * len(goals) * len(phis)
    k = 0
    for t in thetas:
        th = np.radians(t)
        for g in goals:
            gg = np.radians(g)
            for p in phis:
                ph = np.radians(p)
                net.reset(1)
                for _ in range(steps):
                    net.clear_drive()
                    net.set_visual_scene(th, 1.0)
                    net.inject_goal(gg, goal_gain)
                    bvi.apply(net.clamp_val, speed, ph)
                    net.step(DT)
                s = net.read_steering()
                rows.append({'theta': t, 'goal': g, 'phi': p,
                             'u': float(s['turn']), 'left': s['left'],
                             'right': s['right']})
                k += 1
                if k % 32 == 0:
                    print(f'    {k}/{n}', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--quick', action='store_true', help='格子を粗くする（動作確認用）')
    a = ap.parse_args()

    if a.quick:
        thetas, goals, phis = [0, 180], [0, 90, 180, 270], [0, 90, 180, 270]
    else:
        thetas = [0, 90, 180, 270]
        goals = list(range(0, 360, 45))
        phis = list(range(0, 360, 45))

    out = {'prereg': PREREG,
           'grid': {'theta': thetas, 'goal': goals, 'phi': phis,
                    'steps': a.steps, 'speed': 1.0},
           'note': '目標は FC2 のみに注入（実験 0 の交絡を除いた条件）。'
                   '当てはめる式は制御則の記述であって、配線から導いた式ではない。',
           'synthetic': {}, 'network': {}}

    # --- 採点系の対照を先に
    print('=== 採点系の対照（合成モデル）')
    print(f"{'モデル':10s} {'K_H':>8s} {'K_T':>8s} {'ρ':>7s} {'R²':>7s} "
          f"{'Δu/U':>7s}  判定")
    for kind, want in (('heading', 'A'), ('travel', "B'"), ('mixed', 'C')):
        rows = synthetic(kind, thetas, goals, phis)
        fit = fit_servo(rows); ss = sideslip_index(rows)
        v, why = verdict(fit, ss)
        ok = '✓' if v == want else f'✗ 期待 {want}'
        out['synthetic'][kind] = {'fit': fit, 'sideslip': ss, 'verdict': v,
                                  'expected': want, 'passes_control': v == want}
        print(f"{kind:10s} {fit['K_H']:+8.3f} {fit['K_T']:+8.3f} {fit['rho']:7.3f} "
              f"{fit['r2']:7.3f} {ss['ratio']:7.3f}  {v}  {ok}")

    if not all(v['passes_control'] for v in out['synthetic'].values()):
        print('\n**採点系が合成モデルを区別できていない。実回路を通す前に直すこと。**')
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        return

    # --- 実回路
    conds = [('実測（分流なし）', 0.0), ('分流 f=1.0', 1.0)]
    print(f'\n=== 実回路（{len(thetas)}×{len(goals)}×{len(phis)} = '
          f'{len(thetas)*len(goals)*len(phis)} 条件、各 {a.steps} ステップ）')
    for name, sf in conds:
        print(f'  [{name}]', flush=True)
        rows = sweep_network(thetas, goals, phis, steps=a.steps, shunt_frac=sf)
        fit = fit_servo(rows); ss = sideslip_index(rows)
        v, why = verdict(fit, ss)
        out['network'][name] = {'shunt_frac': sf, 'fit': fit, 'sideslip': ss,
                                'verdict': v, 'verdict_why': why, 'rows': rows}
        if fit:
            print(f"    K_H {fit['K_H']:+.3f}  K_T {fit['K_T']:+.3f}  "
                  f"ρ {fit['rho']:.3f}  R² {fit['r2']:.3f}  "
                  f"Δu/U {ss['ratio'] if ss['ratio'] is None else round(ss['ratio'], 3)}")
            print(f'    → {why}')

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
