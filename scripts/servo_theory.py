#!/usr/bin/env python3
"""理想化した閉ループで、制御理論の予測を数値で確かめる。

  python3 scripts/servo_theory.py
  → data/servo_theory.json

**ここでは回路（connectome）を一切通さない。** 扱うのは

    プラント   Ḣ = K·u,   T = H + φ
    制御則     u = K_H·sin(G − H) + K_T·sin(G − T) + c

だけで、K_H・K_T・c には実測値（data/servo_identify.json）と、
GPT-6 Astra が報告した値（再現していない）を入れる。
docs/control_theory.md の導出と、docs/PAPER_SERVO.*.md の 5・6 章が言う数値の根拠になる。

節の対応
--------
  A  一定の横滑りでの定常偏差 — 厳密解と (1−ρ)φ の線形近似       （5.4 節）
  B  安定条件 K_H cosφ + K_T > 0 と、符号が反転したときの π 側の平衡点（5.5 節）
  C  前段：4 基底のフェーザ和が e^{i(H+φ)} になる                   （5.6 節）
  D  後段：二乗の比較器が sin(G − T) を出す（12 細胞の離散和で厳密）   （5.7 節）
  E  PLL：一定速度で回る横滑りへの追従誤差 arcsin(φ̇/K) とロック範囲   （5.8 節）
  F  世界に固定された風：1 + ∂φ/∂H > 0 が崩れると到達できない方向が出る（5.9 節）
  G  実験 4 と同じ OU 横滑りでの誤差 — 理論が説明できる量              （6 章）

G は data/bearing_vs_homing.json の合成モデルの値（7.8° / 40.0°）を
**同じ種・同じ手順で再現すること**を最初に確かめる。再現できなければ比較に意味が無い。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'data', 'servo_theory.json')
TAU = 2 * np.pi

# 実験 4（scripts/bearing_vs_homing.py）と同じ課題の定数。変えない。
DT, K_PLANT, SETTLE = 0.02, 2.6, 150
SIGMA_DEG, TAU_S, MAX_DEG = 55.0, 4.0, 60.0
SEED, TRIALS, STEPS = 20260919, 12, 900
GOALS_DEG = (30, 90, 150, 210, 270, 330)

# 制御則の係数。出典を混ぜないこと。
GAINS = {
    'plain': {'K_H': 0.7483, 'K_T': -0.0185, 'c': -0.0301,
              'source': 'data/servo_identify.json（実測・分流なし）'},
    'augmented': {'K_H': 0.060, 'K_T': 0.703, 'c': 0.0,
                  'source': 'GPT-6 Astra の報告（再現していない。c は報告なしのため 0）'},
    'ideal_heading': {'K_H': 1.0, 'K_T': 0.0, 'c': 0.0, 'source': '理想'},
    'ideal_travel': {'K_H': 0.0, 'K_T': 1.0, 'c': 0.0, 'source': '理想'},
}


# GPT-6 Astra が報告した値（再現していない）。理論の予測と突き合わせるためだけに使う。
REPORTED = {'augmented_sideslip_median_deg': 15.14, 'augmented_sideslip_p90_deg': 36.48,
            'flipped_rho': 0.978, 'flipped_err_deg': 160.49}


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def rho(KH, KT):
    return abs(KT) / (abs(KH) + abs(KT))


# ------------------------------------------------------------------ A・B 定常偏差と安定性

def equilibria(KH, KT, phi, n=7201):
    """K_H·sin(e+φ) + K_T·sin e = 0 の根と、その安定性。

    e = G − T。φ が一定なら ė = −K·[K_H sin(e+φ) + K_T sin e]。
    根で f'(e) = K_H cos(e+φ) + K_T cos e > 0 なら安定。
    """
    # 格子を半目ずらし、端をつないで周期的に走査する（e = ±π の根を取りこぼさないため。
    # 以前は端点ちょうどに根があると符号変化を検出できず、φ = 0 の π の根を落としていた）
    step = TAU / n
    e = -np.pi + (np.arange(n) + 0.5) * step
    f = KH * np.sin(e + phi) + KT * np.sin(e)
    roots = []
    for i in range(n):
        j = (i + 1) % n
        if f[i] == 0 or f[i] * f[j] < 0:
            a = e[i]
            b = a + step
            for _ in range(60):   # 二分法
                m = 0.5 * (a + b)
                fm = KH * np.sin(m + phi) + KT * np.sin(m)
                fa = KH * np.sin(a + phi) + KT * np.sin(a)
                if fa * fm <= 0:
                    b = m
                else:
                    a = m
            r = float(wrap(0.5 * (a + b)))
            slope = KH * np.cos(r + phi) + KT * np.cos(r)
            roots.append({'e': float(r), 'stable': bool(slope > 0)})
    return roots


def stable_error(KH, KT, phi):
    """安定な平衡点の |e|（度）。無ければ None。"""
    st = [r for r in equilibria(KH, KT, phi) if r['stable']]
    if not st:
        return None
    return float(min(abs(np.degrees(wrap(r['e']))) for r in st))


def section_a():
    rows = []
    for name in ('plain', 'augmented'):
        g = GAINS[name]
        for phi_deg in (5, 10, 20, 40, 60):
            phi = np.radians(phi_deg)
            exact = stable_error(g['K_H'], g['K_T'], phi)
            lin = abs((g['K_H'] / (g['K_H'] + g['K_T'])) * phi_deg)
            rows.append({'model': name, 'phi_deg': phi_deg, 'exact_deg': exact,
                         'linear_deg': lin, 'one_minus_rho': 1 - rho(g['K_H'], g['K_T'])})
    return {'rows': rows,
            'note': '線形近似 |e*| = K_H/(K_H+K_T)·|φ|。K_H, K_T > 0 のときだけ (1−ρ)|φ| に等しい。'}


def section_b():
    """符号が反転した例。ρ = 0.978 で K_T < 0。

    GPT-6 の報告にあるのは ρ = 0.978・K_T < 0・誤差 160.49° だけで、K_H の値は無い。
    ここでは ρ を合わせた代表値を置く（**報告値の再現ではない**）。
    """
    out = {}
    for label, KH, KT in (('flipped (ρ=0.978, K_T<0)', 0.0155, -0.689),
                          ('augmented', 0.060, 0.703)):
        eq = {}
        for phi_deg in (0, 20, 40):
            eq[str(phi_deg)] = equilibria(KH, KT, np.radians(phi_deg))
        errs = [stable_error(KH, KT, np.radians(p)) for p in range(-60, 61, 5)]
        errs = [e for e in errs if e is not None]
        out[label] = {'K_H': KH, 'K_T': KT, 'rho': rho(KH, KT),
                      'sum_positive': KH + KT > 0, 'equilibria': eq,
                      'median_stable_err_deg': float(np.median(errs))}
    out['note'] = ('K_H + K_T < 0 だと e = 0 が不安定になり、安定点は e ≈ π 側へ移る。'
                   '符号の反転だけで説明できるのは「≈180° 付近」までで、報告の 160.49° との差は'
                   'ここでは説明しない。')
    return out


# ------------------------------------------------------------------ C・D 前段と後段の演算

def phasor_sum(H, phi, prefs_deg=(45, -45, -135, 135), speed=1.0):
    p = np.radians(prefs_deg)
    A = speed * np.maximum(0.0, np.cos(phi - p))
    return np.sum(A * np.exp(1j * (H + p)))


def section_c():
    errs, mags = [], []
    for H in np.radians(np.arange(0, 360, 15)):
        for phi in np.radians(np.arange(-180, 180, 5)):
            z = phasor_sum(H, phi)
            errs.append(abs(np.degrees(wrap(np.angle(z) - (H + phi)))))
            mags.append(abs(z))
    return {'max_phase_err_deg': float(max(errs)),
            'magnitude_min': float(min(mags)), 'magnitude_max': float(max(mags))}


def pfl3_comparator(G, T, n_cells=12, delta_deg=90.0, square=True):
    """PFL3 の補助入力を左右で足し合わせ、(R − L)/(R + L) を返す。

    F = [cos(G − γ) + cos(T − γ − δ)]²，L は δ = +90°，R は δ = −90°。
    square=False は二乗を外したもの（線形の比較）。
    """
    gam = np.arange(n_cells) * TAU / n_cells

    def side(delta):
        x = np.cos(G - gam) + np.cos(T - gam - np.radians(delta))
        return np.sum(x ** 2) if square else np.sum(x)

    L, R = side(+delta_deg), side(-delta_deg)
    den = R + L
    return (R - L) / den if abs(den) > 1e-12 else 0.0, L, R


def section_d():
    err_sq, lin_abs = [], []
    for G in np.radians(np.arange(0, 360, 10)):
        for T in np.radians(np.arange(0, 360, 10)):
            u, _, _ = pfl3_comparator(G, T, square=True)
            err_sq.append(abs(u - np.sin(G - T)))
            _, L, R = pfl3_comparator(G, T, square=False)
            lin_abs.append(max(abs(L), abs(R)))
    return {'max_abs_err_squared': float(max(err_sq)),
            'max_abs_linear_side_sum': float(max(lin_abs)),
            'note': '二乗ありで (R−L)/(R+L) = sin(G−T) が 12 細胞の離散和で厳密に成り立つ。'
                    '二乗を外すと、細胞位相で和をとった左右の入力はどちらも恒等的に 0 になり、'
                    'G−T の情報が残らない。'}


# ------------------------------------------------------------------ E  PLL のランプ追従

def run_ramp(KH, KT, rate, K=K_PLANT, T_s=60.0, dt=DT, G=0.0):
    """横滑りが一定速度 rate [rad/s] で回るとき。後半の誤差と、周回（cycle slip）の回数。"""
    n = int(T_s / dt)
    H, phi = 0.0, 0.0
    errs, unwrapped, prev = [], 0.0, 0.0
    for i in range(n):
        T = H + phi
        u = KH * np.sin(wrap(G - H)) + KT * np.sin(wrap(G - T))
        H = H + K * u * dt
        phi = phi + rate * dt
        e = wrap(G - (H + phi))
        d = wrap(e - prev)
        unwrapped += d
        prev = e
        if i > n // 2:
            errs.append(e)
    slips = abs(unwrapped) / TAU
    return float(np.degrees(np.mean(errs))), float(slips)


def section_e():
    Kloop = K_PLANT * 1.0          # 理想の travel servo のループ利得
    rows = []
    for frac in (0.1, 0.3, 0.5, 0.8, 0.95, 1.2, 2.0):
        rate = frac * Kloop
        mean_e, slips = run_ramp(0.0, 1.0, rate)
        pred = float(np.degrees(np.arcsin(frac))) if frac < 1 else None
        rows.append({'rate_over_K': frac, 'rate_deg_s': float(np.degrees(rate)),
                     'mean_err_deg': mean_e, 'pred_deg': pred, 'cycle_slips': slips})
    return {'K_loop': Kloop, 'rows': rows,
            'note': '一次 PLL と同じ ė = −K sin e − φ̇。|φ̇| < K でロックし誤差は arcsin(φ̇/K)。'
                    '|φ̇| > K ではロックが外れ、位相が周回する。'}


# ------------------------------------------------------------------ F  世界に固定された風

def section_f():
    """φ(H) = β·sin(w − H)。T(H) = H + φ(H) が単調であるための条件は β < 1。"""
    out = []
    H = np.linspace(-np.pi, np.pi, 3601)
    for beta in (0.3, 0.7, 0.95, 1.3, 2.0):
        phi = beta * np.sin(0.0 - H)
        dT = 1 + np.gradient(phi, H)
        T = np.unwrap(H + phi)
        reach = []
        for G in np.radians(np.arange(-180, 180, 5)):
            # 閉ループ：travel servo（理想）
            h = 0.0
            for _ in range(3000):
                t = h + beta * np.sin(-h)
                h = h + K_PLANT * np.sin(wrap(G - t)) * DT
            t = h + beta * np.sin(-h)
            reach.append(abs(np.degrees(wrap(G - t))) < 5)
        out.append({'beta': beta, 'min_dT_dH': float(dT.min()),
                    'monotone': bool(dT.min() > 0),
                    'reached_fraction': float(np.mean(reach))})
        del T
    return {'rows': out,
            'note': '風が世界に固定されていると φ は H に依存し、Ṫ = (1 + ∂φ/∂H)·Ḣ。'
                    'β < 1（常に 1 + ∂φ/∂H > 0）なら全方向に到達できる。'}


# ------------------------------------------------------------------ G  実験 4 と同じ課題

def sideslip_track(n, dt, sigma_deg, tau_s, max_deg, rng):
    """bearing_vs_homing.py と同じ OU 過程（同じ乱数の使い方）。"""
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


def run_mixed(KH, KT, c, goal, phi_track, K=K_PLANT, dt=DT, settle=SETTLE):
    H = 0.0
    errs = []
    for i, phi in enumerate(phi_track):
        T = H + phi
        u = KH * np.sin(wrap(goal - H)) + KT * np.sin(wrap(goal - T)) + c
        H = wrap(H + K * u * dt)
        if i >= settle:
            errs.append(abs(np.degrees(wrap(T - goal))))
    return errs


def tracks(sigma):
    rng = np.random.default_rng(SEED)
    return [sideslip_track(STEPS, DT, sigma, TAU_S, MAX_DEG, rng) for _ in range(TRIALS)]


def linear_sigma_e(KH, KT, K=K_PLANT, sigma_deg=SIGMA_DEG, tau_s=TAU_S):
    """線形化した閉ループで、OU 横滑りに対する誤差の標準偏差（度）。

    e(s) = −(s + K·K_H)/(s + K·(K_H+K_T)) · φ(s)。φ のスペクトルは 2σ²a/(ω²+a²)、a = 1/τ。
    Var e = σ²·[A + B·a/α]、A = (β²−a²)/(α²−a²)、B = (α²−β²)/(α²−a²)、
    α = K(K_H+K_T)、β = K·K_H。クリップ（|φ| ≤ 60°）は入れていない。
    """
    a = 1.0 / tau_s
    al, be = K * (KH + KT), K * KH
    if al <= 0:
        return None
    A = (be ** 2 - a ** 2) / (al ** 2 - a ** 2)
    B = (al ** 2 - be ** 2) / (al ** 2 - a ** 2)
    return float(sigma_deg * np.sqrt(max(0.0, A + B * a / al)))


def section_g():
    goals = [np.radians(g) for g in GOALS_DEG]
    tr_ss, tr_0 = tracks(SIGMA_DEG), tracks(0.0)
    res = {}
    for name, g in GAINS.items():
        row = {'K_H': g['K_H'], 'K_T': g['K_T'], 'c': g['c'], 'source': g['source'],
               'rho': rho(g['K_H'], g['K_T'])}
        for key, trs in (('sideslip', tr_ss), ('no_sideslip', tr_0)):
            errs = []
            for k, t in enumerate(trs):
                errs += run_mixed(g['K_H'], g['K_T'], g['c'], goals[k % len(goals)], t)
            row[key] = {'median_deg': float(np.median(errs)),
                        'p90_deg': float(np.percentile(errs, 90))}
        sd = linear_sigma_e(g['K_H'], g['K_T'])
        row['linear_sigma_deg'] = sd
        row['linear_median_deg'] = None if sd is None else float(0.6745 * sd)
        res[name] = row

    # 定常成分だけ：φ を試行ごとの一定値に置き換えた (1−ρ)|φ| の中央値
    phis = np.abs(np.concatenate([t[SETTLE:] for t in tr_ss]))
    for name in ('plain', 'augmented'):
        g = GAINS[name]
        res[name]['static_component_median_deg'] = float(
            np.median(np.degrees(phis)) * g['K_H'] / (g['K_H'] + g['K_T']))
    res['median_abs_phi_deg'] = float(np.median(np.degrees(phis)))

    # 理想の travel servo：ループ利得を振ると追従誤差がどう減るか
    sweep = []
    for Kl in (0.5, 1.0, 2.0, 2.6, 5.0, 10.0, 20.0):
        errs = []
        for k, t in enumerate(tr_ss):
            errs += run_mixed(0.0, 1.0, 0.0, goals[k % len(goals)], t, K=Kl)
        sweep.append({'K_loop': Kl, 'median_deg': float(np.median(errs)),
                      'linear_median_deg': float(0.6745 * linear_sigma_e(0.0, 1.0, K=Kl))})
    res['travel_gain_sweep'] = sweep
    res['budget'] = error_budget(res)
    return res


def error_budget(res):
    """観測値を「制御則で説明できる部分」と「それ以外の床」に分ける。

    **仮定**：二つの成分が独立で、中央値が二乗和の平方根（RSS）で合成される。
    これは近似であって導出ではない。素の回路で当てはまりを確かめてから、
    追加モデルの床を逆算し、「横滑りなしの誤差」の予測として出す。
    """
    bh = json.load(open(os.path.join(ROOT, 'data', 'bearing_vs_homing.json'), encoding='utf-8'))
    net = bh['network']
    obs_ss = net['横滑りあり（σ=55°）']
    obs_0 = net['横滑りなし（σ=0°、対照）']
    p = res['plain']
    floor = float(np.sqrt(max(0.0, obs_0['median_deg'] ** 2 - p['no_sideslip']['median_deg'] ** 2)))
    pred_ss = float(np.sqrt(p['sideslip']['median_deg'] ** 2 + floor ** 2))
    a = res['augmented']
    obs_aug = REPORTED['augmented_sideslip_median_deg']
    floor_aug = float(np.sqrt(max(0.0, obs_aug ** 2 - a['sideslip']['median_deg'] ** 2)))
    return {
        'assumption': '独立な二成分の中央値が RSS で合成される（近似）',
        'plain': {'observed_sideslip_deg': obs_ss['median_deg'], 'observed_ci95': obs_ss['ci95'],
                  'observed_no_sideslip_deg': obs_0['median_deg'],
                  'law_sideslip_deg': p['sideslip']['median_deg'],
                  'law_no_sideslip_deg': p['no_sideslip']['median_deg'],
                  'floor_deg': floor, 'rss_prediction_sideslip_deg': pred_ss,
                  'prediction_within_ci': bool(obs_ss['ci95'][0] <= pred_ss <= obs_ss['ci95'][1])},
        'augmented': {'observed_sideslip_deg': obs_aug,
                      'law_sideslip_deg': a['sideslip']['median_deg'],
                      'static_component_deg': a['static_component_median_deg'],
                      'implied_floor_deg': floor_aug,
                      'predicted_no_sideslip_deg': floor_aug,
                      'note': '追加モデルの「横滑りなし」の誤差は報告されていない。'
                              'RSS の仮定が正しければ ≈ implied_floor_deg になるはず（検証可能な予測）。'},
    }


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()

    out = {'note': '回路を通さない理想化した閉ループ。K_H・K_T の出典は GAINS に記す。',
           'task': {'dt': DT, 'K_plant': K_PLANT, 'settle': SETTLE, 'sigma_deg': SIGMA_DEG,
                    'tau_s': TAU_S, 'max_deg': MAX_DEG, 'seed': SEED, 'trials': TRIALS,
                    'steps': STEPS, 'goals_deg': list(GOALS_DEG)},
           'gains': GAINS, 'reported': REPORTED}
    out['A_static'] = section_a()
    out['B_stability'] = section_b()
    out['C_phasor'] = section_c()
    out['D_comparator'] = section_d()
    out['E_pll'] = section_e()
    out['F_worldwind'] = section_f()
    out['G_task'] = section_g()

    g = out['G_task']
    print('=== G 実験 4 と同じ課題（中央値 / 90 パーセンタイル）')
    for name in GAINS:
        r = g[name]
        lm = r['linear_median_deg']
        print(f"  {name:14s} ρ={r['rho']:.3f}  横滑りあり {r['sideslip']['median_deg']:5.1f}° / "
              f"{r['sideslip']['p90_deg']:5.1f}°   なし {r['no_sideslip']['median_deg']:5.1f}°   "
              f"線形予測 {'-' if lm is None else f'{lm:.1f}°'}")
    print(f"  定常成分 (1−ρ)|φ|: plain {g['plain']['static_component_median_deg']:.1f}°, "
          f"augmented {g['augmented']['static_component_median_deg']:.1f}°")
    bd = g['budget']
    print(f"  誤差予算 plain: 床 {bd['plain']['floor_deg']:.1f}° → RSS 予測 "
          f"{bd['plain']['rss_prediction_sideslip_deg']:.1f}°（観測 {bd['plain']['observed_sideslip_deg']:.1f}°、"
          f"CI 内 {bd['plain']['prediction_within_ci']}）")
    print(f"  誤差予算 augmented: 制御則 {bd['augmented']['law_sideslip_deg']:.1f}° / 観測 15.14° → "
          f"床 {bd['augmented']['implied_floor_deg']:.1f}°（＝横滑りなしの予測）")
    print('  travel servo の利得掃引:',
          ', '.join(f"K={s['K_loop']}: {s['median_deg']:.1f}°" for s in g['travel_gain_sweep']))
    print('=== A', [(r['model'], r['phi_deg'], round(r['exact_deg'], 2), round(r['linear_deg'], 2))
                    for r in out['A_static']['rows']])
    print('=== B', {k: (v['median_stable_err_deg'], v['sum_positive'])
                    for k, v in out['B_stability'].items() if isinstance(v, dict)})
    print('=== C', out['C_phasor'])
    print('=== D', {k: v for k, v in out['D_comparator'].items() if k != 'note'})
    print('=== E', [(r['rate_over_K'], round(r['mean_err_deg'], 2), r['pred_deg'] and round(r['pred_deg'], 2),
                     round(r['cycle_slips'], 2)) for r in out['E_pll']['rows']])
    print('=== F', [(r['beta'], r['monotone'], r['reached_fraction']) for r in out['F_worldwind']['rows']])

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f'→ {a.out}')


if __name__ == '__main__':
    main()
