#!/usr/bin/env python3
"""実験 5 — 理想の hΔB を注入して、「配線の不足」と「動作点の不足」を切り分ける。

  python3 scripts/servo_ideal_hdb.py
  → data/servo_ideal_hdb.json

何が未決着だったか
------------------
実験 1（7.3 節）で実回路は Model A だった（ρ = 0.024）。だが、その原因は 2 つありうる。

  (i) **配線の不足**   hΔB → PFL3 が総入力の 0.09% しかない（3.1 節）ので、
                        仮に hΔB が進行方向を運んでいても操舵へ届かない
  (ii) **動作点の不足** hΔB がそもそも進行方向を運んでいない（6.9〜6.10 節の b ≈ 0）ので、
                        配線が支えていても運ぶ信号が無い

実験 1 はこの 2 つを切り分けていない。切り分けるには **(ii) を外から埋めて (i) だけ残す**。
つまり 7 章の数理モデルが出す理想の hΔB を、実回路の hΔB 細胞にそのまま書き込み、
下流（hΔI/hΔA → PFL3、および hΔB → PFL3 の直接路）が travel servo を作るかを見る。

これは 6.7 節で「PFN を黙らせて理想の速度ベクトルを hΔ へ直接注入した」のと同じ作法である。

注入するもの
------------
世界座標の進行方向 ψ = θ + φ にピークを持つ半波整流の正弦波を、
hΔB の各細胞へその位相に応じて書き込む（クランプする）。

    r_i = A · [cos(phase_i − ψ)]₊

A は振幅で、0.2 / 0.5 / 1.0 を振る。**これはモデルへの外部からの注入であって、
回路が作った信号ではない。** hΔB 以外は一切触らない。

対照（偽陽性の検出。これが無いと結果を信用できない）
----------------------------------------------------
  travel   ψ = θ + φ を書き込む（本命）
  heading  ψ = θ を書き込む。**進行方向の情報を含まない。** ここで ρ が上がったら
           それは φ 以外の経路で u が動いているということで、実験のほうが壊れている
  shuffled ψ = θ + φ だが、hΔB の位相ラベルを固定の順列で入れ替えて書き込む。
           **同じ量・同じ分布の活動を、位相地図だけ壊して入れる。**
           ここで ρ が上がったら、効いているのは位相地図ではないということになる

事前登録した判定（走らせる前に固定）
------------------------------------
**実験 1 と同じ採点系・同じ閾値**を使う（ρ < 0.2 / R² ≥ 0.5 / Δu/U ≥ 0.2）。
新しい基準は作らない。

  travel 条件で R² ≥ 0.5 かつ ρ ≥ 0.2
      → **配線は travel servo を支えられる。足りないのは動作点。**
        記事の描像は「hΔB が進行方向を運べば成立する」ところまで絞られる
  travel 条件でも ρ < 0.2
      → **配線側が travel servo を支えていない。** 3.1 節の 0.09% と整合する
  heading または shuffled の対照で ρ ≥ 0.2 が出た
      → **実験が壊れている。** 上の判定は下さず、そう報告する
  travel 条件の当てはまりが R² < 0.5
      → **切り分けできず。** 注入そのものが回路を壊している可能性があるので、
        「配線の不足」と読んではいけない。ρ は解釈しない
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
from servo_identify import (PREREG, fit_servo, sideslip_index,  # noqa: E402
                            verdict)

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
OUT = os.path.join(ROOT, 'data', 'servo_ideal_hdb.json')
DT = 0.02
TAU = 2 * np.pi


def hdb_cells(net):
    idx = np.array([i for i in range(net.N)
                    if net.type_of(i).startswith('hDeltaB') and net.phase[i] > -8],
                   dtype=np.int64)
    return idx, net.phase[idx]


def sweep(kind, amp, thetas, goals, phis, steps=300, speed=1.0, seed=7):
    """hΔB に理想の活動を書き込みながら、実験 1 と同じ掃引をする。"""
    net = CXNetworkNP(MODEL)
    net.set_goal_types(['FC2'])          # 実験 0 の交絡を除いた条件
    idx, ph = hdb_cells(net)

    # 位相ラベルを壊す対照（順列は固定。条件間で同じものを使う）
    perm = np.random.default_rng(seed).permutation(len(idx))
    ph_used = ph[perm] if kind == 'shuffled' else ph

    # hΔB をクランプ集合に入れる（この掃引の中だけ。既定のモデルは変えない）
    net.clamped = net.clamped.copy()
    net.clamped[idx] = True

    bvi = BodyVelocityInput(net)
    rows = []
    for t in thetas:
        th = np.radians(t)
        for g in goals:
            gg = np.radians(g)
            for p in phis:
                phi = np.radians(p)
                psi = th + phi if kind in ('travel', 'shuffled') else th
                ideal = amp * speed * np.maximum(0.0, np.cos(ph_used - psi))
                net.reset(1)
                for _ in range(steps):
                    net.clear_drive()
                    net.set_visual_scene(th, 1.0)
                    net.inject_goal(gg, 0.3)
                    bvi.apply(net.clamp_val, speed, phi)
                    net.clamp_val[idx] = ideal      # ここだけが外からの注入
                    net.step(DT)
                s = net.read_steering()
                rows.append({'theta': t, 'goal': g, 'phi': p, 'u': float(s['turn'])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()

    if a.quick:
        thetas, goals, phis, amps = [0, 180], [0, 90, 180, 270], [0, 90, 180, 270], [1.0]
    else:
        thetas = [0, 90, 180, 270]
        goals = list(range(0, 360, 45))
        phis = list(range(0, 360, 45))
        amps = [0.2, 0.5, 1.0]

    out = {'prereg': PREREG,
           'note': 'hΔB への外部からの注入であって、回路が作った信号ではない。'
                   '判定は実験 1 と同じ採点系・同じ閾値を使う（新しい基準は作らない）。',
           'grid': {'theta': thetas, 'goal': goals, 'phi': phis,
                    'amplitudes': amps, 'steps': a.steps},
           'rows': []}

    print(f"{'条件':26s} {'K_H':>8s} {'K_T':>8s} {'ρ':>7s} {'R²':>7s} {'Δu/U':>7s}  判定")
    for amp in amps:
        for kind in ('travel', 'heading', 'shuffled'):
            rows = sweep(kind, amp, thetas, goals, phis, steps=a.steps)
            fit = fit_servo(rows); ss = sideslip_index(rows)
            v, why = verdict(fit, ss)
            name = f'{kind} 振幅 {amp}'
            out['rows'].append({'kind': kind, 'amplitude': amp, 'fit': fit,
                                'sideslip': ss, 'verdict': v, 'verdict_why': why})
            r = ss['ratio']
            print(f"{name:26s} {fit['K_H']:+8.3f} {fit['K_T']:+8.3f} {fit['rho']:7.3f} "
                  f"{fit['r2']:7.3f} {'—' if r is None else f'{r:7.3f}'}  {v}", flush=True)

    # 事前登録した判定
    tr = [r for r in out['rows'] if r['kind'] == 'travel']
    ctrl = [r for r in out['rows'] if r['kind'] in ('heading', 'shuffled')]
    ctrl_bad = [r for r in ctrl
                if r['fit'] and r['fit']['r2'] >= PREREG['r2_min']
                and r['fit']['rho'] >= PREREG['rho_A_max']]
    # 「最良」は、当てはまりが基準を満たす条件の中から選ぶ。
    # 満たす条件が無ければ、当てはまりが最も良いものを診断用に持っておく。
    _ok = [r for r in tr if r['fit'] and r['fit']['r2'] >= PREREG['r2_min']]
    best = (max(_ok, key=lambda r: r['fit']['rho']) if _ok else
            max(tr, key=lambda r: r['fit']['r2'] if r['fit'] else -1))

    # 当てはまりが悪い条件は、そもそも判定の対象にならない（実験 1 と同じ基準）。
    tr_fit = [r for r in tr if r['fit'] and r['fit']['r2'] >= PREREG['r2_min']]
    if ctrl_bad:
        concl = 'broken'
        why = ('対照（heading / shuffled）でも ρ が閾値を超えた。実験が壊れている。'
               f"該当: {[r['kind'] + ' 振幅 ' + str(r['amplitude']) for r in ctrl_bad]}")
    elif not tr_fit:
        # **ここを「配線の不足」と読んではいけない。** 注入そのものが回路を壊していて、
        # u が sin 型の比較器として記述できなくなっているだけかもしれない。
        concl = 'inconclusive'
        r2s = [round(r['fit']['r2'], 3) for r in tr if r['fit']]
        why = (f'travel 条件の当てはまりが全振幅で R² < {PREREG["r2_min"]}（{r2s}）。'
               '事前登録どおり「どのモデルでもない」で、切り分けはできていない。'
               'ρ の値は解釈しない。')
    elif (best['fit'] and best['fit']['r2'] >= PREREG['r2_min']
          and best['fit']['rho'] >= PREREG['rho_A_max']):
        concl = 'operating_point'
        why = ('配線は travel servo を支えられる。足りないのは動作点。'
               f"（travel 振幅 {best['amplitude']} で ρ = {best['fit']['rho']:.3f}）")
    else:
        concl = 'wiring'
        why = ('理想の hΔB を入れても、当てはまりが十分な条件で ρ は閾値未満。'
               f"配線側が travel servo を支えていない。（ρ = {best['fit']['rho']:.3f}、"
               f"R² = {best['fit']['r2']:.3f}）")

    out['conclusion'] = concl
    out['conclusion_why'] = why
    out['best_travel'] = {'amplitude': best['amplitude'], 'fit': best['fit'],
                          'sideslip': best['sideslip']}
    print(f'\n事前登録の判定: [{concl}] {why}')

    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f'→ {a.out}')


if __name__ == '__main__':
    main()
