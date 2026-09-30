#!/usr/bin/env python3
"""制御理論の予測（docs/control_theory.md、docs/PAPER_SERVO.*.md の 5・6 章）を固定するテスト。

回路は通さない。numpy だけで動く（Mac でも走らせられるように scipy を使わない）。

  python3 tests/test_servo_theory.py

control_theory.md が挙げた 4 ケースに、PLL・世界固定風・課題の再現の 3 つを足している。
  1. φ = 0 で heading servo と travel servo の軌跡が一致する
  2. 一定の φ での定常偏差が (1−ρ)φ（小角度）に一致する
  3. φ を全周スイープしたフェーザ和の位相が H + φ（b = 1）
  4. K_H + K_T の符号を反転させると平衡点が π 側へ移る
  5. PLL：一定速度の横滑りへの追従誤差が arcsin(φ̇/K)、|φ̇| > K で周回する
  6. 世界固定風：β < 1 なら全方向に到達できる
  7. 実験 4 の合成モデルの値を、同じ種でビット単位に再現する
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import servo_theory as st  # noqa: E402

DATA = json.load(open(os.path.join(ROOT, 'data', 'servo_theory.json'), encoding='utf-8'))


class TestClosedLoop(unittest.TestCase):

    def test_1_no_sideslip_heading_equals_travel(self):
        """φ = 0 なら T = H なので、二つの制御則は同じ軌跡を描く"""
        phi = np.zeros(600)
        for g in np.radians([30, 150, 270]):
            a = st.run_mixed(1.0, 0.0, 0.0, g, phi)
            b = st.run_mixed(0.0, 1.0, 0.0, g, phi)
            self.assertTrue(np.allclose(a, b, atol=1e-12))

    def test_2_steady_state_is_one_minus_rho_phi(self):
        """K_H, K_T > 0 のとき、小角度で |e*| = (1−ρ)|φ|"""
        KH, KT = 0.060, 0.703
        for phi_deg in (2, 5, 10):
            exact = st.stable_error(KH, KT, np.radians(phi_deg))
            pred = (1 - st.rho(KH, KT)) * phi_deg
            self.assertAlmostEqual(exact, pred, delta=0.03 * pred + 1e-3)

    def test_2b_one_minus_rho_needs_positive_gains(self):
        """K_T < 0 では K_H/(K_H+K_T) ≠ 1−ρ。ρ の定義が符号を捨てているため"""
        KH, KT = 0.7483, -0.0185
        self.assertNotAlmostEqual(KH / (KH + KT), 1 - st.rho(KH, KT), places=3)

    def test_3_phasor_sum_is_exact(self):
        c = DATA['C_phasor']
        self.assertLess(c['max_phase_err_deg'], 1e-9)
        self.assertAlmostEqual(c['magnitude_min'], 1.0, places=9)
        self.assertAlmostEqual(c['magnitude_max'], 1.0, places=9)

    def test_4_sign_flip_moves_equilibrium_to_pi(self):
        KH, KT = 0.0155, -0.689
        self.assertAlmostEqual(st.rho(KH, KT), 0.978, places=3)
        for phi_deg in (0, 20, 40):
            eq = st.equilibria(KH, KT, np.radians(phi_deg))
            stable = [abs(np.degrees(st.wrap(r['e']))) for r in eq if r['stable']]
            self.assertTrue(stable)
            self.assertGreater(min(stable), 170.0, 'e = 0 側に安定点が無いこと')

    def test_4b_positive_sum_is_stable_at_zero(self):
        for phi_deg in (0, 20, 40, 60):
            self.assertLess(st.stable_error(0.060, 0.703, np.radians(phi_deg)), 10.0)

    def test_5_pll_ramp_error_and_cycle_slip(self):
        for r in DATA['E_pll']['rows']:
            if r['rate_over_K'] < 1:
                self.assertAlmostEqual(abs(r['mean_err_deg']), r['pred_deg'], delta=0.1)
                self.assertLess(r['cycle_slips'], 0.5, 'ロック中は周回しない')
            else:
                self.assertGreater(r['cycle_slips'], 5, '|φ̇| > K ではロックが外れる')

    def test_6_world_fixed_wind(self):
        for r in DATA['F_worldwind']['rows']:
            self.assertEqual(r['monotone'], r['beta'] < 1)
            if r['beta'] <= 0.7:
                self.assertEqual(r['reached_fraction'], 1.0)
            if r['beta'] >= 1.3:
                self.assertLess(r['reached_fraction'], 1.0)

    def test_7_reproduces_experiment_4_synthetic(self):
        """同じ種・同じ手順で、実験 4 の合成モデルの値がビット単位で一致すること"""
        bh = json.load(open(os.path.join(ROOT, 'data', 'bearing_vs_homing.json'),
                            encoding='utf-8'))['synthetic']
        key = '横滑りあり（σ=55°）'
        g = DATA['G_task']
        self.assertEqual(g['ideal_travel']['sideslip']['median_deg'],
                         bh['bearing']['by_cond'][key]['median_deg'])
        self.assertEqual(g['ideal_heading']['sideslip']['median_deg'],
                         bh['heading']['by_cond'][key]['median_deg'])


class TestComparator(unittest.TestCase):

    def test_squared_comparator_gives_sin(self):
        self.assertLess(DATA['D_comparator']['max_abs_err_squared'], 1e-12)

    def test_linear_sum_carries_no_difference(self):
        """二乗を外すと、細胞位相で和をとった入力は恒等的に 0"""
        self.assertLess(DATA['D_comparator']['max_abs_linear_side_sum'], 1e-9)

    def test_comparator_live(self):
        for G, T in ((0.3, -0.4), (2.0, 1.0), (-1.2, 2.9)):
            u, _, _ = st.pfl3_comparator(G, T)
            self.assertAlmostEqual(u, np.sin(G - T), places=12)


class TestTaskBudget(unittest.TestCase):

    def test_ordering(self):
        """理想 travel < 追加モデルの制御則 < … < 素の回路の制御則 ≈ 理想 heading"""
        g = DATA['G_task']
        m = {k: g[k]['sideslip']['median_deg'] for k in
             ('ideal_travel', 'augmented', 'plain', 'ideal_heading')}
        self.assertLess(m['ideal_travel'], m['augmented'])
        self.assertLess(m['augmented'], 20.0)
        self.assertGreater(m['plain'], 35.0)
        self.assertLess(abs(m['plain'] - m['ideal_heading']), 3.0)

    def test_static_component_is_small(self):
        """(1−ρ)|φ| の定常成分は、追加モデルで数度しかない"""
        self.assertLess(DATA['G_task']['augmented']['static_component_median_deg'], 5.0)

    def test_gain_sweep_is_monotone(self):
        """理想の travel servo の誤差はループ利得で決まる（利得を上げると下がる）"""
        s = [r['median_deg'] for r in DATA['G_task']['travel_gain_sweep']]
        self.assertTrue(all(a > b for a, b in zip(s, s[1:])))

    def test_budget_plain_prediction_within_ci(self):
        """RSS の仮定で、素の回路の横滑りあり誤差が CI 内に予測できること"""
        self.assertTrue(DATA['G_task']['budget']['plain']['prediction_within_ci'])

    def test_budget_augmented_floor(self):
        b = DATA['G_task']['budget']['augmented']
        self.assertEqual(b['observed_sideslip_deg'], 15.14)
        self.assertGreater(b['implied_floor_deg'], b['static_component_deg'])
        self.assertLess(b['implied_floor_deg'],
                        DATA['G_task']['budget']['plain']['floor_deg'],
                        '追加モデルの床は素の回路の床より小さいはず（そうでないと 15.14° にならない）')

    def test_json_is_fresh(self):
        """data/servo_theory.json がスクリプトの現在の出力と一致すること（古い JSON を引用しない）"""
        g = st.section_g()
        self.assertEqual(g['augmented']['sideslip']['median_deg'],
                         DATA['G_task']['augmented']['sideslip']['median_deg'])
        self.assertEqual(g['plain']['sideslip']['median_deg'],
                         DATA['G_task']['plain']['sideslip']['median_deg'])

    def test_sources_are_labelled(self):
        """GPT-6 の値は「再現していない」と出典に書いてあること"""
        self.assertIn('再現していない', DATA['gains']['augmented']['source'])
        self.assertIn('servo_identify.json', DATA['gains']['plain']['source'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
