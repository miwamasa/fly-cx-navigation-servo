#!/usr/bin/env python3
"""tests/test_pi.py — 経路積分まわりの回帰テスト

実行:  python3 tests/test_pi.py

固定しているのは大きく 3 つ:
  1. numpy 実装 (scripts/cxnet_np.py) が JS 実装 (web/cxnet.js) と一致すること
     — ここがずれると、学習した読み出しをブラウザへ持っていった瞬間に壊れる
  2. 課題の設計が「速度の信号を本当に要求する」ものになっていること
     — 速度をほぼ一定にしていたせいで偽陽性が出た経緯があるので、その退行を防ぐ
  3. docs/PATH_INTEGRATION.md に書いた配線の実測値
"""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from pi_task import (  # noqa: E402
    random_walk, heading_only_baseline, heading_speed_baseline, oracle_baseline,
    home_errors, ridge_fit, ridge_apply, bootstrap_ci, SelfMotionInput,
)

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
CIRCUIT = os.path.join(ROOT, 'data', 'pi_circuit.json')


class TestShuntingDefaultIsInert(unittest.TestCase):
    """分流抑制を足しても、既定（shunt_frac = 0）の力学が変わっていないこと。

    6.10 節で宣言して入れた機構だが、**入れる前の全結果がそのまま有効である**
    ことが前提になる。f = 0 の経路は式の上で従来と恒等なので、
    軌跡が浮動小数の精度で一致しなければならない。ここが崩れたら、
    6.1〜6.9 節の数値はすべて測り直しになる。
    """

    def run_net(self, shunt_frac, steps=300, theta=0.4):
        from cxnet_np import CXNetworkNP
        net = CXNetworkNP(MODEL)
        net.shunt_frac = shunt_frac
        net.reset(1)
        for _ in range(steps):
            net.clear_drive()
            net.set_visual_scene(theta, 1.0)
            net.step(0.02)
        return net.r.copy()

    def test_default_is_zero(self):
        from cxnet_np import CXNetworkNP
        net = CXNetworkNP(MODEL)
        self.assertEqual(net.shunt_frac, 0.0, '既定は分流なし')

    def test_zero_matches_limit(self):
        import numpy as np
        a = self.run_net(0.0)
        b = self.run_net(1e-12)
        self.assertLess(float(np.abs(a - b).max()), 1e-9,
                        'f = 0 の経路が従来の式と一致していない')

    def test_shunting_actually_changes_something(self):
        """陰性対照の裏返し: f を上げれば力学は実際に変わること。"""
        import numpy as np
        a = self.run_net(0.0)
        c = self.run_net(1.0)
        self.assertGreater(float(np.abs(a - c).max()), 1e-6,
                           'f = 1 でも何も変わっていない（実装が効いていない）')


class TestTrajectory(unittest.TestCase):
    def test_stop_go_actually_stops(self):
        """止まる時間がなければ、方位を積むだけで解けてしまう（過去に踏んだ穴）"""
        t = random_walk(3000, rng=np.random.default_rng(0))
        frac_stopped = float(np.mean(t['v'] == 0))
        self.assertGreater(frac_stopped, 0.2, '止まっている時間が短すぎる')
        self.assertLess(frac_stopped, 0.8, '止まっている時間が長すぎる')

    def test_heading_keeps_changing_while_stopped(self):
        """止まっている間も向きが変わらないと、速度の有無が成績に効かない"""
        t = random_walk(3000, rng=np.random.default_rng(1))
        stopped = t['v'] == 0
        dth = np.diff(np.unwrap(t['th']))      # 長さ n-1
        self.assertGreater(np.std(dth[stopped[:-1]]), 1e-4)

    def test_home_vector_is_displacement(self):
        t = random_walk(500, rng=np.random.default_rng(2))
        np.testing.assert_allclose(t['home'], -t['pos'], rtol=1e-6)
        # 出発点では原点にいる
        self.assertLess(np.hypot(*t['home'][0]), 0.2)

    def test_speed_matters_for_the_answer(self):
        """同じ方位系列でも、速度が違えばホームベクトルは変わるはず。

        これが成り立たない課題設計だと、自己運動を使えたかどうかを測れない。
        """
        rng_a = np.random.default_rng(5)
        t1 = random_walk(1500, rng=rng_a)
        # 同じ方位・別の速度プロファイル
        t2 = dict(t1)
        v2 = np.ones_like(t1['v']) * float(np.mean(t1['v']))
        step = (v2 * t1['dt'])[:, None] * np.stack(
            [np.cos(t1['th']), np.sin(t1['th'])], axis=1)
        pos2 = np.cumsum(step, axis=0)
        ang1 = np.arctan2(t1['home'][-1][1], t1['home'][-1][0])
        ang2 = np.arctan2(-pos2[-1][1], -pos2[-1][0])
        diff = abs((ang1 - ang2 + np.pi) % (2 * np.pi) - np.pi)
        self.assertGreater(np.degrees(diff), 5,
                           '速度を一定にしても答えがほぼ同じ＝課題が速度を要求していない')


class TestSideslip(unittest.TestCase):
    """横滑りを入れた課題（docs/PATH_INTEGRATION.md 6節）"""

    def test_travel_direction_differs_from_heading(self):
        t = random_walk(1500, rng=np.random.default_rng(11), sideslip_sigma_deg=55)
        d = np.degrees(np.abs((t['travel'] - t['th'] + np.pi) % (2 * np.pi) - np.pi))
        self.assertGreater(np.median(d), 15, '横滑りが小さすぎて課題にならない')
        np.testing.assert_allclose(t['travel'], t['th'] + t['beta'], atol=1e-9)

    def test_no_sideslip_by_default(self):
        """既定は前進のみ。前の実験（1〜5節）の条件が変わっていないこと。"""
        t = random_walk(500, rng=np.random.default_rng(12))
        np.testing.assert_array_equal(t['beta'], np.zeros(500))
        np.testing.assert_allclose(t['travel'], t['th'])

    def test_sideslip_is_clipped(self):
        t = random_walk(2000, rng=np.random.default_rng(13), sideslip_sigma_deg=90,
                        sideslip_max_deg=60)
        self.assertLessEqual(np.degrees(np.abs(t['beta'])).max(), 60 + 1e-6)

    def test_left_right_flow_encodes_sideslip(self):
        """横滑りは左右の視覚流の比としてしか回路に入らない。

        β>0 なら左、β<0 なら右が強くなる、という向きが崩れていないこと。
        ここが反転すると、ベクトル計算の符号が丸ごと逆になる。
        """
        class FakeNet:
            lno_left = lno_right = spsp_left = spsp_right = ()
        smi = SelfMotionInput(FakeNet(), flow_axis_deg=45.0)
        l0, r0 = smi.drive(1.0, 0.0)
        self.assertAlmostEqual(l0, r0, places=9, msg='横滑り 0 なら左右は等しいはず')
        lp, rp = smi.drive(1.0, np.radians(40))
        self.assertGreater(lp, rp)
        ln, rn = smi.drive(1.0, np.radians(-40))
        self.assertLess(ln, rn)

    def test_baselines_degrade_with_sideslip(self):
        """課題設計の確認: 横滑りが無いと『方位＋速度』で完全に解けてしまう。

        これが成り立たないと、ネットワークが勝てる余地の無い課題を測ることになる。
        """
        def score(sigma, fn):
            F, Y = [], []
            for k in range(8):
                t = random_walk(700, rng=np.random.default_rng(200 + k),
                                sideslip_sigma_deg=sigma, sideslip_tau=4.0)
                F.append(fn(t)[60::4]); Y.append(t['home'][60::4])
            F = np.concatenate(F); Y = np.concatenate(Y)
            W = ridge_fit(F, Y, alpha=1.0)
            return home_errors(ridge_apply(W, F), Y, with_ci=False)['r2']

        self.assertGreater(score(0, heading_speed_baseline), 0.95,
                           '横滑り無しなら方位＋速度でほぼ完全に解けるはず')
        self.assertLess(score(55, heading_speed_baseline), 0.85,
                        '横滑りを入れても解けてしまうと、課題として意味がない')
        self.assertGreater(score(55, oracle_baseline), 0.95,
                           '進行方向を知っていれば解ける＝課題自体は解ける範囲にある')


class TestHeadingOnlyBaseline(unittest.TestCase):
    def test_baseline_is_beatable_but_not_trivial(self):
        """方位だけのベースラインが、それなりに当たるが完璧ではないこと"""
        rng = np.random.default_rng(7)
        F, Y = [], []
        for k in range(12):
            t = random_walk(900, rng=np.random.default_rng(100 + k))
            F.append(heading_only_baseline(t)[60::4])
            Y.append(t['home'][60::4])
        F = np.concatenate(F); Y = np.concatenate(Y)
        W = ridge_fit(F, Y, alpha=1.0)
        e = home_errors(ridge_apply(W, F), Y, with_ci=False)
        self.assertLess(e['angle_median_deg'], 45, 'ベースラインが弱すぎて比較にならない')
        self.assertGreater(e['angle_median_deg'], 5, 'ベースラインが強すぎる＝課題が易しすぎる')


class TestRidge(unittest.TestCase):
    def test_recovers_known_linear_map(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 8))
        Wtrue = rng.standard_normal((8, 2))
        Y = X @ Wtrue + 0.6
        W = ridge_fit(X, Y, alpha=1e-6)
        np.testing.assert_allclose(W[:8], Wtrue, atol=1e-3)
        np.testing.assert_allclose(W[8], [0.6, 0.6], atol=1e-3)

    def test_bootstrap_ci_brackets_median(self):
        v = np.random.default_rng(3).gamma(2, 10, 800)
        lo, hi = bootstrap_ci(v)
        self.assertLess(lo, np.median(v))
        self.assertGreater(hi, np.median(v))


@unittest.skipUnless(os.path.exists(MODEL), 'model/flybrain-cx.gguf が無い')
class TestNumpyMatchesJS(unittest.TestCase):
    """numpy 実装と JS 実装が一致すること。

    期待値は web/cxnet.js を node で 300 ステップ回して得た値（下の手順で再生成できる）:
        net.reset(3); 毎ステップ setVisualScene(0.7,1.0) + injectGoal(2.0,0.6) + step(0.02)
    JS は r を Float32Array で持つので、一致は float32 の精度まで。
    """
    JS_AT_300 = {
        'theta': 0.6516016579002225,
        'strength': 0.8682052636897293,
        'turn': 0.5538622757050686,
        'mean_r': 0.021889851146911884,
    }

    def test_matches_js_reference(self):
        from cxnet_np import CXNetworkNP
        net = CXNetworkNP(MODEL)
        net.reset(3)
        for _ in range(300):
            net.clear_drive()
            net.set_visual_scene(0.7, 1.0)
            net.inject_goal(2.0, 0.6)
            net.step(0.02)
        h = net.read_heading(); s = net.read_steering()
        got = {'theta': h['theta'], 'strength': h['strength'],
               'turn': s['turn'], 'mean_r': float(net.r.mean())}
        for k, want in self.JS_AT_300.items():
            self.assertAlmostEqual(
                got[k], want, delta=1e-5,
                msg=f'{k}: numpy {got[k]!r} と JS {want!r} が食い違う')

    def test_input_layers_are_the_same_as_js(self):
        """クランプする集団（入力層）が JS と同じであること"""
        from cxnet_np import CXNetworkNP, ROLE
        net = CXNetworkNP(MODEL)
        expected = np.zeros(net.N, dtype=bool)
        for rl in (ROLE['ER'], ROLE['ExR'], ROLE['SPSP'], ROLE['LNO']):
            expected |= (net.role == rl)
        np.testing.assert_array_equal(net.clamped, expected)
        self.assertEqual(int(net.clamped.sum()), 351)


@unittest.skipUnless(os.path.exists(CIRCUIT), 'data/pi_circuit.json が無い')
class TestMeasuredCircuit(unittest.TestCase):
    """docs/PATH_INTEGRATION.md に書いた配線の実測値を固定する。"""

    @classmethod
    def setUpClass(cls):
        cls.C = json.load(open(CIRCUIT, encoding='utf-8'))

    def test_self_motion_goes_to_pfn(self):
        for carrier, lo in [('LNO', 80), ('SpsP', 75)]:
            share = self.C['self_motion_carriers'][carrier]['targets']['PFN']['share_pct']
            self.assertGreater(share, lo, f'{carrier} → PFN の割合が下がっている')

    def test_lno_is_contralateral_and_spsp_ipsilateral(self):
        """LNO は対側、SpsP は同側。ベクトル計算の前提になる左右分離。"""
        lno = self.C['lateralisation']['LNO2']
        self.assertAlmostEqual(lno['L->PFNd']['pb_left'], 0, delta=1)
        self.assertLess(lno['L->PFNd']['pb_right'], -1000)
        self.assertLess(lno['R->PFNd']['pb_left'], -1000)
        sps = self.C['lateralisation']['SpsP']
        self.assertLess(sps['L->PFNd']['pb_left'], -100)
        self.assertAlmostEqual(sps['L->PFNd']['pb_right'], 0, delta=1)

    def test_pb_to_fb_is_mirror_symmetric(self):
        """左: 列 = 糸球体 − 1 / 右: 列 = 10 − 糸球体"""
        m = self.C['pb_to_fb_columns']['PFNd']
        for k, cols in m.items():
            side, glom = k[0], int(k[1:])
            want = glom - 1 if side == 'L' else 10 - glom
            self.assertIn(want, cols, f'{k} の投射先が鏡像の規則から外れた')

    def test_left_right_write_offset_is_a_vector_basis(self):
        """左右 PFN の書き込み位相が 45〜90° ずれていること（ベクトル計算の基底）"""
        for k in ['PFNd->vDelta', 'PFNv->vDelta', 'PFNa->vDelta']:
            off = self.C['pfn_lr_offset'][k]['offset_deg']
            self.assertGreater(off, 45, f'{k} のずれが小さすぎる')
            self.assertLess(off, 95, f'{k} のずれが大きすぎる')
            # 左右がほぼ 0 をはさんで対称であること
            l = self.C['pfn_lr_offset'][k]['left_peak_deg']
            r = self.C['pfn_lr_offset'][k]['right_peak_deg']
            self.assertGreater(l, 0); self.assertLess(r, 0)

    def test_hdelta_recurrence_is_all_excitatory(self):
        h = self.C['hdelta']
        self.assertEqual(h['recurrent_negative'], 0.0)
        self.assertGreater(h['input_share_pct']['hDelta'], 20)
        self.assertEqual(h['self_connections'], 0.0)

    def test_hdelta_drives_pfl3_more_than_fc(self):
        """帰巣の出口。hΔ→PFL3 が FC→PFL3 より太いこと。"""
        o = self.C['output']
        self.assertGreater(o['hDelta->PFL3']['total'], o['FC->PFL3']['total'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
