#!/usr/bin/env python3
"""Unit tests for the added circuit of paper B section 7 (scripts/servo_augmented.py).

  python3 tests/test_servo_augmented.py

What is pinned
--------------
1. With every mechanism off, the network's dynamics are bit-identical to the base model.
   (Otherwise "plain" and "augmented" would not be comparable.)
2. The alignment weights M are non-negative and live only on PFN -> hDeltaB pairs that
   exist in the connectome.
3. Fed ideal decoded inputs, the squaring comparator gives a steering signal with the
   sign of sin(G - T); swapping delta reverses it; without the square it vanishes.
4. The delay line delays what the circuit sees by exactly the requested number of steps.
5. The multiplicative gate rotates the PFN phasor sum by phi; the additive control does not.
6. The saved results respect the pre-registered bookkeeping (if data/servo_augmented.json exists).
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

import servo_augmented as sa                     # noqa: E402

DATA = os.path.join(ROOT, 'data', 'servo_augmented.json')


def run_steps(net, aug, bvi, n=60, theta=0.7, phi=0.4, speed=1.0, goal=1.2):
    traj = []
    net.reset(1)
    for _ in range(n):
        net.clear_drive()
        net.set_visual_scene(theta, 1.0)
        net.inject_goal(goal, 0.3)
        bvi.apply(net.clamp_val, speed, phi)
        aug.apply(speed, phi)
        net.step(sa.DT)
        traj.append(net.r.copy())
    return np.array(traj)


class TestInertWhenOff(unittest.TestCase):
    def test_bit_identical(self):
        net0 = sa.make_net(); bvi0 = sa.BodyVelocityInput(net0)

        class Nothing:
            def apply(self, *_):
                pass
        ref = run_steps(net0, Nothing(), bvi0)
        for variant in ('N', 'C'):
            net = sa.make_net(); bvi = sa.BodyVelocityInput(net)
            aug = sa.Augmentation(net, variant, mechs=(), m=0.3, nu_P=0.25, nu_B=0.5, lam=0.2)
            got = run_steps(net, aug, bvi)
            self.assertTrue(np.array_equal(ref, got), f'variant {variant} with mechs=() changed the dynamics')

    def test_scalings_only_apply_with_their_mechanism(self):
        net = sa.make_net()
        g0 = net.row_gain.copy()
        sa.Augmentation(net, 'N', mechs=(3,), nu_P=0.25, nu_B=0.5)
        self.assertTrue(np.array_equal(g0, net.row_gain))


class TestAlignmentWeights(unittest.TestCase):
    def test_nonnegative_and_on_existing_pairs(self):
        net = sa.make_net(); bvi = sa.BodyVelocityInput(net)
        aug = sa.Augmentation(net, 'N', mechs=(1, 2), m=0.1, nu_P=0.25, nu_B=0.5)
        _, recs = sa.front_sweep(aug, bvi, [0, 180], [0, 90], speeds=(0.5, 1.0), steps=120, record=True)
        M = sa.fit_M(aug, recs, amp=0.2)
        self.assertTrue((M >= 0).all())
        self.assertFalse((M[~aug.allowed] != 0).any(), 'weight on a pair absent from the connectome')
        W = net.W.tocsr()[aug.hdb][:, aug.pfn].toarray()
        self.assertTrue(((M > 0) <= (W != 0)).all())

    def test_shuffle_keeps_values_and_pairs_exist(self):
        net = sa.make_net()
        aug = sa.Augmentation(net, 'N', mechs=())
        rng = np.random.default_rng(0)
        M = np.zeros_like(aug.allowed, dtype=float)
        idx = np.argwhere(aug.allowed)[:20]
        M[idx[:, 0], idx[:, 1]] = np.linspace(0.1, 2.0, 20)
        S = aug._shuffle_pairs(M, rng)
        self.assertEqual(sorted(S[S > 0]), sorted(M[M > 0]))
        self.assertTrue(aug.allowed[S > 0].all())

    def test_random_pairs_are_degree_matched(self):
        net = sa.make_net()
        aug = sa.Augmentation(net, 'N', mechs=())
        R = sa.random_pairs_like(aug, np.random.default_rng(0))
        self.assertTrue(np.array_equal(R.sum(1), aug.allowed.sum(1)))


class TestComparator(unittest.TestCase):
    """The squaring term on its own: (R - L)/(R + L) of the added input vs sin(G - T)."""

    def added_turn(self, aug, G, T):
        g = np.cos(G - aug.gamma)
        t = np.cos(T - aug.gamma - aug.delta)
        x = g + t
        x = x * x if aug.mech3 != 'linear' else x
        L, R = x[aug.side < 0].sum(), x[aug.side > 0].sum()
        return (R - L) / (R + L) if aug.mech3 != 'linear' else R - L

    def test_sign_follows_sin_g_minus_t(self):
        net = sa.make_net()
        aug = sa.Augmentation(net, 'N', mechs=())
        for G in np.radians([0, 60, 150, 250]):
            for d in np.radians([-120, -60, -20, 20, 60, 120]):
                u = self.added_turn(aug, G, G - d)
                self.assertEqual(np.sign(u), np.sign(np.sin(d)), f'G={G:.2f}, G-T={d:.2f}')

    def test_swapped_delta_reverses(self):
        net = sa.make_net()
        a = sa.Augmentation(net, 'N', mechs=())
        b = sa.Augmentation(net, 'N', mechs=(), mech3='flipped')
        for d in np.radians([-90, -30, 30, 90]):
            self.assertEqual(np.sign(self.added_turn(a, 0.5, 0.5 - d)),
                             -np.sign(self.added_turn(b, 0.5, 0.5 - d)))

    def test_twelve_even_cells_are_exact(self):
        """With evenly spaced phases the identity is exact (paper B section 3.6)."""
        gam = np.radians(np.arange(12) * 30.0)
        for d in np.radians([-150, -45, 10, 80]):
            G, T = 0.3, 0.3 - d
            L = ((np.cos(G - gam) + np.cos(T - gam - np.pi / 2)) ** 2).sum()
            R = ((np.cos(G - gam) + np.cos(T - gam + np.pi / 2)) ** 2).sum()
            self.assertAlmostEqual((R - L) / (R + L), np.sin(d), places=12)


class TestMultiplication(unittest.TestCase):
    def test_multiplicative_gate_rotates_additive_does_not(self):
        """Phasor sum of group-gated heading bumps written with offsets p_g."""
        p = np.radians([45, -45, -135, 135])
        cells = np.radians(np.arange(16) * 22.5)
        H = 0.8
        for phi in np.radians([-100, -30, 20, 70, 160]):
            A = np.maximum(0, np.cos(phi - p))
            zm = 0j
            for g in range(4):
                gate = 1 + np.cos(H - cells)
                zm += (A[g] * gate * np.exp(1j * (cells + p[g]))).sum()
            err_m = abs(np.angle(zm * np.exp(-1j * (H + phi))))
            self.assertLess(np.degrees(err_m), 1e-6)
        # the additive version: its angle does not depend on phi at all
        z1 = sum((0.5 * (np.maximum(0, np.cos(0.0 - p[g])) + 1 + np.cos(H - cells))
                  * np.exp(1j * (cells + p[g]))).sum() for g in range(4))
        z2 = sum((0.5 * (np.maximum(0, np.cos(2.0 - p[g])) + 1 + np.cos(H - cells))
                  * np.exp(1j * (cells + p[g]))).sum() for g in range(4))
        # velocity enters only as a per-group constant, which cancels over the cells:
        # the additive sum is the same complex number whatever phi is
        self.assertLess(abs(z1 - z2), 1e-9)


class TestDelayLine(unittest.TestCase):
    def test_delay_is_exact(self):
        net = sa.make_net(); bvi = sa.BodyVelocityInput(net)
        seen = []
        track = np.arange(30) * 0.01
        sa.closed_loop(_SpyAug(net, seen), bvi, 0.0, track, settle_steps=0, delay_steps=3)
        # step i sees the input of step max(0, i - 3)
        self.assertEqual(seen[:4], [track[0]] * 4)
        self.assertTrue(np.allclose(seen[3:], track[:-3]))


class _SpyAug:
    def __init__(self, net, seen):
        self.net, self.seen = net, seen

    def apply(self, speed, phi):
        self.seen.append(phi)


class TestSavedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DATA):
            raise unittest.SkipTest('run scripts/servo_augmented.py first')
        with open(DATA, encoding='utf-8') as f:
            cls.d = json.load(f)
        if cls.d.get('quick'):
            raise unittest.SkipTest('saved results are from --quick')

    def test_plain_matches_existing_measurement(self):
        """Same code path as scripts/bearing_vs_homing.py and servo_identify.py."""
        with open(os.path.join(ROOT, 'data', 'bearing_vs_homing.json'), encoding='utf-8') as f:
            bh = list(json.load(f)['network'].values())
        self.assertAlmostEqual(self.d['plain']['sideslip']['median_deg'], bh[0]['median_deg'], places=6)
        self.assertAlmostEqual(self.d['plain']['no_sideslip']['median_deg'], bh[1]['median_deg'], places=6)
        with open(os.path.join(ROOT, 'data', 'servo_identify.json'), encoding='utf-8') as f:
            si = next(iter(json.load(f)['network'].values()))['fit']
        self.assertAlmostEqual(self.d['plain']['back']['K_H'], si['K_H'], places=6)
        self.assertAlmostEqual(self.d['plain']['back']['K_T'], si['K_T'], places=6)

    def test_lambda_rule_was_followed(self):
        for v, rec in self.d['variants'].items():
            cands = rec['config']['lambda_candidates']
            met = [c for c in cands if c['chosen_rule_met']]
            want = met[0]['lam'] if met else max(cands, key=lambda c: c['rho'])['lam']
            self.assertEqual(rec['config']['lam'], want, v)

    def test_verdicts_are_consistent_with_numbers(self):
        c = self.d['prereg']
        for v, rec in self.d['variants'].items():
            e = rec['evaluation']
            fit = e['back']['fit']
            want = fit['r2'] >= c['back']['r2_min'] and fit['rho'] > c['back']['rho_min'] and fit['K_T'] > 0
            self.assertEqual(e['back']['verdict']['pass'], want, v)
            cl = e['closed_loop']
            want = (cl['sideslip']['median_deg'] <= c['closed_loop']['sideslip_max_deg']
                    and cl['no_sideslip']['median_deg'] <= c['closed_loop']['no_sideslip_max_deg'])
            self.assertEqual(cl['pass'], want, v)

    def test_swapped_delta_gives_negative_k_t(self):
        """Theory (section 3.4): the sign of K_T flips and the loop runs away from the goal."""
        for v, rec in self.d['variants'].items():
            row = rec['ablations']['(3) delta swapped']
            self.assertLess(row['back']['K_T'], 0, v)
            self.assertGreater(row['sideslip']['median_deg'], 90, v)


if __name__ == '__main__':
    unittest.main(verbosity=1)
