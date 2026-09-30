#!/usr/bin/env python3
"""Consistency tests for the English papers (paper/) and Medium articles (medium/).

Why these exist: text goes stale before data does. This project once quoted
numbers in its report that no data file supported any more (see tests/test_docs.py).
These tests pin every key number in the English deliverables to the file it
comes from, check that every figure link resolves, and enforce the scope
decisions (English only; paper B uses no numbers from the unreproduced
augmented model; Medium articles use no LaTeX or tables).

  python3 tests/test_papers_en.py

Needs only the standard library, except TestNumberSheetIsFresh, which needs
numpy/matplotlib (it re-imports paper/scripts/fig_b.py) and skips without them.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, 'paper')
MEDIUM = os.path.join(ROOT, 'medium')

PAPER_A = os.path.join(PAPER, 'a_navigation', 'PAPER.md')
PAPER_B = os.path.join(PAPER, 'b_servo_control', 'PAPER.md')
MED_A = os.path.join(MEDIUM, 'a_fly_brain_navigation.md')
MED_B = os.path.join(MEDIUM, 'b_fly_brain_servo_control.md')


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def data(name):
    return json.loads(read(os.path.join(ROOT, 'data', name)))


def sheet(name):
    return json.loads(read(os.path.join(PAPER, 'data', name)))


# The Medium articles are optional: the stand-alone reproduction package (toGitHub/)
# carries the papers only, and every Medium check is skipped when they are absent.
HAS_MEDIUM = os.path.exists(MED_A) and os.path.exists(MED_B)
TEXT = {p: read(p) for p in (PAPER_A, PAPER_B, MED_A, MED_B) if os.path.exists(p)}


def forms(s):
    """The papers use U+2212 for minus; accept either spelling."""
    return {s, s.replace('-', '−'), s.replace('−', '-')}


class NumberCase(unittest.TestCase):
    def assert_quoted(self, path, needle, why):
        if path not in TEXT:          # an optional Medium article that is not present
            return
        text = TEXT[path]
        self.assertTrue(any(f in text for f in forms(needle)),
                        f'{os.path.relpath(path, ROOT)}: {why} -- "{needle}" not found')


# ---------------------------------------------------------------- paper A
class TestPaperANumbers(NumberCase):
    """Every key number in paper A matches data/*.json or paper/data/paper_a_measurements.json."""

    @classmethod
    def setUpClass(cls):
        cls.m = sheet('paper_a_measurements.json')
        cls.audit = data('claims_audit_maze.json')['claims']
        cls.rep = data('cx_extract_report.json')

    def test_extraction_counts(self):
        r = self.rep
        self.assert_quoted(PAPER_A, f'{r["neurons"]:,} neurons in {r["cell_types"]} cell types',
                           'neurons and cell types')
        self.assert_quoted(PAPER_A, f'{r["edges_signed"]:,} signed', 'signed edges')
        self.assert_quoted(PAPER_A, f'{int(r["synapses_total"]):,} synapses', 'total synapses')
        self.assert_quoted(PAPER_A, f'{r["excitatory_edges"]:,} excitatory', 'excitatory edges')
        self.assert_quoted(PAPER_A, f'{r["inhibitory_edges"]:,} inhibitory', 'inhibitory edges')
        ph = r['phase']
        self.assert_quoted(PAPER_A, f'{ph["with_phase"]:,} of the 2,308', 'cells with a phase')
        self.assert_quoted(PAPER_A, f'{ph["after_labels"]:,} from labels', 'phases from labels')
        self.assert_quoted(PAPER_A, f'{ph["propagated_round_1"]} and {ph["propagated_round_2"]}',
                           'propagation rounds')

    def test_file_size(self):
        size = os.path.getsize(os.path.join(ROOT, 'model', 'flybrain-cx.gguf'))
        self.assertEqual(size, self.m['gguf']['file_bytes'])
        self.assert_quoted(PAPER_A, f'{size:,}-byte', 'model file size')

    def test_ring(self):
        ring = self.m['ring']
        self.assert_quoted(PAPER_A, ' '.join(ring['glomerulus_order']), 'recovered glomerulus order')
        self.assert_quoted(PAPER_A, f'{ring["max_dev_from_stored_deg"]:.1f}°', 'embedding reproduces phase')
        ev = ring['eigenvalues']
        self.assert_quoted(PAPER_A, f'{ev[1]:.3f} and {ev[2]:.3f}', 'eigenvalues of the cos/sin pair')
        self.assertEqual(ring['n_epg'], 50)

    def test_comparator_offsets(self):
        off = self.m['pfl3_offsets_deg']
        self.assert_quoted(PAPER_A, f'{off["left"]:+.1f}°', 'left FC2->PFL3 offset')
        self.assert_quoted(PAPER_A, f'{off["right"]:+.1f}°', 'right FC2->PFL3 offset')
        self.assert_quoted(PAPER_A, f'{off["balance_point_deg"]:.1f}°', 'structural balance point')
        # the offsets must be a mirror pair of about 73 degrees
        self.assertAlmostEqual(off['left'], -73, delta=1.0)
        self.assertAlmostEqual(off['right'], 73, delta=1.0)

    def test_running_balance_point(self):
        bp = self.audit['pfl3_balance_point_deg']['median']
        self.assert_quoted(PAPER_A, f'{bp:.1f}°', 'running balance point')

    def test_closed_loop_by_bits(self):
        by = self.audit['bit_depth_heading_error_deg']
        for bits in ('8', '4', '3', '2'):
            a = by[bits]['abs']
            s = f'{a["median"]:.1f}° [{a["ci95"][0]:.1f}, {a["ci95"][1]:.1f}]'
            self.assert_quoted(PAPER_A, s, f'closed-loop error at {bits} bit')
        self.assert_quoted(PAPER_A, f'{by["8"]["signed"]["median"]:.1f}°', 'systematic bias')

    def test_python_replicates_javascript(self):
        """The numpy engine reproduces the JavaScript audit (paper A says so)."""
        by = self.audit['bit_depth_heading_error_deg']
        py = self.m['closed_loop_python']
        for bits in ('8', '4', '3', '2'):
            self.assertAlmostEqual(py[bits]['median_abs_deg'], by[bits]['abs']['median'], delta=0.5,
                                   msg=f'{bits} bit: Python {py[bits]["median_abs_deg"]:.2f} '
                                       f'vs JS {by[bits]["abs"]["median"]:.2f}')

    def test_pruning(self):
        syn = self.m['synapses_by_bits']
        for b in ('8', '4', '3', '2'):
            self.assert_quoted(PAPER_A, f'{syn[b]:,}', f'non-zero synapses at {b} bit')
        frac = 100 * self.m['pruned_fraction_by_bits']['4']
        self.assert_quoted(PAPER_A, f'{frac:.1f}%', 'pruned fraction at 4 bit')
        self.assert_quoted(MED_A, f'{frac:.0f}%', 'pruned fraction at 4 bit (Medium)')

    def test_minimal_circuit_table(self):
        tab = {r['config']: r for r in self.m['minimal_circuit_table']}
        mini = tab['EPG + FC2 + PFL3']
        audit = self.audit['minimal_circuit']
        self.assertEqual((mini['neurons'], mini['synapses']), (audit['neurons'], audit['edges']),
                         'Python and JavaScript agree on the minimal circuit')
        self.assertTrue(mini['pass'])
        self.assertEqual(mini['steering_sign_correct'], mini['trials'])
        for name, r in tab.items():
            kbit = f'{r["kbit"]:,.0f}' if r['kbit'] >= 100 else f'{r["kbit"]:.1f}'
            row = (f'{r["neurons"]:,} | {r["synapses"]:,} | {r["bits"]} | {kbit} | '
                   f'{r["closed_loop_median_abs_deg"]:.1f}° | {r["steering_sign_correct"]}/{r["trials"]}')
            plain = TEXT[PAPER_A].replace('**', '')      # the chosen row is bold
            self.assertTrue(row in plain, f'minimal-circuit table row "{name}": "{row}" not found')
        for path in (PAPER_A, MED_A):
            self.assert_quoted(path, '166 neurons and 932 synapses', 'minimal circuit size')

    def test_ablations(self):
        abl = {k: v['abs'] for k, v in self.audit['ablation_heading_error_deg'].items()}
        vals = sorted(v['median'] for v in abl.values())
        # intact 14.7, ER 15.3, Δ7+ER 105.8, Δ7 106.7
        self.assert_quoted(PAPER_A, f'{vals[-1]:.1f}° [', 'Δ7 ablation')
        self.assert_quoted(PAPER_A, f'{vals[-2]:.1f}° [', 'Δ7 + ER ablation')
        self.assert_quoted(PAPER_A, f'{vals[1]:.1f}°', 'ER ablation')


# ---------------------------------------------------------------- paper B
class TestPaperBNumbers(NumberCase):
    @classmethod
    def setUpClass(cls):
        cls.m = sheet('paper_b_measurements.json')

    def test_control_law_fit(self):
        f = self.m['back_stage_fit']
        s = (f'K_H = {f["K_H"]:+.3f}, K_T = {f["K_T"]:+.3f}, c = {f["c"]:+.3f}, '
             f'R² = {f["r2"]:.3f}, ρ = {f["rho"]:.3f}')
        self.assert_quoted(PAPER_B, s, 'fitted control law')
        self.assert_quoted(MED_B, f'K_H = {f["K_H"]:+.3f}, K_T = {f["K_T"]:+.3f}, ρ = {f["rho"]:.3f}',
                           'fitted control law (Medium)')
        signed = f['K_T'] / (f['K_H'] + f['K_T'])
        self.assert_quoted(PAPER_B, f'K_T/(K_H + K_T) = {signed:.3f}',
                           'signed rho')
        self.assert_quoted(PAPER_B, f'K_H + K_T = {f["K_H"] + f["K_T"]:.3f}', 'stability margin')

    def test_ideal_clamp(self):
        r = self.m['ideal_hdb_clamp_rho']
        self.assert_quoted(PAPER_B, f'ρ = {r["travel"]:.3f}, {r["heading"]:.3f} and {r["shuffled"]:.3f}',
                           'ideal hΔB clamp controls')

    def test_task_errors(self):
        t = self.m['task_error_deg']
        ci = t['circuit_sideslip_ci95']
        for path in (PAPER_B, MED_B):
            self.assert_quoted(path, f'{t["circuit_sideslip"]:.1f}° [{ci[0]:.1f}, {ci[1]:.1f}]',
                               'circuit error with sideslip')
            self.assert_quoted(path, f'{t["rss_prediction"]:.1f}°', 'RSS prediction')
            self.assert_quoted(path, f'{t["ideal_heading_sideslip"]:.1f}°', 'ideal heading servo')
            self.assert_quoted(path, f'{t["ideal_travel_sideslip"]:.1f}°', 'ideal travel servo')
            self.assert_quoted(path, f'{t["plain_law_sideslip"]:.1f}°', 'fitted law alone')
            self.assert_quoted(path, f'{t["floor"]:.1f}°', 'floor')
        self.assertTrue(t['prediction_within_ci'])
        nci = t['circuit_no_sideslip_ci95']
        self.assert_quoted(PAPER_B, f'{t["circuit_no_sideslip"]:.1f}° [{nci[0]:.1f}, {nci[1]:.1f}]',
                           'circuit error without sideslip')
        self.assert_quoted(PAPER_B, f'√({t["circuit_no_sideslip"]:.1f}² − {t["plain_law_no_sideslip"]:.1f}²)',
                           'floor derivation')
        self.assert_quoted(PAPER_B, f'median |φ| over the task is {t["median_abs_phi"]:.1f}°', 'median sideslip')
        sw = {r['K_loop']: r['median_deg'] for r in self.m['travel_gain_sweep']}
        self.assert_quoted(PAPER_B, f'from {sw[2.6]:.1f}° to {sw[20.0]:.1f}°', 'lag vs loop gain')

    def test_wiring(self):
        pfl3 = self.m['hdb_to_pfl3']
        share = 100 * self.m['pfl3_routes']['T']
        for path in (PAPER_B, MED_B):
            self.assert_quoted(path, f'{share:.2f}%', 'hΔB share of PFL3 input')
            self.assert_quoted(path, f'{pfl3["n_synapses"]} synapses', 'hΔB -> PFL3 synapses')
        self.assert_quoted(PAPER_B, f'{pfl3["n_synapses"]} synapses in {pfl3["n_connections"]} connections',
                           'hΔB -> PFL3 connections')
        routes = self.m['pfl3_routes']
        for k in ('hD_other', 'G', 'H', 'PFN'):
            self.assert_quoted(PAPER_B, f'{100 * routes[k]:.1f}%', f'PFL3 route {k}')

    def test_front_stage(self):
        w = self.m['pfn_write_offset_deg']
        s = ', '.join(f'{w[g]:+.0f}°' for g in ('PFNd-L', 'PFNd-R', 'PFNv-L'))
        s += f' and {w["PFNv-R"]:+.0f}°'
        for path in (PAPER_B, MED_B):
            self.assert_quoted(path, s, 'four write offsets')
        conc = self.m['pfn_concentration']
        for g in ('PFNd-L', 'PFNd-R', 'PFNv-L', 'PFNv-R'):
            self.assert_quoted(PAPER_B, f'| {conc[g]:.2f} |', f'concentration {g}')
        self.assertEqual(self.m['n_hdb_all_four'], self.m['n_hdb'])
        self.assert_quoted(PAPER_B, f'{self.m["n_hdb"]} of {self.m["n_hdb"]}', 'hΔB convergence')
        self.assert_quoted(PAPER_B, f'{abs(self.m["pfn_pairwise_deg"]["PFNd-L−PFNd-R"]):.0f}° apart',
                           'PFNd bases apart')

    def test_transform_fails(self):
        n = self.m['coord_transform_n_conditions']
        self.assertEqual(self.m['coord_transform_n_pass'], 0)
        self.assert_quoted(PAPER_B, f'0 of {n}', 'no condition passes')
        self.assert_quoted(PAPER_B, f'{self.m["coord_transform_max_travel_gain"]:+.2f}', 'max travel gain')

    def test_shunting(self):
        rows = {(r['f'], r['k'], r['sign']): r for r in self.m['shunting_rows']}
        base, f1 = rows[(0.0, 1.0, 'measured')], rows[(1.0, 1.0, 'measured')]
        for path in (PAPER_B,):
            self.assert_quoted(path, f'from {base["maze_deg"]:.1f}° to {f1["maze_deg"]:.1f}°',
                               'shunting maze cost')
        self.assert_quoted(PAPER_B, f'{100 * f1["PFNv_alive_moving"]:.0f}% active while walking',
                           'PFNv wakes')
        p = self.m['pfn_collapsed_pref_deg']
        s = ', '.join(f'{p[g]:+.0f}°' for g in ('PFNd-L', 'PFNd-R', 'PFNv-L')) + f', {p["PFNv-R"]:+.0f}°'
        self.assert_quoted(PAPER_B, s, 'collapsed preferences')
        loc = self.m['localize_f1']
        self.assert_quoted(PAPER_B, f'| {loc["basis"]["travel_gain"]:+.2f}** |'.replace('| ', '| **'),
                           'basis travel gain')
        sh = self.m['hdb_input_share_f1']
        self.assert_quoted(PAPER_B, f'{100 * (sh["PFNd"] + sh["PFNv"]):.0f}% of hΔB', 'PFN share of hΔB input')

    def test_uptake(self):
        k = list(self.m['sideslip_uptake_k'].values())
        self.assert_quoted(PAPER_B, f'k = {k[0]["k"]:+.3f} ± {k[0]["ci95"]:.3f}', 'baseline uptake')
        for c in k[1:]:
            self.assert_quoted(PAPER_B, f'{c["k"]:+.3f} ± {c["ci95"]:.3f}'.lstrip('+'), 'control uptake')

    def test_confound(self):
        c = self.m['goal_injection_confound_deg']
        self.assert_quoted(PAPER_B, f'{c[str(["FC2"])]:.1f}°', 'FC2-only injection')
        self.assert_quoted(PAPER_B, f'{c[str(["hDeltaB"])]:.1f}° (chance)', 'hΔB-only injection')

    def test_pll(self):
        rows = {r['rate_over_K']: r for r in self.m['pll']}
        self.assertAlmostEqual(abs(rows[0.5]['mean_err_deg']), 30.0, places=3)
        for r in self.m['pll']:
            if r['pred_deg'] is not None:
                self.assertLess(abs(abs(r['mean_err_deg']) - r['pred_deg']), 0.1)
        self.assert_quoted(PAPER_B, f'slips {rows[1.2]["cycle_slips"]:.0f} cycles', 'cycle slips at 1.2')
        self.assert_quoted(PAPER_B, f'slips {rows[2.0]["cycle_slips"]:.0f}', 'cycle slips at 2.0')
        wind = {r['beta']: r['reached_fraction'] for r in self.m['world_wind']}
        self.assert_quoted(PAPER_B, f'{100 * wind[1.3]:.0f}% at β = 1.3', 'world wind 1.3')
        self.assert_quoted(PAPER_B, f'{100 * wind[2.0]:.0f}% at β = 2.0', 'world wind 2.0')


class TestPaperBSection7(NumberCase):
    """Section 7 (the added circuit) against data/servo_augmented.json via the number sheet."""

    @classmethod
    def setUpClass(cls):
        cls.a = sheet('paper_b_measurements.json').get('augmented')
        if cls.a is None:
            raise unittest.SkipTest('run scripts/servo_augmented.py and paper/scripts/fig_b.py')

    def test_not_quick(self):
        self.assertFalse(self.a['quick'], 'section 7 must be written from the full run')

    def test_config_table(self):
        for v, r in self.a['variants'].items():
            c = r['config']
            row = (f"| {v} | {c['m']:g} | {c['nu_P']:g} | {c['lam']:g} | {r['M_nonzero']} | "
                   f"{r['M_pfn_cells']} / {r['M_hdb_cells']} |")
            self.assert_quoted(PAPER_B, row, f'configuration row {v}')

    def test_front_stage(self):
        n, c = self.a['variants']['N'], self.a['variants']['C']
        for v, r in (('N', n), ('C', c)):
            h = r['front_held_out']
            self.assert_quoted(PAPER_B, f"b = {h['travel_gain']:.2f}", f'held-out b {v}')
            self.assert_quoted(PAPER_B, f"{h['residual_R']:.3f}", f'held-out R {v}')
            self.assert_quoted(PAPER_B, f"{h['amp_r2']:.3f}", f'held-out amplitude {v}')
            self.assertFalse(h['pass'], 'the text says the amplitude criterion fails')
            self.assertLess(h['amp_r2'], 0.9)

    def test_back_stage(self):
        for v, r in self.a['variants'].items():
            b = r['back']
            s = f"K_H = {b['K_H']:+.3f}, K_T = {b['K_T']:+.3f}, ρ = {b['rho']:.3f} (R² = {b['r2']:.3f})"
            self.assert_quoted(PAPER_B, s, f'back-stage fit {v}')
            self.assertTrue(b['pass'])
            self.assert_quoted(MED_B, f"{b['rho']:.3f}", f'rho {v} (Medium)')

    def test_closed_loop_table(self):
        for v, r in self.a['variants'].items():
            ss, no = r['sideslip'], r['no_sideslip']
            self.assert_quoted(PAPER_B, f"{ss['median_deg']:.1f}° [{ss['ci95'][0]:.1f}, {ss['ci95'][1]:.1f}]",
                               f'closed loop {v}')
            self.assert_quoted(PAPER_B, f"{no['median_deg']:.1f}° [{no['ci95'][0]:.1f}, {no['ci95'][1]:.1f}]",
                               f'no sideslip {v}')
            self.assert_quoted(PAPER_B, f"{ss['p90_deg']:.1f}°", f'90th percentile {v}')
            self.assert_quoted(MED_B, f"{ss['median_deg']:.1f}°", f'closed loop {v} (Medium)')
            self.assertTrue(r['closed_loop_pass'])
        n, c = self.a['variants']['N'], self.a['variants']['C']
        self.assert_quoted(PAPER_B, f"{n['trials_improved']} of {n['n_trials']} trials for N and "
                                    f"in {c['trials_improved']} of {c['n_trials']} for C", 'trials improved')
        p = self.a['plain']['sideslip']
        self.assert_quoted(PAPER_B, f"{p['p90_deg']:.1f}°", 'plain 90th percentile')

    def test_budget(self):
        n, c = self.a['variants']['N']['budget'], self.a['variants']['C']['budget']
        self.assert_quoted(PAPER_B, f"{n['law_sideslip_deg']:.1f}° (N) and {c['law_sideslip_deg']:.1f}° (C)", 'law alone')
        self.assert_quoted(PAPER_B, f"{n['effective_loop_gain']:.2f} and {c['effective_loop_gain']:.2f} rad/s",
                           'effective gains')
        self.assert_quoted(PAPER_B, f"predicts {n['rss_prediction_deg']:.1f}° for N", 'RSS N')
        self.assert_quoted(PAPER_B, f"{c['rss_prediction_deg']:.1f}° for C", 'RSS C')

    def test_ablation_table(self):
        names = {'full added circuit': None,
                 'without (1) multiplication': 'without (1) multiplication',
                 'without (2) alignment': 'without (2) alignment',
                 'without (3) comparator': 'without (3) comparator',
                 '(3) linear, no square': '(3) linear, no square',
                 '(3) δ_L, δ_R swapped': '(3) delta swapped',
                 '(1) additive instead of multiplicative': '(1) additive, not multiplicative',
                 '(2) fitted weights moved to shuffled existing pairs': '(2) weights on shuffled pairs',
                 '(2) refitted on degree-matched random pairs': '(2) refitted on random pairs'}
        for label, key in names.items():
            cells = []
            for v in ('N', 'C'):
                r = self.a['variants'][v]
                if key is None:
                    b, e = r['back'], r['sideslip']['median_deg']
                    kt, rho = b['K_T'], b['rho']
                else:
                    ab = r['ablations'][key]
                    kt, rho, e = ab['K_T'], ab['rho'], ab['sideslip_deg']
                cells += [f'{kt:+.3f}, {rho:.3f}', f'{e:.1f}°']
            row = f"| {label} | " + ' | '.join(cells) + ' |'
            self.assert_quoted(PAPER_B, row, f'ablation row "{label}"')

    def test_robustness_table(self):
        R = {v: self.a['variants'][v]['robustness'] for v in ('N', 'C')}
        B = self.a['plain_robustness']
        f = lambda d, k: f"{d[k]['median_deg']:.1f}°"
        for k, label in (('goals shifted 15 deg', 'goals shifted by 15°'),
                         ('random initial heading', 'random initial heading'),
                         ('fluctuating speed', 'fluctuating speed (OU, 0.5–1.5)')):
            self.assert_quoted(PAPER_B, f"| {label} | {f(B, k)} | {f(R['N'], k)} | {f(R['C'], k)} |", label)
        pair = lambda d, a, b: f"{f(d, a)} / {f(d, b)}"
        for a, b, label in (('self-motion noise 0.1', 'self-motion noise 0.2', 'self-motion noise, sd 0.1 / 0.2'),
                            ('delay 40 ms', 'delay 100 ms', 'sensory delay 40 ms / 100 ms'),
                            ('loop gain K = 1.3', 'loop gain K = 5.2', 'loop gain K = 1.3 / 5.2 rad/s')):
            row = f"| {label} | {pair(B, a, b)} | {pair(R['N'], a, b)} | {pair(R['C'], a, b)} |"
            self.assert_quoted(PAPER_B, row, label)
        seeds = lambda d: [d[f'seed set {k}']['median_deg'] for k in range(1, 5)]
        rng = lambda d: f"{min(seeds(d)):.1f}–{max(seeds(d)):.1f}°"
        self.assert_quoted(PAPER_B, f"| four other seed sets (range) | {rng(B)} | {rng(R['N'])} | {rng(R['C'])} |",
                           'seed sets')
        w = lambda d: ' / '.join(f"{d[f'world-fixed wind beta {b}']['median_deg']:.1f}°" for b in ('0.3', '0.7', '1.3'))
        self.assert_quoted(PAPER_B, f"| world-fixed wind β = 0.3 / 0.7 / 1.3 | {w(B)} | {w(R['N'])} | {w(R['C'])} |",
                           'world wind')
        lam = lambda d: f"{d['lambda x0.5']['median_deg']:.1f}° / {d['lambda x2']['median_deg']:.1f}°"
        self.assert_quoted(PAPER_B, f"| λ × 0.5 / × 2 | — | {lam(R['N'])} | {lam(R['C'])} |", 'lambda sweep')


class TestNumberSheetIsFresh(unittest.TestCase):
    """paper/data/paper_b_measurements.json must equal what fig_b.collect() reads today."""

    def test_sheet_matches_data(self):
        try:
            sys.path.insert(0, os.path.join(PAPER, 'scripts'))
            import fig_b  # noqa: E402
        except ImportError as e:  # pragma: no cover
            self.skipTest(f'needs numpy/matplotlib: {e}')
        fresh = fig_b.collect()
        aug = fig_b.collect_augmented()
        if aug is not None:
            fresh['augmented'] = aug
        fresh = json.loads(json.dumps(fresh))
        self.assertEqual(fresh, sheet('paper_b_measurements.json'),
                         'run python3 paper/scripts/fig_b.py to refresh the number sheet')


# ---------------------------------------------------------------- structure and scope
LINK = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)\)')
CJK = re.compile(r'[぀-ヿ㐀-䶿一-鿿＀-￯]')


class TestLinksAndFigures(unittest.TestCase):
    def test_relative_links_resolve(self):
        for path, text in TEXT.items():
            for target in LINK.findall(text):
                if target.startswith(('http://', 'https://', '#', 'mailto:')):
                    continue
                full = os.path.normpath(os.path.join(os.path.dirname(path), target.split('#')[0]))
                self.assertTrue(os.path.exists(full), f'{os.path.relpath(path, ROOT)} -> {target}')

    def test_every_paper_figure_is_used(self):
        for paper, path in (('a_navigation', PAPER_A), ('b_servo_control', PAPER_B)):
            figdir = os.path.join(PAPER, paper, 'figures')
            for name in os.listdir(figdir):
                self.assertIn(f'figures/{name}', TEXT[path], f'{paper}: {name} is never shown')

    def test_medium_images_match_paper_figures(self):
        """medium/images are copies; they must not drift from the paper figures."""
        if not HAS_MEDIUM:
            self.skipTest('no medium/ in this checkout')
        for name in os.listdir(os.path.join(MEDIUM, 'images')):
            src = os.path.join(PAPER, 'a_navigation' if name.startswith('figA') else 'b_servo_control',
                               'figures', name)
            with open(src, 'rb') as a, open(os.path.join(MEDIUM, 'images', name), 'rb') as b:
                self.assertEqual(a.read(), b.read(), f'medium/images/{name} is stale')

    def test_html_is_rendered(self):
        for paper, path in (('a_navigation', PAPER_A), ('b_servo_control', PAPER_B)):
            html = read(os.path.join(PAPER, paper, 'PAPER.html'))
            title = TEXT[path].splitlines()[0].lstrip('# ').strip()
            self.assertIn('<h1', html)
            self.assertIn(title.split('*')[0][:40], html)
            self.assertNotIn('](figures/', html, f'{paper}: an image was not embedded')


class TestEnglishOnly(unittest.TestCase):
    def test_no_cjk_in_deliverables(self):
        files = list(TEXT)
        for d in (os.path.join(PAPER, 'scripts'),):
            files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.py', '.sh'))]
        files += [p for p in (os.path.join(PAPER, 'README.md'), os.path.join(MEDIUM, 'README.md'))
                  if os.path.exists(p)]
        for path in files:
            m = CJK.search(read(path))
            self.assertIsNone(m, f'{os.path.relpath(path, ROOT)} contains "{m.group(0) if m else ""}"')


class TestScope(unittest.TestCase):
    AUGMENTED = ('15.14', '0.921', '160.49', '0.703', '36.48', 'GPT-6', 'Astra')

    def test_paper_b_uses_no_augmented_numbers(self):
        for path in (p for p in (PAPER_B, MED_B) if p in TEXT):
            for s in self.AUGMENTED:
                self.assertFalse(s in TEXT[path], f'{os.path.relpath(path, ROOT)} cites "{s}"')

    def test_paper_a_stays_on_the_navigation_algorithm(self):
        for path in (p for p in (PAPER_A, MED_A) if p in TEXT):
            body = TEXT[path].split('## References')[0].lower()   # cited titles may say anything
            for s in ('path integration', 'path-integrat', 'hδb →', 'shunting'):
                self.assertFalse(s in body, f'{os.path.relpath(path, ROOT)} strays into "{s}"')


@unittest.skipUnless(HAS_MEDIUM, 'no medium/ in this checkout')
class TestMediumFormat(unittest.TestCase):
    def test_no_latex_and_no_tables(self):
        for path in (MED_A, MED_B):
            text = TEXT[path]
            self.assertNotIn('$$', text)
            self.assertNotIn('\\(', text)
            self.assertIsNone(re.search(r'^\s*\|.*\|\s*$', text, re.M),
                              f'{os.path.relpath(path, ROOT)} has a Markdown table')

    def test_links_back_to_paper(self):
        self.assertIn('../paper/a_navigation/PAPER.md', TEXT[MED_A])
        self.assertIn('../paper/b_servo_control/PAPER.md', TEXT[MED_B])

    def test_length(self):
        for path in (MED_A, MED_B):
            words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", TEXT[path]))
            self.assertTrue(1500 <= words <= 3500, f'{os.path.relpath(path, ROOT)}: {words} words')


class TestPaperStructure(unittest.TestCase):
    REQUIRED = ('## Abstract', '## 1. Introduction', '## Reproduction', '## References', 'Limitations')

    def test_sections(self):
        for path in (PAPER_A, PAPER_B):
            for s in self.REQUIRED:
                self.assertIn(s, TEXT[path], f'{os.path.relpath(path, ROOT)} lacks {s}')

    def test_figures_are_numbered_in_order(self):
        for path in (PAPER_A, PAPER_B):
            nums = [int(n) for n in re.findall(r'^\*\*Figure (\d+)\.\*\*', TEXT[path], re.M)]
            self.assertEqual(nums, list(range(1, len(nums) + 1)), os.path.relpath(path, ROOT))


if __name__ == '__main__':
    unittest.main(verbosity=1)
