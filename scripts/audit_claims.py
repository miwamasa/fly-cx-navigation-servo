#!/usr/bin/env python3
"""誤差棒の棚卸し（配線・集団活動の側）。

  python3 scripts/audit_claims.py
  → data/claims_audit_wiring.json

なぜこれが要るのか
------------------
本プロジェクトは「軌跡 6 本・単一シードの点推定」を陽性結果として論文に書き、
あとで再現しないことが分かった（PATH_INTEGRATION 6 節）。同じ危険は他にもある。
結線カーネルのピーク（FC2→PFL3 の ∓73°、左右 PFN の 66° ずれ）も、
β を振ったときの集団ベクトルの回転（27〜53°）も、1 回測った値をそのまま書いていた。

ここでは 2 種類の誤差を付ける。

  * **細胞のブートストラップ** — カーネルのピークは「たまたまこの細胞群だったから」
    かもしれない。前シナプス細胞を復元抽出して測り直し、分布を見る。
  * **動作点の掃引** — 集団ベクトルの回転量は自己運動ゲインに強く依存する。
    1 点だけ報告すると、選んだゲインが結論を決めていることになる。

いずれも「壊れているか」ではなく「誤差棒を付けても同じことが言えるか」を確かめる。
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

from cxnet_np import CXNetworkNP                       # noqa: E402
from pi_task import SelfMotionInput                    # noqa: E402

NPZ = os.path.join(ROOT, 'data', 'cx_network.npz')
MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
OUT = os.path.join(ROOT, 'data', 'claims_audit_wiring.json')
TAU = 2 * np.pi


def load_npz():
    d = np.load(NPZ, allow_pickle=True)
    types = np.array([str(x) for x in d['types']])
    n = len(types)
    W = np.zeros((n, n), dtype=np.float32)
    W[d['post'], d['pre']] = d['weight']
    return {'types': types, 'phase': d['phase'], 'pb_side': d['pb_side'],
            'side': d['side'], 'W': W, 'n': n}


def grp(D, *prefixes):
    return np.array([i for i in range(D['n'])
                     if any(D['types'][i].startswith(p) for p in prefixes)], dtype=int)


def kernel_peak(D, pre, post, bins=16):
    """位相差でビン分けし、正の部分の円周平均でピーク位相を出す。"""
    ph = D['phase']
    pre = np.asarray(pre); post = np.asarray(post)
    pre = pre[ph[pre] > -8]; post = post[ph[post] > -8]
    if len(pre) == 0 or len(post) == 0:
        return None
    acc = np.zeros(bins); cnt = np.zeros(bins)
    sub = D['W'][np.ix_(post, pre)]
    for a, i in enumerate(post):
        k = ((ph[i] - ph[pre]) % TAU / TAU * bins).astype(int) % bins
        np.add.at(acc, k, sub[a]); np.add.at(cnt, k, 1)
    prof = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    pos = np.maximum(prof, 0)
    if pos.sum() <= 0:
        return None
    ang = (np.arange(bins) + 0.5) / bins * TAU
    pk = np.degrees(np.arctan2((pos * np.sin(ang)).sum(), (pos * np.cos(ang)).sum()))
    return float((pk + 180) % 360 - 180)


def bootstrap_peaks(D, pre, post, n_boot=400, seed=0):
    """前シナプス細胞を復元抽出してカーネルのピークを測り直す。

    返すのは「全標本での点推定」と、その周りのブートストラップ偏差から作った
    基本ブートストラップ信頼区間。復元抽出の平均をそのまま代表値にすると、
    標本が小さく歪んでいるときに点推定から系統的にずれるので、そうしない。
    """
    rng = np.random.default_rng(seed)
    pre = np.asarray(pre)
    pre = pre[D['phase'][pre] > -8]
    point = kernel_peak(D, pre, post)
    if point is None:
        return None
    devs = []
    for _ in range(n_boot):
        s = rng.choice(pre, size=len(pre), replace=True)
        v = kernel_peak(D, s, post)
        if v is not None:
            devs.append(np.degrees((np.radians(v - point) + np.pi) % TAU - np.pi))
    d = np.asarray(devs)
    return {'point_deg': float(point),
            'ci95_deg': [float(point + np.percentile(d, 2.5)),
                         float(point + np.percentile(d, 97.5))],
            'sd_deg': float(d.std(ddof=1)), 'n_cells': int(len(pre)),
            'n_boot': int(len(d))}


def pfl3_offsets(D, pfl3_sel, epg, goal):
    """build_gguf.measure_pfl3_offsets と同じ量を、指定した PFL3 細胞集合で測る。

    各 PFL3 細胞について「目標入力の平均位相 − 方位入力の平均位相」を出し、
    その円周平均をとる。文書が引用している −73.1° / +73.0° はこの量。
    """
    ph, W = D['phase'], D['W']

    def mean_phase(cell, src):
        w = np.maximum(W[cell, src], 0)
        if w.sum() <= 0:
            return None
        return np.arctan2((w * np.sin(ph[src])).sum(), (w * np.cos(ph[src])).sum())

    ds = []
    for b in pfl3_sel:
        h, g = mean_phase(b, epg), mean_phase(b, goal)
        if h is None or g is None:
            continue
        ds.append(g - h)
    if not ds:
        return None
    d = np.asarray(ds)
    return float(np.arctan2(np.mean(np.sin(d)), np.mean(np.cos(d)))), d


def bootstrap_pfl3(D, pfl3_sel, epg, goal, n_boot=2000, seed=0):
    """PFL3 細胞を復元抽出して、方位入力と目標入力の位相差の信頼区間を出す。"""
    r = pfl3_offsets(D, pfl3_sel, epg, goal)
    if r is None:
        return None
    point, per_cell = r
    rng = np.random.default_rng(seed)
    devs = []
    for _ in range(n_boot):
        s = per_cell[rng.integers(0, len(per_cell), len(per_cell))]
        m = np.arctan2(np.mean(np.sin(s)), np.mean(np.cos(s)))
        devs.append(np.degrees((m - point + np.pi) % TAU - np.pi))
    d = np.asarray(devs)
    return {'point_deg': float(np.degrees(point)),
            'ci95_deg': [float(np.degrees(point) + np.percentile(d, 2.5)),
                         float(np.degrees(point) + np.percentile(d, 97.5))],
            'sd_deg': float(d.std(ddof=1)), 'n_cells': int(len(per_cell)),
            'per_cell_deg': [float(np.degrees(x)) for x in per_cell]}


def pfn_rotation(net, smi, beta_deg, settle=400):
    """横滑り β を与えて落ち着かせ、PFN の集団ベクトルの角度を返す。"""
    ph = net.phase
    sel = net.pfn[ph[net.pfn] > -8]
    net.reset(1)
    for _ in range(settle):
        net.clear_drive()
        net.set_visual_scene(0.0, 1.0)
        smi.apply(net.clamp_val, 1.0, np.radians(beta_deg))
        net.step(0.02)
    a = net.r[sel]
    if a.sum() <= 1e-9:
        return None, sel, a
    return float(np.arctan2((a * np.sin(ph[sel])).sum(),
                            (a * np.cos(ph[sel])).sum())), sel, a


def rotation_with_cell_bootstrap(net, gain, n_boot=400, seed=0):
    """β = −60° → +60° で PFN 集団ベクトルが何度回るか。細胞ブートストラップつき。"""
    smi = SelfMotionInput(net, lno_gain=gain, spsp_gain=gain)
    ph = net.phase
    acts = {}
    for b in (-60.0, +60.0):
        ang, sel, a = pfn_rotation(net, smi, b)
        if ang is None:
            return None
        acts[b] = (sel, a.copy())
    rng = np.random.default_rng(seed)
    sel = acts[-60.0][0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(sel), len(sel))
        out = []
        for b in (-60.0, +60.0):
            s, a = acts[b]
            aa = a[idx]; pp = ph[s[idx]]
            out.append(np.arctan2((aa * np.sin(pp)).sum(), (aa * np.cos(pp)).sum()))
        vals.append(np.degrees((out[1] - out[0] + np.pi) % TAU - np.pi))
    v = np.asarray(vals)
    return {'rotation_deg': float(np.median(v)),
            'ci95_deg': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            'n_boot': int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=400)
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()

    D = load_npz()
    res = {'claims': {}}

    print('1. PFL3 が受け取る「目標 − 方位」の位相差（PFL3 細胞のブートストラップ）')
    goal = grp(D, 'FC2', 'hDeltaB')
    goal = goal[D['phase'][goal] > -8]
    epg = grp(D, 'EPG')
    epg = epg[D['phase'][epg] > -8]
    pfl3 = grp(D, 'PFL3')
    kern = {}
    for name, sd in (('左 PFL3', -1), ('右 PFL3', +1)):
        post = pfl3[D['side'][pfl3] == sd]
        kern[name] = bootstrap_pfl3(D, post, epg, goal)
        k = kern[name]
        print(f'   {name}: {k["point_deg"]:+7.1f}° '
              f'[{k["ci95_deg"][0]:+.1f}, {k["ci95_deg"][1]:+.1f}]  '
              f'n={k["n_cells"]} 細胞')
    lo = np.radians(kern['左 PFL3']['point_deg'])
    hi = np.radians(kern['右 PFL3']['point_deg'])
    null = np.degrees(np.arctan2(np.sin(lo) + np.sin(hi), np.cos(lo) + np.cos(hi)))
    kern['釣り合い点'] = {'point_deg': float(null)}
    print(f'   釣り合い点（左右の中点）: {null:+.2f}°')
    res['claims']['pfl3_goal_minus_heading_deg'] = kern

    print('2. 左右 PFN が扇状体へ書き込む位相のずれ')
    vd = grp(D, 'vDelta')
    off = {}
    for t in ('PFNd', 'PFNv', 'PFNa'):
        g = grp(D, t)
        gl = g[D['pb_side'][g] == -1]; gr = g[D['pb_side'][g] == +1]
        L = bootstrap_peaks(D, gl, vd, n_boot=a.boot, seed=1)
        R = bootstrap_peaks(D, gr, vd, n_boot=a.boot, seed=2)
        d = (L['point_deg'] - R['point_deg'] + 180) % 360 - 180
        off[t] = {'left': L, 'right': R, 'offset_deg': float(d)}
        print(f'   {t}→vΔ 左 {L["point_deg"]:+.1f}° '
              f'[{L["ci95_deg"][0]:+.1f}, {L["ci95_deg"][1]:+.1f}]  '
              f'右 {R["point_deg"]:+.1f}° '
              f'[{R["ci95_deg"][0]:+.1f}, {R["ci95_deg"][1]:+.1f}]  → ずれ {d:+.1f}°')
    res['claims']['pfn_lr_write_offset'] = off

    print('3. β を振ったときの PFN 集団ベクトルの回転（自己運動ゲインを掃引）')
    net = CXNetworkNP(MODEL)
    rot = {}
    for gain in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        r = rotation_with_cell_bootstrap(net, gain, n_boot=a.boot)
        if r is None:
            print(f'   ゲイン {gain:.2f}: PFN が沈黙して測れない')
            continue
        rot[f'{gain:.2f}'] = r
        print(f'   ゲイン {gain:.2f}: {r["rotation_deg"]:+6.1f}° '
              f'[{r["ci95_deg"][0]:+.1f}, {r["ci95_deg"][1]:+.1f}]')
    res['claims']['pfn_rotation_by_gain_deg'] = rot
    vals = [v['rotation_deg'] for v in rot.values()]
    res['claims']['pfn_rotation_range_deg'] = {
        'min': float(min(vals)), 'max': float(max(vals)),
        'gains': sorted(rot.keys()),
        'note': '回転量は自己運動ゲインに強く依存する。1 点だけ報告してはいけない。'}

    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
