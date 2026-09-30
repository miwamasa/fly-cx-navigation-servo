#!/usr/bin/env python3
"""PFL3 に何が入っているかを実測する — heading servo か travel servo かを配線で判定する。

  python3 scripts/pfl3_inputs.py
  → data/pfl3_inputs.json

なぜこれを測るか
----------------
note 記事の整理では、PFL3 の制御則が 3 通りありうる。

  Model A（heading servo）        u = K·sin(G − H)     H は頭方位（EPG/Δ7 由来）
  Model B（travel servo）         u = K·sin(G − T)     T は進行方向（hΔB 由来）
  Model C（混合）                 u = K_H·sin(G−H) + K_T·sin(G−T)

どれが成立しうるかは、**PFL3 が H と T をそれぞれどれだけ受けているか**で決まる。
これはコネクトームで直接測れる量である。記事が言うとおり、文献で直接実証されて
いるのは EPG/Δ7 → PFL3 の経路であって、hΔB → PFL3 ではない。
雄コネクトームではどうなっているかを、こちらで測る。

測るもの
--------
  composition   PFL3 の入力を前シナプス型ごとに分解（興奮性・抑制性を分けて）
  routes        H 系（EPG/Δ7/PEG/PEN）・T 系（hΔB 経由）・G 系（FC/FS/FR）の割合
  offsets       型ごとの位相オフセット（円周平均と集中度）
  hdb_to_fc     hΔB → FC2 の直接経路があるか（記事の「FC をからめて」の部分）
  two_hop       hΔB から PFL3 への 2 ホップ経路の強さ（中継先ごと）

注意
----
**ここで測るのは配線であって、動かしたときの寄与ではない。**
入力の重みが大きくても、その集団が動作点で沈黙していれば寄与は 0 である
（6.9 節の PFNv がまさにそれだった）。動かしたときの寄与は別途 pfl3_servo.py で測る。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

NPZ = os.path.join(ROOT, 'data', 'cx_network.npz')
OUT = os.path.join(ROOT, 'data', 'pfl3_inputs.json')
TAU = 2 * np.pi

# 記事の 3 モデルに対応する、入力の系統わけ。
# 前シナプス型の接頭辞で分類する（type 名は MaleCNS のもの）。
ROUTE = {
    'H': ['EPG', 'EPGt', 'PEG', 'PEN', 'Delta7', 'D7'],      # 頭方位 H
    'T': ['hDeltaB'],                                         # 進行方向 T（の担い手）
    'G': ['FC1', 'FC2', 'FC3', 'FS', 'FR'],                   # 目標 G
    'hD_other': ['hDelta'],                                   # hΔB 以外の hΔ
    'PFN': ['PFN'],
    'vD': ['vDelta'],
}


def load():
    d = np.load(NPZ, allow_pickle=True)
    return {'types': np.array([str(x) for x in d['types']]),
            'phase': d['phase'], 'side': d['side'],
            'pre': d['pre'], 'post': d['post'], 'w': d['weight']}


def base_type(t):
    """型名から側や添字を落とした基本名。hDeltaB_a → hDeltaB。"""
    return re.sub(r'[_\s].*$', '', str(t))


def route_of(t):
    """前シナプス型を H / T / G / … のどれに割り当てるか。長い接頭辞を優先。"""
    b = base_type(t)
    best = None
    for route, prefixes in ROUTE.items():
        for p in prefixes:
            if b.startswith(p) and (best is None or len(p) > best[1]):
                best = (route, len(p))
    return best[0] if best else 'other'


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def circ(dphi, w):
    """位相差の重みつき円周平均と集中度。"""
    if len(dphi) == 0 or np.sum(w) <= 0:
        return None, 0.0
    x = float(np.sum(w * np.cos(dphi))); y = float(np.sum(w * np.sin(dphi)))
    R = float(np.hypot(x, y) / np.sum(np.abs(w)))
    return float(np.degrees(np.arctan2(y, x))), R


def inputs_to(D, targets, signed=True):
    """targets への入力を前シナプス型ごとに集計する。

    signed=True なら重みは符号つき（抑制性は負）。位相オフセットは
    「後シナプスの位相 − 前シナプスの位相」で、重みの絶対値で重みづけする。
    """
    tset = set(int(i) for i in np.asarray(targets))
    ph = D['phase']
    agg = collections.defaultdict(lambda: {'w': 0.0, 'w_abs': 0.0, 'n_conn': 0,
                                           'd': [], 'dw': []})
    for i, j, w in zip(D['post'], D['pre'], D['w']):
        i, j, w = int(i), int(j), float(w)
        if i not in tset:
            continue
        t = base_type(D['types'][j])
        a = agg[t]
        a['w'] += w if signed else abs(w)
        a['w_abs'] += abs(w)
        a['n_conn'] += 1
        if ph[i] > -8 and ph[j] > -8:
            a['d'].append(wrap(ph[i] - ph[j])); a['dw'].append(abs(w))

    tot_abs = sum(a['w_abs'] for a in agg.values()) or 1.0
    out = {}
    for t, a in agg.items():
        off, R = circ(np.array(a['d']), np.array(a['dw']))
        out[t] = {'weight': a['w'], 'weight_abs': a['w_abs'],
                  'share': a['w_abs'] / tot_abs,
                  'n_connections': a['n_conn'],
                  'n_synapses': int(round(a['w_abs'])),
                  'sign': 'exc' if a['w'] > 0 else ('inh' if a['w'] < 0 else 'mixed'),
                  'offset_deg': off, 'offset_R': R, 'route': route_of(t)}
    return out, tot_abs


def by_route(comp):
    """型ごとの内訳を H / T / G … に畳む。"""
    out = collections.defaultdict(lambda: {'share': 0.0, 'weight': 0.0, 'types': []})
    for t, v in comp.items():
        r = out[v['route']]
        r['share'] += v['share']; r['weight'] += v['weight']
        r['types'].append(t)
    for r in out.values():
        r['types'] = sorted(r['types'])
    return dict(out)


def path_weight(D, src, dst):
    """src → dst の符号つきシナプス数と、結合（細胞ペア）の本数。

    **この 2 つは別の量である。** 以前は本数のほうを 'n_syn' と呼んでいて、
    本文に「10 シナプス」と誤って書いてしまった（正しくは結合 10 本・シナプス 32 個）。
    """
    s = set(int(i) for i in np.asarray(src)); t = set(int(i) for i in np.asarray(dst))
    w = 0.0; n = 0
    for i, j, ww in zip(D['post'], D['pre'], D['w']):
        if int(i) in t and int(j) in s:
            w += float(ww); n += 1
    return w, n


def two_hop(D, src, dst, min_share=0.01):
    """src → X → dst の 2 ホップ経路を、中継 X の型ごとに強さ順で返す。

    強さは「src→X の重み（正のぶん）× X→dst の重み（正のぶん）」の和を、
    中継の型ごとにまとめたもの。順位づけのための量であって、
    実際の伝達量ではない（動作点に依存するため）。
    """
    s = set(int(i) for i in np.asarray(src)); t = set(int(i) for i in np.asarray(dst))
    out_of_src = collections.defaultdict(float)   # X -> src からの重み
    into_dst = collections.defaultdict(float)     # X -> dst への重み
    for i, j, w in zip(D['post'], D['pre'], D['w']):
        i, j, w = int(i), int(j), float(w)
        if j in s and i not in s:
            out_of_src[i] += w
        if i in t:
            into_dst[j] += w
    prod = collections.defaultdict(lambda: {'via': 0.0, 'n_cells': 0})
    for x, w1 in out_of_src.items():
        w2 = into_dst.get(x, 0.0)
        if w1 <= 0 or w2 == 0:
            continue
        p = prod[base_type(D['types'][x])]
        p['via'] += w1 * w2; p['n_cells'] += 1
    tot = sum(abs(v['via']) for v in prod.values()) or 1.0
    rows = [{'via_type': k, 'strength': v['via'], 'share': abs(v['via']) / tot,
             'n_cells': v['n_cells']}
            for k, v in prod.items() if abs(v['via']) / tot >= min_share]
    return sorted(rows, key=lambda r: -abs(r['strength']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=OUT)
    a = ap.parse_args()

    D = load()
    T = D['types']
    idx = lambda pred: np.array([i for i in range(len(T)) if pred(T[i])], dtype=np.int64)

    pfl3 = idx(lambda t: base_type(t).startswith('PFL3'))
    fc2 = idx(lambda t: base_type(t).startswith('FC2'))
    hdb = idx(lambda t: base_type(t) == 'hDeltaB')
    epg = idx(lambda t: base_type(t) in ('EPG', 'EPGt'))
    d7 = idx(lambda t: base_type(t).startswith('Delta7'))

    print(f'PFL3 {len(pfl3)} 細胞 / FC2 {len(fc2)} / hΔB {len(hdb)} / '
          f'EPG {len(epg)} / Δ7 {len(d7)}')

    comp, tot = inputs_to(D, pfl3)
    routes = by_route(comp)
    print(f'\n=== PFL3 の入力（総重み {tot:.0f}、絶対値ベースの割合）')
    for t, v in sorted(comp.items(), key=lambda kv: -kv[1]['share'])[:14]:
        off = '—' if v['offset_deg'] is None else f"{v['offset_deg']:+6.1f}°(R={v['offset_R']:.2f})"
        print(f"  {t:14s} {v['share']*100:5.1f}%  {v['sign']:5s} {off}  [{v['route']}]")
    print('\n  系統ごと:', ', '.join(
        f"{k} {v['share']*100:.1f}%" for k, v in
        sorted(routes.items(), key=lambda kv: -kv[1]['share'])))

    # FC2 の入力も見る（記事の「FC をからめて」の部分）
    comp_fc2, tot_fc2 = inputs_to(D, fc2)
    routes_fc2 = by_route(comp_fc2)
    print(f'\n=== FC2 の入力（総重み {tot_fc2:.0f}）')
    for t, v in sorted(comp_fc2.items(), key=lambda kv: -kv[1]['share'])[:10]:
        off = '—' if v['offset_deg'] is None else f"{v['offset_deg']:+6.1f}°(R={v['offset_R']:.2f})"
        print(f"  {t:14s} {v['share']*100:5.1f}%  {v['sign']:5s} {off}  [{v['route']}]")

    # 直接経路
    w_hdb_pfl3, n1 = path_weight(D, hdb, pfl3)
    w_hdb_fc2, n2 = path_weight(D, hdb, fc2)
    w_fc2_pfl3, n3 = path_weight(D, fc2, pfl3)
    w_epg_pfl3, n4 = path_weight(D, epg, pfl3)
    w_d7_pfl3, n5 = path_weight(D, d7, pfl3)
    print('\n=== 直接経路の総重み（符号つき / シナプス数）')
    for lab, w, n in (('hΔB → PFL3', w_hdb_pfl3, n1), ('hΔB → FC2', w_hdb_fc2, n2),
                      ('FC2  → PFL3', w_fc2_pfl3, n3), ('EPG  → PFL3', w_epg_pfl3, n4),
                      ('Δ7   → PFL3', w_d7_pfl3, n5)):
        print(f'  {lab:12s} シナプス {w:+10.0f}  （結合 {n} 本）')

    # 左右を分けて測る。4 節で報告した FC2→PFL3 の ±73° は左右別の量なので、
    # 左右をまとめると打ち消し合って 0 付近・低い集中度になる（上の表がそれ）。
    side = D['side']
    per_side = {}
    print('\n=== 左右別（4 節の ±73° と突き合わせる）')
    print(f"{'型':14s} {'左 PFL3':>22s} {'右 PFL3':>22s}")
    for sd, lab in ((-1, 'L'), (+1, 'R')):
        tgt = np.array([i for i in pfl3 if side[i] == sd], dtype=np.int64)
        comp_s, tot_s = inputs_to(D, tgt)
        per_side[lab] = {'inputs': comp_s, 'total_abs': tot_s,
                         'routes': by_route(comp_s), 'n_cells': len(tgt)}
    for ty in ('FC2A', 'FC2B', 'FC2C', 'EPG', 'Delta7', 'hDeltaA', 'hDeltaI'):
        cells = []
        for lab in ('L', 'R'):
            v = per_side[lab]['inputs'].get(ty)
            cells.append('—' if not v or v['offset_deg'] is None else
                         f"{v['offset_deg']:+7.1f}° R={v['offset_R']:.2f} "
                         f"{v['share']*100:4.1f}%")
        print(f'  {ty:14s} {cells[0]:>22s} {cells[1]:>22s}')

    th = two_hop(D, hdb, pfl3)
    print('\n=== hΔB → X → PFL3 の中継（強さ順）')
    for r in th[:8]:
        print(f"  {r['via_type']:14s} 割合 {r['share']*100:5.1f}%  "
              f"({r['n_cells']} 細胞)")

    res = {
        'note': 'これは配線の測定であって、動作点での寄与ではない。'
                '重みが大きくても沈黙していれば寄与は 0（6.9 節の PFNv）。',
        'counts': {'PFL3': len(pfl3), 'FC2': len(fc2), 'hDeltaB': len(hdb),
                   'EPG': len(epg), 'Delta7': len(d7)},
        'pfl3_inputs': comp, 'pfl3_input_total_abs': tot, 'pfl3_routes': routes,
        'fc2_inputs': comp_fc2, 'fc2_input_total_abs': tot_fc2, 'fc2_routes': routes_fc2,
        'direct_note': 'weight = 符号つきシナプス数、n_connections = 細胞ペアの本数。'
                       'この 2 つは別物（hΔB→PFL3 は結合 10 本・シナプス 32 個）。',
        'direct': {'hdb_to_pfl3': {'weight': w_hdb_pfl3, 'n_connections': n1,
                                   'n_synapses': int(round(abs(w_hdb_pfl3)))},
                   'hdb_to_fc2': {'weight': w_hdb_fc2, 'n_connections': n2,
                                  'n_synapses': int(round(abs(w_hdb_fc2)))},
                   'fc2_to_pfl3': {'weight': w_fc2_pfl3, 'n_connections': n3,
                                   'n_synapses': int(round(abs(w_fc2_pfl3)))},
                   'epg_to_pfl3': {'weight': w_epg_pfl3, 'n_connections': n4,
                                   'n_synapses': int(round(abs(w_epg_pfl3)))},
                   'd7_to_pfl3': {'weight': w_d7_pfl3, 'n_connections': n5,
                                  'n_synapses': int(round(abs(w_d7_pfl3)))}},
        'hdb_to_pfl3_two_hop': th,
        'pfl3_inputs_per_side': per_side,
    }
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f'\n→ {a.out}')


if __name__ == '__main__':
    main()
