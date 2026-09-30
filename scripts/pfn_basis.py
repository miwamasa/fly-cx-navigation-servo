#!/usr/bin/env python3
"""B1: PFNd/PFNv → hΔB の「4 基底」構造が雄コネクトームにあるかを実測する。

  python3 scripts/pfn_basis.py
  → data/pfn_basis.json

文献（Lyu/Abbott/Maimon, Lu ら 2022）の主張はこうである。

  ・PFNd と PFNv は「頭方位 × 身体座標の移動方向」の**結合表現**を持つ
  ・左右の PFNd（前方・同側寄り）と左右の PFNv（後方・対側寄り）が
    **4 本のベクトル基底**をなす
  ・同じ *世界座標の* 進行方向を表す 4 集団が**同じ hΔB へ収束**し、
    その和として hΔB が世界座標の進行方向を表す

6.7〜6.8 節でうちのモデルは「PFN→hΔ の変換段が逆向きの変位を表現できない」で
止まっているが、うちの自己運動入力は左右の視覚流を LNO と SpsP に同じ形で入れており、
**PFNv に独立した駆動が無い**。文献の 4 基底のうち 2 本しか使っていない可能性がある。

そこで、機構を足す前に配線の側を確かめる。測るのは 3 つ。

  1. 自己運動の入口の分岐。LNO1/LNO2/LNOa/SpsP がそれぞれ PFNd/PFNv/PFNa の
     どれへ、どちらの側へ入るか。
  2. PFNd/PFNv 左右（4 集団）が hΔB へ書き込む位相。hΔB の位相座標は
     6.8 節で結線から決め直したもの（出力側、R ≥ 0.3）を使う。
  3. 収束。各 hΔB について 4 集団それぞれからの入力重みを出し、
     「4 集団すべてが同じ hΔB へ入る」のか「集団ごとに別の hΔB へ入る」のかを見る。

判定は先に決めておく。

  ・4 集団が同じ hΔB に収束している ⇔ 各 hΔB が 4 集団すべてから
    （総入力の 5% 以上を）受けている割合が過半
  ・基底が 4 方向に散っている ⇔ 4 集団の書き込み位相の差が
    互いに 45° 以上離れている
"""

from __future__ import annotations

import collections
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NPZ = os.path.join(ROOT, 'data', 'cx_network.npz')
HDP = os.path.join(ROOT, 'data', 'hdelta_phase.json')
OUT = os.path.join(ROOT, 'data', 'pfn_basis.json')
TAU = 2 * np.pi


def wrap(a):
    return (a + np.pi) % TAU - np.pi


def circ_mean(ph, wt):
    x = float((wt * np.cos(ph)).sum()); y = float((wt * np.sin(ph)).sum())
    s = float(wt.sum())
    return (float(np.arctan2(y, x)) % TAU, (np.hypot(x, y) / s) if s > 0 else 0.0, s)


def load():
    d = np.load(NPZ, allow_pickle=True)
    return {'names': [str(n) for n in d['names']],
            'types': [str(t) for t in d['types']],
            'pre': d['pre'], 'post': d['post'], 'weight': d['weight'].astype(float),
            'phase': d['phase'].astype(float), 'side': d['side'],
            'pb_side': d['pb_side'], 'body_id': d['body_id']}


def fam(t):
    return re.sub(r'_.*', '', t)


def input_routing(g):
    """自己運動の入口: LNO/SpsP の型 × 側 → PFN の型 × 側 の重み。"""
    names, types, pre, post, w = g['names'], g['types'], g['pre'], g['post'], g['weight']
    side, pb = g['side'], g['pb_side']
    rows = collections.defaultdict(float)
    for k in range(len(pre)):
        a, b = int(pre[k]), int(post[k])
        ta, tb = types[a], types[b]
        if not fam(ta).startswith(('LNO', 'SpsP')):
            continue
        if not fam(tb).startswith('PFN'):
            continue
        sa = {-1: 'L', 1: 'R'}.get(int(side[a]), '?')
        sb = {-1: 'L', 1: 'R'}.get(int(pb[b]), '?')
        rows[(ta, sa, fam(tb), sb)] += abs(w[k])
    out = []
    by_src = collections.defaultdict(float)
    for key, v in rows.items():
        by_src[(key[0], key[1])] += v
    for (ta, sa, tb, sb), v in sorted(rows.items(), key=lambda x: -x[1]):
        out.append({'src_type': ta, 'src_side': sa, 'dst_type': tb, 'dst_pb_side': sb,
                    'weight': v, 'frac_of_src': v / by_src[(ta, sa)]})
    return out


def basis_writes(g, hd_phase, groups):
    """4 集団それぞれの **書き込みオフセット** と、hΔB ごとの内訳。

    集団全体の「書き込み位相」を平均しても意味が無い（どの集団も全列に散っているので
    ほぼ一様になる。最初これで測って R = 0.03 という無意味な値を出した）。
    測るべきは **各 PFN 細胞の位相と、その細胞が書き込む hΔB の位相の差** で、
    これが集団ごとに揃っていれば「位相地図を一定量ずらして書く」＝ベクトル基底になる。
    """
    pre, post, w, phase = g['pre'], g['post'], g['weight'], g['phase']
    per_group_off = collections.defaultdict(lambda: ([], []))
    per_cell = collections.defaultdict(lambda: collections.defaultdict(float))
    for k in range(len(pre)):
        a, b = int(pre[k]), int(post[k])
        if b not in hd_phase or w[k] <= 0:
            continue
        gname = groups.get(a)
        if gname is None or phase[a] <= -8:
            continue
        d, ww = per_group_off[gname]
        d.append(wrap(hd_phase[b] - phase[a])); ww.append(w[k])
        per_cell[b][gname] += w[k]

    per_group = {}
    for gname, (d, ww) in per_group_off.items():
        d = np.array(d); ww = np.array(ww)
        m, R, s = circ_mean(d, ww)
        per_group[gname] = {'write_offset': float(wrap(m)), 'R': R, 'total_weight': s,
                            'n_synapses': len(d)}

    # 収束: 各 hΔB が 4 集団のうちいくつから「総入力の 5% 以上」を受けているか
    conv = []
    for b, dd in per_cell.items():
        tot = sum(dd.values())
        got = [gname for gname, v in dd.items() if v / tot >= 0.05]
        conv.append({'cell': int(b), 'n_groups': len(got), 'groups': sorted(got),
                     'total': tot,
                     'frac': {gname: v / tot for gname, v in sorted(dd.items())}})
    return per_group, conv


def main():
    g = load()
    hp = json.load(open(HDP, encoding='utf-8'))
    # hΔB の位相は 6.8 節の結線由来（出力側、R ≥ 0.3）
    idx_by_body = {int(b): i for i, b in enumerate(g['body_id'])}
    hd_phase = {idx_by_body[c['body_id']]: c['phase_out']
                for c in hp['cells'] if c['type'] == 'hDeltaB' and c['R_out'] >= 0.3}
    print(f"hΔB で位相が定まる細胞: {len(hd_phase)} / "
          f"{sum(1 for c in hp['cells'] if c['type'] == 'hDeltaB')}")

    # 4 集団 = PFNd/PFNv × PB 左右
    groups = {}
    for i, t in enumerate(g['types']):
        f = fam(t)
        if f in ('PFNd', 'PFNv'):
            s = {-1: 'L', 1: 'R'}.get(int(g['pb_side'][i]))
            if s:
                groups[i] = f'{f}-{s}'

    routing = input_routing(g)
    per_group, conv = basis_writes(g, hd_phase, groups)

    print('\n自己運動の入口（重みの上位）:')
    for r in routing[:12]:
        print(f"  {r['src_type']:6s}{r['src_side']} → {r['dst_type']:5s}(PB{r['dst_pb_side']})  "
              f"{r['weight']:7.0f}  （その入口の {r['frac_of_src']*100:.0f}%）")

    print('\n4 集団の書き込みオフセット（hΔB の位相 − その PFN 自身の位相）:')
    for gname, v in sorted(per_group.items()):
        print(f"  {gname:8s} {np.degrees(v['write_offset']):+7.1f}°  集中度 R={v['R']:.2f}  "
              f"総重み {v['total_weight']:7.0f}  シナプス {v['n_synapses']}")
    names = sorted(per_group)
    print('\n  オフセットの差（基底が何度離れているか）:')
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dd = np.degrees(wrap(per_group[names[i]]['write_offset']
                                 - per_group[names[j]]['write_offset']))
            print(f"    {names[i]:8s} − {names[j]:8s} = {dd:+7.1f}°")

    n_all = sum(1 for c in conv if c['n_groups'] == 4)
    hist = collections.Counter(c['n_groups'] for c in conv)
    print(f"\n収束（総入力の 5% 以上を受けている集団の数）: "
          f"{dict(sorted(hist.items()))}  → 4 集団すべて: {n_all}/{len(conv)}")

    res = {'n_hdb_with_phase': len(hd_phase),
           'routing': routing, 'groups': per_group, 'convergence': conv,
           'convergence_hist': {str(k): v for k, v in sorted(hist.items())},
           'n_all_four': n_all, 'n_hdb_cells': len(conv),
           'pairwise_deg': {f'{names[i]}−{names[j]}':
                            float(np.degrees(wrap(per_group[names[i]]['write_offset']
                                                  - per_group[names[j]]['write_offset'])))
                            for i in range(len(names)) for j in range(i + 1, len(names))}}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f'\n→ {OUT}')


if __name__ == '__main__':
    main()
