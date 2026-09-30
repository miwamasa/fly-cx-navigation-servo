"""Shared helpers for the English figures in paper/.

Everything here reads data that already lives in the repository
(model/flybrain-cx.gguf, data/cx_network.npz, data/*.json). Nothing is
fetched, and no connectome weight is changed on disk.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
SCRIPTS = os.path.join(ROOT, 'scripts')
sys.path.insert(0, SCRIPTS)

MODEL = os.path.join(ROOT, 'model', 'flybrain-cx.gguf')
NPZ = os.path.join(ROOT, 'data', 'cx_network.npz')
TAU = 2 * np.pi

# One palette for every figure: blue = heading, orange = goal / travel,
# purple = third series, red only for "broken / unstable" states.
INK, DIM, GRID = '#1b2028', '#6d7681', '#e3e7ec'
BLUE, ORANGE, PURPLE, GREEN = '#2f88b0', '#c26a1a', '#7a5fb5', '#2e8b57'
RED = '#c94f4f'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#9aa3ad',
    'axes.labelcolor': INK,
    'xtick.color': '#4b5563',
    'ytick.color': '#4b5563',
    'font.size': 9.5,
})


def load_json(name):
    with open(os.path.join(ROOT, 'data', name), encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, name):
    path = os.path.join(PAPER, 'data', name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f'-> paper/data/{name}')


def save(fig, paper, name):
    path = os.path.join(PAPER, paper, 'figures', f'{name}.png')
    fig.savefig(path, dpi=150, facecolor='white')
    plt.close(fig)
    print(f'-> paper/{paper}/figures/{name}.png')


def panel_title(ax, s):
    ax.set_title(s, loc='left', fontsize=11, weight='bold', color=INK)


def note(ax, s, y=-0.18):
    ax.text(0.0, y, s, transform=ax.transAxes, fontsize=8, color=DIM, va='top')


def wrap(x):
    return (x + np.pi) % TAU - np.pi


def load_npz():
    d = np.load(NPZ, allow_pickle=True)
    return {
        'names': np.array([str(x) for x in d['names']]),
        'types': np.array([str(x) for x in d['types']]),
        'phase': d['phase'].astype(np.float64),
        'side': d['side'], 'pb_side': d['pb_side'],
        'pre': d['pre'], 'post': d['post'], 'weight': d['weight'].astype(np.float64),
        'n': len(d['types']),
    }


def dense(D, weight=None):
    """Dense post x pre matrix of signed synapse counts (or the given weights)."""
    W = np.zeros((D['n'], D['n']))
    W[D['post'], D['pre']] = D['weight'] if weight is None else weight
    return W


def select(types, *prefixes, exact=False):
    if exact:
        return np.array([i for i, t in enumerate(types) if t in prefixes], dtype=int)
    return np.array([i for i, t in enumerate(types)
                     if any(t.startswith(p) for p in prefixes)], dtype=int)


def requantize(W_csr, bits):
    """Row-wise requantization, identical to CXNetwork.applyWeightBudget in web/cxnet.js.

    Each row (one postsynaptic cell) is scaled by its largest |w|, rounded to
    2^(bits-1)-1 levels and scaled back. bits >= 8 leaves the Q8_0 weights as they are.
    """
    W = W_csr.copy().tocsr()
    if bits >= 8:
        return W
    lv = (1 << (bits - 1)) - 1
    for i in range(W.shape[0]):
        a, b = W.indptr[i], W.indptr[i + 1]
        if a == b:
            continue
        m = np.abs(W.data[a:b]).max()
        if m > 0:
            W.data[a:b] = np.round(W.data[a:b] / m * lv) / lv * m
    return W


def phase_kernel(W, phase, pre, post, bins=16):
    """Mean weight per (post - pre) phase-difference bin, over all pre/post pairs.

    Pairs without a connection count as zero, so the curve is a density of
    connection strength as a function of phase difference.
    """
    pre = pre[phase[pre] > -8]
    post = post[phase[post] > -8]
    sub = W[np.ix_(post, pre)]
    diff = wrap(phase[post][:, None] - phase[pre][None, :])
    k = np.floor((diff + np.pi) / TAU * bins).astype(int) % bins
    acc = np.bincount(k.ravel(), weights=sub.ravel(), minlength=bins)
    cnt = np.bincount(k.ravel(), minlength=bins)
    centers = -180 + (np.arange(bins) + 0.5) * 360 / bins
    return centers, np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
