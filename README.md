# Fly central complex: navigation and servo control from the MaleCNS connectome

This is the minimal package needed to reproduce two papers built on a rate model of the *Drosophila* central complex, extracted from the Janelia FlyEM **MaleCNS v1.0** connectome: 2,308 neurons and 187,137 signed connections, stored as a 1 MB Q8_0 GGUF file.

| | Paper | What it shows |
|---|---|---|
| A | [`paper/a_navigation/PAPER.md`](paper/a_navigation/PAPER.md) · [HTML](paper/a_navigation/PAPER.html) | **The navigation algorithm is in the wiring.** The heading ring is recovered from connectivity alone. The connection kernels contain the ring-attractor motifs and a ±73° shifted comparator. The model steers in closed loop. Quantization is pruning, and a 166-neuron / 932-synapse circuit still steers. |
| B | [`paper/b_servo_control/PAPER.md`](paper/b_servo_control/PAPER.md) · [HTML](paper/b_servo_control/PAPER.html) | **Heading servo or travel servo?** A control-theoretic reading of coordinate transformation: ρ as a disturbance-rejection ratio, the two bilinear operations a travel servo needs, and the PLL isomorphism. The plain circuit is a heading servo. §7 adds the two missing multiplications as a declared circuit and tests its performance, ablations and robustness. |

The HTML files are self-contained, with all figures and the typeset maths embedded.

**Interactive demos.** Each page also comes as one self-contained `*-standalone.html` file that opens with a double-click (see *Running the demos* below).

- [`web/minimal.html`](web/minimal.html) is **Minimal Brain**, a browser game built on paper A §5. You delete cell types, lower the bits per weight and prune synapses, and see whether the fly can still cross a maze.
- [`web/servo.html`](web/servo.html) is **Fly Servo Lab**, built on paper B §7. Two flies fly through the same sideslip: one runs the plain circuit, the other runs it with the added circuit (variant N or C). You can switch each added mechanism on and off and watch the PFN, hΔB and PFL3 activity.

## Requirements

- Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib` and `markdown` (`pip install -r requirements.txt`). Tested with Python 3.11, numpy 2.4, scipy 1.17, matplotlib 3.11 and markdown 3.11.
- Node.js ≥ 18, only for the JavaScript engine and the demos: `tests/test_cxnet.mjs`, `tests/test_minimal_demo.mjs`, `tests/test_servo_demo.mjs` and `scripts/audit_claims.mjs`.
- Optional: KaTeX 0.16.9 (`npm install katex@0.16.9`) with `KATEX_DIR=node_modules/katex/dist`, to embed the maths when re-rendering the HTML.

## Quick start

```sh
pip install -r requirements.txt
./reproduce.sh test        # all test suites, about a minute
./reproduce.sh figures     # re-measure paper A, rebuild number sheets, figures and HTML (~2 min)
git diff --stat            # data, number sheets and figures should not change
```

Larger re-runs:

```sh
./reproduce.sh analyses    # + control theory and the paper B section 7 experiment (~35 min)
./reproduce.sh all         # + every circuit measurement paper B cites (long; see below)
```

Every step overwrites the committed result it produces, so `git diff` shows exactly what a re-run changed. The package was checked this way. In a fresh copy, `servo_theory.py`, `fig_a.py`, `fig_b.py`, `hdelta_phase.py`, `pfn_basis.py` and `pfl3_inputs.py` reproduced their committed JSON outputs byte for byte, all 19 figures came out identical, and all test suites passed. The HTML files are byte-identical only when `KATEX_DIR` points to KaTeX 0.16.9, which the committed pages embed. Without it, the pages differ only in loading KaTeX from a CDN. `scripts/servo_augmented.py --quick` also ran to completion; the full §7 run was not repeated in the copy.

## Layout

```
model/flybrain-cx.gguf        the connectome model: CSR sparse matrix, Q8_0 weights, F32 phases
data/                         measurement outputs cited by the papers (see the table below)
scripts/                      numpy engine and every experiment script the papers rely on
web/gguf.js, web/cxnet.js     the dependency-free JavaScript engine (runs the same model)
web/minimal.html              the Minimal Brain demo (needs a local web server)
web/minimal-standalone.html   the same demo with engine and model inlined (double-click to open)
web/servo.html                the Fly Servo Lab demo (paper B section 7; needs a local web server)
web/servo_augmented.js        JavaScript port of the section 7 added circuit (scripts/servo_augmented.py)
web/servo_demo_data.json      calibrated N and C circuits and the reference task, for the port
web/servo-standalone.html     Fly Servo Lab with engine, model and data inlined (double-click to open)
paper/a_navigation/           paper A: PAPER.md, PAPER.html, figures/
paper/b_servo_control/        paper B: PAPER.md, PAPER.html, figures/
paper/data/                   number sheets: every value the papers quote, with its source
paper/scripts/                figure generators (fig_a.py, fig_b.py) and the HTML renderer
tests/                        consistency and regression tests
reproduce.sh                  one entry point for the steps above
```

## Running the demos

- **No setup:** open `web/minimal-standalone.html` or `web/servo-standalone.html` in any modern browser.
- **From a clone:** run `python3 -m http.server 8000` in the repository root and open <http://localhost:8000/web/minimal.html> or <http://localhost:8000/web/servo.html>. A server is needed because browsers block module scripts and `fetch` on `file://` pages.
- **Online:** if GitHub Pages is enabled for this repository (Settings → Pages → deploy from the `main` branch, root folder), the demos are served at `https://<user>.github.io/<repository>/web/minimal.html` and `.../web/servo.html`.

After changing `web/minimal.html` or the engine, rebuild the standalone file and run the demo test:

```sh
python3 scripts/bundle.py --page minimal.html -o web/minimal-standalone.html
node tests/test_minimal_demo.mjs
```

The test recomputes the numbers the page shows from the model: the intact size of 1,495.8 kbit, and 166 neurons / 932 synapses / 3.7 kbit for EPG + FC2 + PFL3 at 4 bit. It checks that this circuit steers correctly in 24 open-loop probes, that the page is in English, and that the standalone file matches a fresh rebuild. The game's pass rule (reach the goal within 60 s with at least 80% steering-sign accuracy) is looser than paper A's tests, so smaller brains than 3.7 kbit can pass in the game. The page says so.

**Fly Servo Lab** runs the added circuit of paper B §7 through `web/servo_augmented.js`, a line-by-line JavaScript port of `scripts/servo_augmented.py`. The calibrated configuration of both variants comes from `web/servo_demo_data.json`, together with the alignment weights M and the 12 sideslip tracks of the reference task. `data/servo_augmented.json` does not store M, so `scripts/export_servo_demo.py` re-runs the deterministic calibration (about 10 minutes). It checks that the result is the evaluated configuration and then writes the file. After changing the page, the port or the data, rebuild and test:

```sh
python3 scripts/export_servo_demo.py          # only if the section 7 calibration changed
python3 scripts/bundle.py --page servo.html --module servo_augmented.js \
        --data servo_demo_data.json -o web/servo-standalone.html
node tests/test_servo_demo.mjs                # about 40 s
```

The test runs the reference task in JavaScript and requires its medians to match the Python results of paper B (42.6°, 17.7° and 14.8° for the plain circuit, N and C) within 1.5°, and every trial's median within 0.5°. In practice they agree to better than 0.01°. Switching off one mechanism at a time must reproduce the ablations of paper B §7 (for example 74.9° for variant C without the PFN multiplication), so the page's toggles are tested too. It also checks the configuration against `data/servo_augmented.json`, checks that every weight in M sits on an existing PFN → hΔB connection, and checks that with every added mechanism off the fly is exactly the plain circuit.

## Where every number comes from

**Paper A.** `paper/scripts/fig_a.py` re-measures everything from `model/flybrain-cx.gguf` and `data/cx_network.npz`: the ring embedding, the kernels, the requantization, the minimal-circuit table and a Python replication of the 48-trial closed-loop audit. It writes `paper/data/paper_a_measurements.json` and figures A0–A7. The JavaScript audit it is checked against is `data/claims_audit_maze.json`.

**Paper B.** `paper/scripts/fig_b.py` copies every cited value into `paper/data/paper_b_measurements.json` and draws figures B1–B11. B3 and B4 are drawn by `scripts/fig_servo_theory.py`. The values come from:

| Data file | Produced by | Used in |
|---|---|---|
| `data/cx_network.npz`, `data/cx_extract_report.json`, `model/flybrain-cx.gguf` | extraction from the raw MaleCNS tables (not included, see below) | A §3, everything |
| `data/hdelta_phase.json` | `scripts/hdelta_phase.py` | hΔB readout phases |
| `data/pfn_basis.json` | `scripts/pfn_basis.py` | B §4.1 |
| `data/pfn_operating.json` | `scripts/pfn_operating.py` | B §4.2 |
| `data/coord_transform.json` | `scripts/coord_transform.py` | B §4.2–4.3 |
| `data/shunting.json` | `scripts/shunting.py` | B §4.3 |
| `data/coord_localize.json` | `scripts/coord_localize.py` | B §4.3 |
| `data/pfn_model.json` | `scripts/pfn_model.py` | B §3.5, §4.3 |
| `data/pi_mechanism.json` | `scripts/mechanism_control.py` | B §4.4 |
| `data/pfl3_inputs.json` | `scripts/pfl3_inputs.py` | B §5.1 |
| `data/servo_confound.json` | `scripts/servo_confound.py` | A §6, B §5.2 |
| `data/servo_identify.json` | `scripts/servo_identify.py` | B §5.2 |
| `data/servo_ideal_hdb.json` | `scripts/servo_ideal_hdb.py` | B §5.2 |
| `data/bearing_vs_homing.json` | `scripts/bearing_vs_homing.py` | B §6 |
| `data/servo_theory.json` | `scripts/servo_theory.py` | B §3, §6 |
| `data/servo_augmented.json` | `scripts/servo_augmented.py` | B §7 |
| `data/claims_audit_maze.json` | `scripts/audit_claims.mjs` (JavaScript engine) | A §4–5 |
| `data/claims_audit_wiring.json` | `scripts/audit_claims.py` | A §6 |
| `data/pi_circuit.json` | reference values for `tests/test_pi.py` | tests only |

`./reproduce.sh all` runs these scripts in dependency order. The heavier measurements (`coord_transform.py`, `shunting.py`, `mechanism_control.py` and `audit_claims.mjs`) each take from several minutes to about an hour. They were not re-timed for this package.

## Tests

| Test | Checks |
|---|---|
| `tests/test_papers_en.py` | Every key number in both papers matches its data file or number sheet. The Python engine reproduces the JavaScript closed-loop audit. Every figure link resolves and every figure is used. The text is English only. Paper B cites no number from the unreproduced external model. |
| `tests/test_servo_augmented.py` | The added circuit of B §7: bit-identical dynamics with every mechanism off, M ≥ 0 only on existing PFN → hΔB pairs, the comparator's sign and its reversal, the delay line, multiplicative vs additive gating, and that the saved results follow the calibration rule. |
| `tests/test_servo_theory.py` | The control theory of B §3 and §6: (1 − ρ)φ, stability, the phasor identity, the squaring comparator, the PLL lag and cycle slips, world-fixed wind, and bit-for-bit reproduction of the task. |
| `tests/test_pi.py` | Rate-model invariants, including that shunting inhibition at f = 0 is bit-identical to the base model (B §4.3). |
| `tests/test_cxnet.mjs` | The JavaScript engine and GGUF reader. |
| `tests/test_minimal_demo.mjs` | The Minimal Brain demo: the numbers it displays, the record circuit's steering, English text, and an up-to-date standalone file. |
| `tests/test_servo_demo.mjs` | The Fly Servo Lab demo: the exported configuration equals the one paper B evaluated, M sits on existing connections, mechanisms off = plain circuit, the JavaScript port reproduces the reference-task medians and the single-mechanism ablations, English text, and an up-to-date standalone file. |

## What is not included, and why

- **Extraction from the raw connectome.** `model/flybrain-cx.gguf` and `data/cx_network.npz` were built from the 1.05 GB MaleCNS v1.0 connection and annotation tables (Janelia FlyEM, CC-BY 4.0, <https://male-cns.janelia.org/>). The extraction and GGUF build scripts need those tables and are not part of this package. The derived files are included, so every step from the model onward is reproducible here.
- **The other browser demos** (the maze game and the 3D viewer) and the Japanese-language project documents referenced in some script comments. Neither is needed to reproduce the papers. The Minimal Brain and Fly Servo Lab demos are included.
- **The Medium articles** that accompany the papers.

## Notes

- Several experiment scripts were written earlier in the project, and their comments and console messages are in Japanese. Their inputs, outputs and command lines are described above, and nothing about running them depends on the comments.
- `data/servo_theory.json` also contains an entry for an externally reported model that this project did not reproduce. Paper B does not use it, and the tests check this.
- Everything the model contains beyond the synaptic weights, such as the activation function, thresholds, time constants, input sites and readouts, is a modelling assumption. The papers mark which quantities come from the connectome and which do not.

## Data licence and citation

The connectome data are from **Janelia FlyEM MaleCNS v1.0** (CC-BY 4.0). Please cite the MaleCNS release when using the model file or anything derived from it. The papers list the other references they build on.
