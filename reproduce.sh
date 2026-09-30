#!/bin/sh
# Reproduce paper A and paper B from this package.
#
#   ./reproduce.sh test       run the test suites (seconds)
#   ./reproduce.sh figures    re-measure paper A, rebuild both number sheets, figures and HTML (~2 min)
#   ./reproduce.sh analyses   also re-run the control theory and the paper B section 7 experiment (~35 min)
#   ./reproduce.sh all        also re-run every circuit measurement paper B cites (long; see README)
#
# Every step writes into data/ or paper/, overwriting the committed results, so `git diff`
# shows exactly what changed. Set KATEX_DIR to a KaTeX dist/ directory to embed the maths
# in the HTML; otherwise the pages load KaTeX from a CDN.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
step() { echo; echo "=== $*"; }

measurements() {
  # order matters: later scripts read the JSON written by earlier ones
  for s in hdelta_phase pfn_basis pfn_operating coord_transform shunting coord_localize pfn_model \
           pfl3_inputs servo_confound servo_identify servo_ideal_hdb bearing_vs_homing \
           mechanism_control audit_claims; do
    step "scripts/$s.py"; $PY scripts/$s.py
  done
  step "scripts/audit_claims.mjs (JavaScript engine)"; node scripts/audit_claims.mjs
}

analyses() {
  step "scripts/servo_theory.py";    $PY scripts/servo_theory.py
  step "scripts/servo_augmented.py"; $PY scripts/servo_augmented.py
}

figures() {
  step "paper/scripts/fig_a.py"; $PY paper/scripts/fig_a.py
  step "paper/scripts/fig_b.py"; $PY paper/scripts/fig_b.py
  step "paper/scripts/render_html.sh"; sh paper/scripts/render_html.sh
}

tests() {
  for t in tests/test_*.py; do step "$t"; $PY "$t"; done
  step "tests/test_cxnet.mjs"; node tests/test_cxnet.mjs
}

case "${1:-figures}" in
  test)     tests ;;
  figures)  figures; tests ;;
  analyses) analyses; figures; tests ;;
  all)      measurements; analyses; figures; tests ;;
  *) echo "usage: $0 [test|figures|analyses|all]"; exit 2 ;;
esac
