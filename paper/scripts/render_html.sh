#!/bin/sh
# Render paper/*/PAPER.md to self-contained HTML (images embedded).
# Set KATEX_DIR to a katex dist directory (npm install katex) to embed the maths;
# otherwise the page loads KaTeX from a CDN when opened.
set -e
cd "$(dirname "$0")/../.."
for p in a_navigation b_servo_control; do
  python3 scripts/render_doc.py --md paper/$p/PAPER.md -o paper/$p/PAPER.html --lang en
done
