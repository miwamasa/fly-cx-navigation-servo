#!/usr/bin/env python3
"""web/ の一式と .gguf を 1 つの HTML にまとめる。

ES モジュールは file:// では読めず、fetch も CORS で弾かれるため、
ダブルクリックで開ける版が欲しい場合はこのバンドルを使う。
GGUF は base64 で埋め込む（1.04 MB → 約 1.4 MB）。
"""

from __future__ import annotations

import argparse
import base64
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strip_module_syntax(src: str) -> str:
    """import 文を落とし、export を外して素のスクリプトにする。"""
    src = re.sub(r"^\s*import\s+[^;]+;\s*$", "", src, flags=re.M)
    src = re.sub(r"^\s*export\s+default\s+\w+;\s*$", "", src, flags=re.M)
    src = re.sub(r"^\s*export\s*\{[^}]*\}\s*;\s*$", "", src, flags=re.M)
    src = re.sub(r"^\s*export\s+(const|class|function)\b", r"\1", src, flags=re.M)
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="index.html", help="web/ 以下のページ名")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--model", default=os.path.join(ROOT, "model", "flybrain-cx.gguf"))
    args = ap.parse_args()

    web = os.path.join(ROOT, "web")
    if args.out is None:
        stem = os.path.splitext(args.page)[0]
        name = "flybrain-maze" if stem == "index" else f"flybrain-{stem}"
        args.out = os.path.join(ROOT, f"{name}-standalone.html")
    html = open(os.path.join(web, args.page), encoding="utf-8").read()
    gguf_js = strip_module_syntax(open(os.path.join(web, "gguf.js"), encoding="utf-8").read())
    cxnet_js = strip_module_syntax(open(os.path.join(web, "cxnet.js"), encoding="utf-8").read())

    with open(args.model, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    # 元の <script type="module"> を取り出して素のスクリプトに変換する
    m = re.search(r'<script type="module">(.*?)</script>', html, re.S)
    if not m:
        raise SystemExit(f"web/{args.page} に <script type=\"module\"> が見つかりません")
    app = strip_module_syntax(m.group(1))
    app = app.replace(
        "gguf = await GGUF.fromURL('../model/flybrain-cx.gguf');",
        "gguf = GGUF.fromArrayBuffer(__ggufBytes().buffer);",
    )

    loader = """
const __GGUF_B64 = "%s";
function __ggufBytes(){
  const bin = atob(__GGUF_B64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}
""" % b64

    bundled = html[: m.start()] + "<script>\n" + loader + gguf_js + "\n" + cxnet_js \
        + "\n" + app + "\n</script>" + html[m.end():]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(bundled)
    print(f"{args.out}  ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
