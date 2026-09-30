#!/usr/bin/env python3
"""docs/EXPLAINER.md を、画像を埋め込んだ 1 枚の HTML に変換する。

Markdown のままだと画像パスが相対参照なので、共有しづらい。
base64 で埋め込んで自己完結にする。
"""
from __future__ import annotations
import argparse, base64, mimetypes, os, re
import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSS = """
:root{--ink:#1b2027;--dim:#5b6470;--line:#e3e7ec;--bg:#fbfbfc;--acc:#b45309;--acc2:#0e7490}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.85 -apple-system,BlinkMacSystemFont,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:860px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:30px;line-height:1.35;margin:0 0 8px;letter-spacing:.01em}
h2{font-size:22px;margin:56px 0 14px;padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:17px;margin:32px 0 10px;color:var(--acc)}
p{margin:14px 0}
img{max-width:100%;display:block;margin:22px auto;border:1px solid var(--line);
    border-radius:8px;background:white}
table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14.5px}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#f1f3f5;font-weight:650}
td:nth-child(n+2){font-variant-numeric:tabular-nums}
code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:13.5px}
pre{background:#1b2027;color:#e6edf3;padding:14px 16px;border-radius:8px;overflow-x:auto}
pre code{background:none;color:inherit;padding:0}
blockquote{margin:20px 0;padding:12px 18px;border-left:4px solid var(--acc2);
  background:#f0f7fa;border-radius:0 6px 6px 0}
blockquote p{margin:6px 0}
a{color:var(--acc2)}
hr{border:none;border-top:1px solid var(--line);margin:44px 0}
ul,ol{padding-left:1.4em}
li{margin:6px 0}
@media (prefers-color-scheme:dark){
  :root{--ink:#e6edf3;--dim:#98a2ad;--line:#2a3038;--bg:#0f1216;--acc:#f0a35e;--acc2:#5ec8f0}
  th{background:#1b2027} code{background:#1b2027} blockquote{background:#141a20}
  img{background:#fff}
}
"""

KATEX_DIR = os.environ.get("KATEX_DIR", "")
KATEX_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist"


def render_katex(maths):
    """数式を node の katex で HTML に前もって描く。使えなければ None（CDN の自動描画に任せる）。

    KATEX_DIR に katex の dist を指すと、CSS とフォントも埋め込んで自己完結の HTML になる。
    """
    if not KATEX_DIR or not os.path.exists(os.path.join(KATEX_DIR, "katex.js")):
        return None
    import json, subprocess
    js = ("const k=require(process.argv[1]);let s='';process.stdin.on('data',d=>s+=d);"
          "process.stdin.on('end',()=>{const a=JSON.parse(s);"
          "process.stdout.write(JSON.stringify(a.map(x=>k.renderToString(x,{displayMode:true,throwOnError:true}))))})")
    src = [m[2:-2].strip() for m in maths]
    out = subprocess.run(["node", "-e", js, os.path.join(KATEX_DIR, "katex.js")],
                         input=json.dumps(src), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def katex_head(inline):
    style = '<style>.math{overflow-x:auto;margin:18px 0}</style>'
    if not inline:
        return (f'<link rel="stylesheet" href="{KATEX_CDN}/katex.min.css">'
                f'<script defer src="{KATEX_CDN}/katex.min.js"></script>'
                f'<script defer src="{KATEX_CDN}/contrib/auto-render.min.js" '
                'onload="renderMathInElement(document.body,{delimiters:'
                "[{left:'$$',right:'$$',display:true}]})\"></script>" + style)
    css = open(os.path.join(KATEX_DIR, "katex.min.css"), encoding="utf-8").read()

    def font(m):
        name = m.group(1)
        path = os.path.join(KATEX_DIR, "fonts", name)
        b64 = base64.b64encode(open(path, "rb").read()).decode("ascii")
        return f"url(data:font/woff2;base64,{b64}) format(\"woff2\")"

    css = re.sub(r'url\(fonts/([^)]+\.woff2)\) format\("woff2"\)', font, css)
    css = re.sub(r',url\(fonts/[^)]+\.(?:woff|ttf)\) format\("(?:woff|truetype)"\)', '', css)
    return f'<style>{css}</style>' + style


def embed(md_path: str, lang: str = "ja") -> str:
    base = os.path.dirname(os.path.abspath(md_path))
    src = open(md_path, encoding="utf-8").read()
    title = src.splitlines()[0].lstrip("# ").strip()

    def repl(m):
        alt, path = m.group(1), m.group(2)
        full = os.path.join(base, path)
        if not os.path.exists(full):
            return m.group(0)
        mime = mimetypes.guess_type(full)[0] or "image/png"
        with open(full, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f'![{alt}](data:{mime};base64,{b64})'

    src = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl, src)

    # 表示数式 $$...$$ は Markdown に通すと _ や \, が壊れるので、先に退避して後で戻す。
    # 数式が無い文書（従来の文書すべて）では何も変わらない。
    maths = []

    def stash(m):
        maths.append(m.group(0))
        return f'<p>@@MATH{len(maths) - 1}@@</p>'

    src = re.sub(r'\$\$.+?\$\$', stash, src, flags=re.S)
    body = markdown.markdown(src, extensions=["tables", "fenced_code", "toc"])
    head_extra = ''
    if maths:
        rendered = render_katex(maths)
        for i, m in enumerate(maths):
            if rendered is not None:
                html = f'<div class="math">{rendered[i]}</div>'
            else:
                esc = m.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html = f'<div class="math">{esc}</div>'
            body = body.replace(f'<p>@@MATH{i}@@</p>', html)
        head_extra = katex_head(rendered is not None)
    return (f'<!DOCTYPE html>\n<html lang="{lang}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><style>{CSS}</style>{head_extra}</head>'
            f'<body><main>{body}</main></body></html>')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=os.path.join(ROOT, "docs", "EXPLAINER.md"))
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "docs", "EXPLAINER.html"))
    ap.add_argument("--lang", default="ja")
    a = ap.parse_args()
    html = embed(a.md, a.lang)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"{a.out}  ({os.path.getsize(a.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
