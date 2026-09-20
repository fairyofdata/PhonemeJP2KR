"""HTML/CSS building blocks for the Streamlit UI.

Every function returns a string and does no Streamlit calls, so the markup
is testable without a running app. Any text that comes from the learner,
the ASR or the LLM is escaped here.
"""

import base64
import json
from html import escape

from .labels import describe_error, tag_label

# design tokens, mirrored in .streamlit/config.toml
INK = "#1d2433"
MUTED = "#687085"
LINE = "#e3e6ee"
ACCENT = "#3b5bdb"
# System fonts only: a webfont adds an external fetch that breaks the page
# when it is blocked or partially cached (observed: mangled glyphs in the
# 700-weight title while 400-weight body text rendered fine).
FONT = ('"Noto Sans JP", "Yu Gothic UI", "Hiragino Kaku Gothic ProN", "Meiryo", '
        '"Malgun Gothic", "Apple SD Gothic Neo", system-ui, sans-serif')
FONT_KR = ('"Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", '
           '"Yu Gothic UI", system-ui, sans-serif')

BAND_COLORS = {
    "typical_faithful": "#2b8a3e",
    "indeterminate": "#d08c00",
    "rare_for_faithful": "#d9480f",
}

GLOBAL_CSS = f"""
<style>
/* set on containers only: icon spans declare their own font and must keep it */
.stApp, .stApp button, .stApp input, .stApp textarea {{ font-family: {FONT}; }}
/* Streamlit's own stylesheet sets Source Sans (no CJK) on markdown content,
   and inheritance loses to it — so name the custom elements explicitly */
.stApp [class^="pc-"], .stApp [class*=" pc-"] {{ font-family: {FONT}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp td, .stApp th,
.stApp [data-testid="stCaptionContainer"] {{ font-family: {FONT}; }}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.pc-header {{ display: flex; align-items: flex-end; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap; margin-bottom: 1.2rem; }}
.pc-title {{ font-size: 1.65rem; font-weight: 700; color: {INK}; line-height: 1.2; }}
.pc-sub {{ color: {MUTED}; font-size: 0.92rem; margin-top: 0.25rem; }}
.pc-chips {{ display: flex; gap: 0.4rem; flex-wrap: wrap; }}
.pc-chip {{ font-size: 0.75rem; color: {MUTED}; border: 1px solid {LINE};
  border-radius: 999px; padding: 0.15rem 0.6rem; background: #fff; }}
.pc-eyebrow {{ font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: {MUTED}; margin-bottom: 0.35rem; }}
.pc-target-kr {{ font-family: {FONT_KR}; font-size: 1.9rem;
  font-weight: 700; color: {INK}; line-height: 1.3; word-break: keep-all; }}
.pc-target-meta {{ display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 0.8rem;
  margin-top: 0.6rem; font-size: 0.9rem; }}
.pc-target-meta dt {{ color: {MUTED}; }}
.pc-target-meta dd {{ margin: 0; color: {INK}; font-family: {FONT_KR}; }}
.pc-ipa {{ font-family: 'Charis SIL', 'Doulos SIL', 'Noto Sans', serif; }}

.pc-hero {{ border: 1px solid {LINE}; border-radius: 14px; background: #fff;
  padding: 1.3rem 1.5rem; display: grid; grid-template-columns: 1fr auto;
  gap: 0.6rem 1.5rem; align-items: center; }}
.pc-band {{ display: inline-flex; align-items: center; gap: 0.45rem; font-weight: 700;
  font-size: 1.15rem; color: var(--band); }}
.pc-band::before {{ content: ""; width: 0.7rem; height: 0.7rem; border-radius: 50%;
  background: var(--band); }}
.pc-band-text {{ color: {MUTED}; font-size: 0.9rem; margin-top: 0.3rem; line-height: 1.55; }}
.pc-score {{ text-align: right; }}
.pc-score .num {{ font-size: 3rem; font-weight: 700; color: {INK}; line-height: 1; }}
.pc-score .den {{ color: {MUTED}; font-size: 1rem; margin-left: 0.15rem; }}
.pc-delta {{ font-size: 0.85rem; margin-top: 0.35rem; color: {MUTED}; }}
.pc-delta b.up {{ color: #2b8a3e; }} .pc-delta b.down {{ color: #c92a2a; }}
.pc-meter {{ grid-column: 1 / -1; position: relative; height: 10px; border-radius: 6px;
  overflow: hidden; display: flex; margin-top: 0.4rem; }}
.pc-meter span {{ height: 100%; }}
.pc-meter-wrap {{ grid-column: 1 / -1; position: relative; }}
.pc-meter-marker {{ position: absolute; top: -4px; width: 3px; height: 18px;
  background: {INK}; border-radius: 2px; transform: translateX(-1px); }}
.pc-meter-scale {{ display: flex; justify-content: space-between; font-size: 0.72rem;
  color: {MUTED}; margin-top: 0.3rem; position: relative; height: 1rem; }}
.pc-meter-scale span {{ position: absolute; transform: translateX(-50%); }}

.pc-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 0.8rem; }}
.pc-card {{ border: 1px solid {LINE}; border-radius: 12px; background: #fff;
  padding: 0.9rem 1rem; }}
.pc-card .kr {{ font-family: {FONT_KR}; font-size: 1.25rem;
  font-weight: 600; color: {INK}; margin: 0.25rem 0; word-break: keep-all; }}
.pc-card .note {{ color: {MUTED}; font-size: 0.8rem; line-height: 1.5; }}

.pc-diff {{ display: flex; flex-wrap: wrap; gap: 0.45rem; padding: 0.4rem 0; }}
.pc-syl {{ display: inline-flex; gap: 2px; padding: 3px; border-radius: 10px;
  background: #eef0f6; align-items: flex-start; }}
.pc-syl.bad {{ background: #ffe3e3; }}
.pc-j {{ min-width: 2.1rem; text-align: center; border-radius: 8px; padding: 0.3rem 0.35rem;
  font-family: {FONT_KR}; font-size: 1.05rem; line-height: 1.15;
  border: 1px solid {LINE}; background: #fff; color: {INK}; }}
.pc-j small {{ display: block; font-size: 0.68rem; color: {MUTED}; }}
.pc-j.sub {{ background: #fff0f0; border-color: #ffc9c9; color: #c92a2a; }}
.pc-j.del {{ background: #f8f9fa; color: #adb5bd; text-decoration: line-through; }}
.pc-j.ins {{ background: #f3f0ff; border-color: #d0bfff; color: #6741d9; }}
.pc-legend {{ display: flex; gap: 1rem; font-size: 0.78rem; color: {MUTED}; flex-wrap: wrap; }}
.pc-legend i {{ display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 3px;
  margin-right: 0.3rem; vertical-align: -1px; border: 1px solid {LINE}; }}

.pc-errors {{ list-style: none; padding: 0; margin: 0.4rem 0 0; }}
.pc-errors li {{ display: flex; justify-content: space-between; gap: 1rem;
  border-bottom: 1px solid {LINE}; padding: 0.5rem 0; font-size: 0.92rem; }}
.pc-errors li:last-child {{ border-bottom: 0; }}
.pc-errors .ev {{ font-family: {FONT_KR}; color: {MUTED}; white-space: nowrap; }}

.pc-bars {{ display: flex; flex-direction: column; gap: 0.55rem; }}
.pc-bar-head {{ display: flex; justify-content: space-between; gap: 0.5rem;
  font-size: 0.86rem; margin-bottom: 0.25rem; }}
.pc-bar-head small {{ color: {MUTED}; font-size: 0.72rem; margin-left: 0.35rem; }}
.pc-bar {{ height: 8px; border-radius: 4px; background: #eef0f6; overflow: hidden; }}
.pc-bar span {{ display: block; height: 100%; background: {ACCENT}; border-radius: 4px; }}
.pc-bar-head .n {{ color: {MUTED}; }}
.pc-empty {{ color: {MUTED}; font-size: 0.9rem; padding: 0.6rem 0; }}
@media (max-width: 640px) {{
  .block-container {{ padding-top: 1.2rem; }}
  .pc-hero {{ grid-template-columns: 1fr; padding: 1.1rem; }}
  .pc-score {{ text-align: left; order: -1; }}
  .pc-target-kr {{ font-size: 1.5rem; }}
}}
</style>
"""


def header_html() -> str:
    chips = "".join(f'<span class="pc-chip">{c}</span>' for c in (
        "Whisper × Wav2Vec2", "標準発音法 G2P", "AI-Hub 615発話で検証済み"))
    return (
        '<div class="pc-header"><div>'
        '<div class="pc-title">韓国語 発音コーチ</div>'
        '<div class="pc-sub">日本語母語話者のための、音素レベルの発音分析とフィードバック</div>'
        f'</div><div class="pc-chips">{chips}</div></div>'
    )


def target_html(target: str, surface: str, ipa: str) -> str:
    return (
        '<div class="pc-eyebrow">発音ガイド</div>'
        f'<div class="pc-target-kr">{escape(target)}</div>'
        '<dl class="pc-target-meta">'
        f'<dt>標準発音</dt><dd>[{escape(surface)}]</dd>'
        f'<dt>IPA</dt><dd class="pc-ipa">/{escape(ipa)}/</dd>'
        '</dl>'
    )


def score_hero_html(score: int, band: dict, band_label: str, band_text: str,
                    ref: dict, previous: int = None) -> str:
    color = BAND_COLORS[band["band"]]
    p10, med = ref["p10"], ref["median"]
    meter = (
        f'<span style="width:{p10}%;background:#ffe8cc"></span>'
        f'<span style="width:{med - p10}%;background:#fff3bf"></span>'
        f'<span style="width:{100 - med}%;background:#d3f9d8"></span>'
    )
    delta = ""
    if previous is not None:
        diff = score - previous
        cls = "up" if diff > 0 else "down" if diff < 0 else ""
        delta = (f'<div class="pc-delta">この文の前回 {previous} 点から '
                 f'<b class="{cls}">{diff:+d}</b></div>')
    return (
        f'<div class="pc-hero" style="--band:{color}">'
        f'<div><div class="pc-band">{escape(band_label)}</div>'
        f'<div class="pc-band-text">参考範囲 {band["low"]}–{band["high"]} 点 · {escape(band_text)}</div></div>'
        f'<div class="pc-score"><span class="num">{score}</span><span class="den">/100</span>{delta}</div>'
        '<div class="pc-meter-wrap">'
        f'<div class="pc-meter">{meter}</div>'
        f'<div class="pc-meter-marker" style="left:{score}%"></div>'
        '<div class="pc-meter-scale">'
        f'<span style="left:0%;transform:none">0</span><span style="left:{p10}%">{p10}</span>'
        f'<span style="left:{med}%">{med}</span><span style="left:100%;transform:translateX(-100%)">100</span>'
        '</div></div></div>'
    )


def _jamo_chip(p) -> str:
    if p.op == "match":
        return f'<span class="pc-j">{escape(p.ref)}</span>'
    if p.op == "sub":
        return f'<span class="pc-j sub">{escape(p.hyp)}<small>{escape(p.ref)}</small></span>'
    if p.op == "del":
        return f'<span class="pc-j del">{escape(p.ref)}</span>'
    return f'<span class="pc-j ins">+{escape(p.hyp)}</span>'


def syllable_groups(pairs) -> list:
    """Split aligned pairs into target syllables (for display only).

    Onset ㅇ is absent from the jamo sequence, so on the target side a
    consonant followed by a vowel opens a syllable, and a vowel opens one
    when the current syllable already has a vowel. Insertions stay with
    the syllable they follow.
    """
    refs = [p.ref for p in pairs]
    groups, has_vowel = [], False
    for i, p in enumerate(pairs):
        opens = False
        if p.ref in _VOWEL_JAMO:
            opens = has_vowel or not groups
            has_vowel = True
        elif p.ref:
            nxt = next((r for r in refs[i + 1:] if r), "")
            if nxt in _VOWEL_JAMO:
                opens, has_vowel = True, False
        if opens or not groups:
            groups.append([])
        groups[-1].append(p)
    return groups


_VOWEL_JAMO = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


def diff_html(pairs) -> str:
    chips = []
    for group in syllable_groups(pairs):
        bad = any(p.op != "match" for p in group)
        chips.append(f'<span class="pc-syl{" bad" if bad else ""}">'
                     + "".join(_jamo_chip(p) for p in group) + "</span>")
    legend = (
        '<div class="pc-legend">'
        f'<span><i style="background:#fff"></i>一致</span>'
        f'<span><i style="background:#fff0f0;border-color:#ffc9c9"></i>置き換え（下段 = 本来の音）</span>'
        f'<span><i style="background:#f8f9fa"></i>脱落</span>'
        f'<span><i style="background:#f3f0ff;border-color:#d0bfff"></i>挿入</span>'
        '</div>'
    )
    return f'<div class="pc-diff">{"".join(chips)}</div>{legend}'


def error_list_html(error_tags) -> str:
    if not error_tags:
        return '<div class="pc-empty">音素レベルの不一致は検出されませんでした。</div>'
    items = []
    for err in error_tags:
        label, _, evidence = describe_error(err).partition(" · ")
        when = f' · {err["timestamp"]:.2f}s' if "timestamp" in err else ""
        items.append(f'<li><span>{escape(label)}</span>'
                     f'<span class="ev">{escape(evidence)}{when}</span></li>')
    return f'<ul class="pc-errors">{"".join(items)}</ul>'


def channel_cards_html(res: dict) -> str:
    katakana = (res.get("llm") or {}).get("katakana")
    cards = [
        ("お手本", res["target"], f'[{res["target_surface"]}]',
         "標準発音法の規則から決定論的に導いた発音"),
        ("聞き取り · Whisper", res["whisper_text"], f'/{res["whisper_ipa"]}/',
         "言語モデルが文脈で補正した認識 — 聞き手が意味を取れたかの参考"),
        ("音響 · Wav2Vec2", res["wav2vec_text"], f'/{res["actual_ipa"]}/',
         "補正なしの音響認識 — スコアはこの結果から算出"),
    ]
    if katakana:
        cards.append(("カタカナで見ると", katakana, "",
                      "実際の発音を日本語の音で表記（LLM による可視化）"))
    html = []
    for title, main, sub, note in cards:
        sub_html = f'<div class="note pc-ipa">{escape(sub)}</div>' if sub else ""
        html.append(f'<div class="pc-card"><div class="pc-eyebrow">{escape(title)}</div>'
                    f'<div class="kr">{escape(main or "—")}</div>{sub_html}'
                    f'<div class="note" style="margin-top:.45rem">{escape(note)}</div></div>')
    return f'<div class="pc-cards">{"".join(html)}</div>'


def weak_points_html(weak_points, drill_by_tag, limit: int = 6) -> str:
    if not weak_points:
        return '<div class="pc-empty">まだデータがありません。発音を分析すると、よく出る誤りがここに集計されます。</div>'
    top = weak_points[:limit]
    peak = max(n for _, n in top)
    rows = []
    for tag, n in top:
        drill = "<small>ドリルあり</small>" if tag in drill_by_tag else ""
        rows.append(
            f'<div><div class="pc-bar-head"><span>{escape(tag_label(tag))}{drill}</span>'
            f'<span class="n">{n}</span></div>'
            f'<div class="pc-bar"><span style="width:{100 * n / peak:.0f}%"></span></div></div>')
    return f'<div class="pc-bars">{"".join(rows)}</div>'


# --- waveform player (runs in a components iframe, so it carries its own CSS) --

def envelope(waveform, bins: int = 280) -> list:
    """Peak-amplitude envelope in [0, 1], one value per display bar."""
    import numpy as np  # app-only dependency; keeps the rest importable in CI

    wave = np.abs(np.asarray(waveform, dtype=float))
    if wave.size == 0:
        return []
    chunks = np.array_split(wave, min(bins, wave.size))
    peaks = np.array([c.max() for c in chunks])
    top = peaks.max() or 1.0
    return [round(float(v), 3) for v in peaks / top]


def audio_mime(audio_bytes: bytes) -> str:
    head = audio_bytes[:12]
    if head[:4] == b"RIFF":
        return "audio/wav"
    if head[:4] == b"fLaC":
        return "audio/flac"
    if head[:4] == b"OggS":
        return "audio/ogg"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "audio/webm"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    return "audio/wav"


AUDIO_SUFFIXES = {"audio/wav": ".wav", "audio/mpeg": ".mp3", "audio/flac": ".flac",
                  "audio/ogg": ".ogg", "audio/webm": ".webm"}


def audio_suffix(audio_bytes: bytes) -> str:
    """File extension matching the sniffed container, for stored clips."""
    return AUDIO_SUFFIXES.get(audio_mime(audio_bytes), ".wav")


def player_height(n_markers: int) -> int:
    return 150 + (46 if n_markers else 0)


def waveform_player_html(audio_bytes: bytes, peaks: list, duration: float,
                         error_tags) -> str:
    markers = [
        {"t": float(e["timestamp"]), "label": describe_error(e)}
        for e in error_tags if "timestamp" in e
    ]
    src = f"data:{audio_mime(audio_bytes)};base64,{base64.b64encode(audio_bytes).decode()}"
    data = json.dumps({"peaks": peaks, "duration": duration, "markers": markers},
                      ensure_ascii=False).replace("</", "<\\/")
    return f"""
<style>
  body {{ margin:0; font-family:{FONT}; color:{INK}; }}
  .wrap {{ border:1px solid {LINE}; border-radius:12px; padding:12px 14px; background:#fff; }}
  .row {{ display:flex; align-items:center; gap:12px; }}
  button.play {{ width:40px; height:40px; border-radius:50%; border:0; background:{ACCENT};
    color:#fff; font-size:15px; cursor:pointer; flex:none; }}
  canvas {{ flex:1; min-width:0; height:72px; cursor:pointer; display:block; }}
  .time {{ font-size:12px; color:{MUTED}; font-variant-numeric:tabular-nums; flex:none; width:78px; text-align:right; }}
  .marks {{ display:flex; gap:6px; flex-wrap:nowrap; overflow-x:auto; margin-top:10px; padding-bottom:4px; }}
  .mk {{ border:1px solid #ffc9c9; background:#fff5f5; color:#c92a2a; border-radius:999px;
    padding:4px 10px; font-size:12px; cursor:pointer; white-space:nowrap; font-family:inherit; }}
  .mk:hover {{ background:#ffe3e3; }}
  .hint {{ font-size:11px; color:{MUTED}; margin-top:6px; }}
</style>
<div class="wrap">
  <div class="row">
    <button class="play" id="play" aria-label="再生">▶</button>
    <canvas id="wave"></canvas>
    <span class="time" id="time">0.00 / 0.00s</span>
  </div>
  <div class="marks" id="marks"></div>
  <div class="hint" id="hint"></div>
</div>
<audio id="audio" src="{src}" preload="auto"></audio>
<script>
const D = {data};
const audio = document.getElementById('audio');
const canvas = document.getElementById('wave');
const ctx = canvas.getContext('2d');
const playBtn = document.getElementById('play');
const timeEl = document.getElementById('time');
let dur = D.duration || 0;
audio.addEventListener('loadedmetadata', () => {{ if (isFinite(audio.duration)) dur = audio.duration; draw(); }});

function draw() {{
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const n = D.peaks.length, gap = 1, bw = Math.max(1, w / n - gap);
  const played = dur ? audio.currentTime / dur : 0;
  D.peaks.forEach((p, i) => {{
    const x = i * (w / n), bh = Math.max(2, p * (h - 8));
    ctx.fillStyle = (i / n) < played ? '{ACCENT}' : '#c5cbe0';
    ctx.fillRect(x, (h - bh) / 2, bw, bh);
  }});
  D.markers.forEach(m => {{
    if (!dur) return;
    const x = (m.t / dur) * w;
    ctx.fillStyle = '#e03131'; ctx.fillRect(x - 1, 0, 2, h);
    ctx.beginPath(); ctx.arc(x, 4, 4, 0, 2 * Math.PI); ctx.fill();
  }});
  timeEl.textContent = audio.currentTime.toFixed(2) + ' / ' + (dur || 0).toFixed(2) + 's';
}}
function seek(t) {{ audio.currentTime = Math.max(0, t); audio.play(); }}
playBtn.onclick = () => audio.paused ? audio.play() : audio.pause();
audio.onplay = () => {{ playBtn.textContent = '❚❚'; tick(); }};
audio.onpause = audio.onended = () => {{ playBtn.textContent = '▶'; draw(); }};
function tick() {{ draw(); if (!audio.paused) requestAnimationFrame(tick); }}
canvas.onclick = e => {{ if (dur) seek((e.offsetX / canvas.clientWidth) * dur); }};
const marks = document.getElementById('marks');
D.markers.forEach(m => {{
  const b = document.createElement('button');
  b.className = 'mk'; b.textContent = m.t.toFixed(2) + 's  ' + m.label;
  b.onclick = () => seek(m.t - 0.1);
  marks.appendChild(b);
}});
document.getElementById('hint').textContent = D.markers.length
  ? '赤い線 = 誤りが検出された位置。ボタンでその直前から再生します。'
  : '波形をクリックすると、その位置から再生します。';
window.addEventListener('resize', draw);
draw();
</script>
"""
