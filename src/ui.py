"""Presentation layer for the Streamlit app: palettes, CSS, HTML components and Plotly styling.

Keeping all visual code here leaves app.py focused on application logic. Every colour comes from a palette that
matches one of the two native Streamlit themes defined in .streamlit/config.toml, so custom HTML components and
native widgets always look like one design system - in both light and dark mode.
"""

# Type hints for readable function signatures
from __future__ import annotations

# HTML escaping for any text that is placed inside generated markup
from html import escape

# Whitespace normalisation for the HTML renderer
import re

# Web app framework
import streamlit as st

# ---------------------------------------------------------------------------------------------
# Palettes (must stay in sync with .streamlit/config.toml)
# ---------------------------------------------------------------------------------------------
PALETTES = {
    "dark": {
        "bg": "#0B1120",
        "surface": "#111A2E",
        "surface2": "#16213A",
        "border": "rgba(148,163,184,0.18)",
        "text": "#E6EAF2",
        "muted": "#93A0B8",
        "accent": "#E3B25A",
        "accent2": "#F4D58D",
        "accent_text": "#1A1405",
        "glow": "rgba(227,178,90,0.16)",
        "good": "#34D399",
        "bad": "#FB7185",
        "info": "#60A5FA",
        "violet": "#A78BFA",
        "orange": "#F59E0B",
        "hero_a": "#14203D",
        "hero_b": "#0B1120",
        "shadow": "0 18px 50px rgba(0,0,0,0.45)",
        "grid": "rgba(148,163,184,0.14)",
        "track": "rgba(148,163,184,0.18)",
        "heat": [[0, "#14203B"], [0.5, "#3B5B93"], [1, "#E3B25A"]],
        "colorway": [
            "#E3B25A",
            "#34D399",
            "#60A5FA",
            "#A78BFA",
            "#FB7185",
            "#F59E0B",
            "#2DD4BF",
            "#F472B6",
        ],
    },
    "light": {
        "bg": "#F7F5F0",
        "surface": "#FFFFFF",
        "surface2": "#FBF9F4",
        "border": "rgba(15,23,42,0.10)",
        "text": "#0F172A",
        "muted": "#5B6578",
        "accent": "#B7791F",
        "accent2": "#D9A441",
        "accent_text": "#FFFFFF",
        "glow": "rgba(183,121,31,0.14)",
        "good": "#059669",
        "bad": "#E11D48",
        "info": "#2563EB",
        "violet": "#7C3AED",
        "orange": "#D97706",
        "hero_a": "#FFFFFF",
        "hero_b": "#F3EEE2",
        "shadow": "0 18px 44px rgba(60,45,10,0.10)",
        "grid": "rgba(15,23,42,0.08)",
        "track": "rgba(15,23,42,0.10)",
        "heat": [[0, "#F6EBD3"], [0.5, "#E0B565"], [1, "#8A5A0B"]],
        "colorway": [
            "#B7791F",
            "#059669",
            "#2563EB",
            "#7C3AED",
            "#E11D48",
            "#D97706",
            "#0D9488",
            "#DB2777",
        ],
    },
}


def get_theme() -> str:
    """Return 'dark' or 'light' for the active Streamlit theme (defaults to dark when unknown)."""
    try:
        # Streamlit reports the resolved theme of the viewer's browser session
        value = st.context.theme.type
    except Exception:  # older Streamlit versions or bare mode
        value = None
    # Fall back to the dark palette, which is the flagship look
    return value if value in PALETTES else "dark"


def palette(theme: str | None = None) -> dict:
    """Palette for the given (or active) theme."""
    return PALETTES[theme or get_theme()]


# ---------------------------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------------------------
def inject_css(theme: str | None = None) -> None:
    """Inject the design-system CSS (variables + component styles + polish for native widgets)."""
    p = palette(theme)
    css = f"""
<style>
:root {{
  --bg:{p['bg']}; --surface:{p['surface']}; --surface2:{p['surface2']}; --border:{p['border']};
  --text:{p['text']}; --muted:{p['muted']}; --accent:{p['accent']}; --accent2:{p['accent2']};
  --accent-text:{p['accent_text']}; --glow:{p['glow']}; --good:{p['good']}; --bad:{p['bad']};
  --info:{p['info']}; --violet:{p['violet']}; --orange:{p['orange']}; --shadow:{p['shadow']};
  --track:{p['track']}; --hero-a:{p['hero_a']}; --hero-b:{p['hero_b']};
  --display:"Playfair Display", Georgia, "Times New Roman", serif;
}}

/* ---------- page chrome ---------- */
.stApp {{ background:
    radial-gradient(1100px 520px at 88% -8%, var(--glow), transparent 60%),
    radial-gradient(900px 480px at -6% 4%, color-mix(in srgb, var(--info) 9%, transparent), transparent 60%),
    var(--bg); }}
[data-testid="stHeader"] {{ background: transparent; }}
.stAppDeployButton, [data-testid="stDecoration"] {{ display:none !important; }}
.stMainBlockContainer {{ max-width: 1260px; padding-top: 3.2rem; padding-bottom: 3rem; }}
::selection {{ background: var(--accent); color: var(--accent-text); }}

/* ---------- hero ---------- */
.hero {{ position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between; gap:2rem;
  border:1px solid var(--border); border-radius:28px; padding:2.4rem 2.7rem; box-shadow: var(--shadow);
  background: radial-gradient(700px 260px at 92% 0%, var(--glow), transparent 65%),
              linear-gradient(135deg, var(--hero-a), var(--hero-b)); animation: rise .7s ease both; }}
.hero .eyebrow {{ font-size:.72rem; letter-spacing:.24em; text-transform:uppercase; font-weight:700; color:var(--accent); }}
.hero .title {{ font-family:var(--display); font-weight:800; font-size:clamp(2rem,4.2vw,3.25rem); line-height:1.06;
  margin:.55rem 0 .75rem; color:var(--text); letter-spacing:-.01em; }}
.hero .title .grad {{ background:linear-gradient(90deg,var(--accent),var(--accent2)); -webkit-background-clip:text;
  background-clip:text; color:transparent; }}
.hero .lede {{ color:var(--muted); font-size:1.04rem; max-width:38rem; line-height:1.55; margin:0; }}
.hero .art {{ flex:0 0 auto; width:300px; max-width:34%; opacity:.98; }}
.chips {{ display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.25rem; }}
.chip {{ display:inline-flex; align-items:center; gap:.4rem; padding:.4rem .85rem; font-size:.78rem; font-weight:500;
  border:1px solid var(--border); border-radius:999px; background: color-mix(in srgb, var(--surface) 70%, transparent);
  color:var(--text); backdrop-filter: blur(6px); }}
.chip b {{ color:var(--accent); font-weight:700; }}
@media (max-width: 900px) {{ .hero {{ padding:1.6rem 1.4rem; }} .hero .art {{ display:none; }} }}

/* ---------- notice bar ---------- */
.notice {{ display:flex; gap:.7rem; align-items:flex-start; margin:1rem 0 .4rem; padding:.7rem 1rem; font-size:.84rem;
  color:var(--muted); border:1px dashed color-mix(in srgb, var(--orange) 55%, transparent); border-radius:14px;
  background: color-mix(in srgb, var(--orange) 7%, transparent); }}
.notice b {{ color:var(--text); }}

/* ---------- section headers ---------- */
.section-label {{ font-size:.7rem; letter-spacing:.2em; text-transform:uppercase; font-weight:700; color:var(--accent); margin:0 0 .25rem; }}
.section-title {{ font-family:var(--display); font-weight:700; font-size:1.55rem; line-height:1.2; margin:0 0 .2rem; color:var(--text); }}
.section-sub {{ color:var(--muted); font-size:.92rem; margin:0 0 .9rem; line-height:1.5; }}
.mini-label {{ font-size:.7rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; color:var(--muted); margin:1rem 0 .15rem; }}

/* ---------- KPI cards ---------- */
.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.9rem; margin:.4rem 0 1rem; }}
.kpi {{ position:relative; overflow:hidden; border:1px solid var(--border); border-radius:18px; padding:1rem 1.1rem 1.05rem;
  background:var(--surface); transition: transform .2s ease, border-color .2s ease; animation: rise .6s ease both; }}
.kpi:hover {{ transform: translateY(-2px); border-color: color-mix(in srgb, var(--accent) 55%, var(--border)); }}
.kpi::before {{ content:""; position:absolute; left:0; top:14%; bottom:14%; width:3px; border-radius:3px; background:var(--accent); }}
.kpi .label {{ font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; font-weight:600; color:var(--muted); }}
.kpi .value {{ font-family:var(--display); font-weight:700; font-size:1.8rem; line-height:1.15; margin:.25rem 0 .1rem; color:var(--text); }}
.kpi .sub {{ font-size:.8rem; color:var(--muted); }}
.kpi .sub.up {{ color:var(--good); font-weight:600; }} .kpi .sub.down {{ color:var(--bad); font-weight:600; }}

/* ---------- verdict ---------- */
.verdict {{ display:flex; align-items:center; gap:1.5rem; padding:1.35rem 1.6rem; border-radius:24px; animation: rise .6s ease both;
  border:1px solid color-mix(in srgb, var(--v) 45%, var(--border));
  background: radial-gradient(420px 180px at 0% 0%, color-mix(in srgb, var(--v) 16%, transparent), transparent 70%), var(--surface); }}
.verdict.good {{ --v: var(--good); }} .verdict.bad {{ --v: var(--bad); }}
.verdict .ring {{ flex:0 0 auto; width:132px; height:132px; }}
.ring .track {{ stroke:var(--track); }} .ring .bar {{ stroke:var(--v); animation: ring 1.1s cubic-bezier(.2,.8,.2,1) both; }}
.ring text.pct {{ fill:var(--text); font-family:var(--display); font-weight:700; font-size:25px; }}
.ring text.cap {{ fill:var(--muted); font-size:8.5px; letter-spacing:.14em; text-transform:uppercase; }}
.verdict .tag {{ display:inline-flex; align-items:center; gap:.4rem; font-size:.68rem; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; color:var(--v); padding:.28rem .7rem; border-radius:999px; background:color-mix(in srgb, var(--v) 14%, transparent); }}
.verdict .headline {{ font-family:var(--display); font-weight:700; font-size:1.6rem; line-height:1.2; margin:.55rem 0 .3rem; color:var(--text); }}
.verdict .desc {{ color:var(--muted); font-size:.92rem; line-height:1.5; }}
@media (max-width: 640px) {{ .verdict {{ flex-direction:column; text-align:center; }} }}

/* ---------- factor list ---------- */
.factor {{ display:flex; align-items:center; gap:.8rem; padding:.7rem .9rem; margin-bottom:.5rem; font-size:.9rem;
  border:1px solid var(--border); border-radius:14px; background:var(--surface); color:var(--text); }}
.factor .dot {{ flex:0 0 auto; width:24px; height:24px; border-radius:50%; display:grid; place-items:center; font-size:.78rem; font-weight:800; }}
.factor.ok .dot {{ background:color-mix(in srgb, var(--good) 18%, transparent); color:var(--good); }}
.factor.no .dot {{ background:color-mix(in srgb, var(--muted) 18%, transparent); color:var(--muted); }}
.factor small {{ display:block; color:var(--muted); font-size:.76rem; margin-top:.1rem; }}

/* ---------- callouts / pills ---------- */
.callout {{ border:1px solid var(--border); border-left:3px solid var(--accent); border-radius:14px; background:var(--surface);
  padding:.9rem 1.1rem; font-size:.9rem; line-height:1.55; color:var(--text); margin:.4rem 0 .8rem; }}
.callout.warn {{ border-left-color:var(--orange); }} .callout.good {{ border-left-color:var(--good); }}
.callout b {{ color:var(--text); }} .callout .muted {{ color:var(--muted); }}
.pill {{ display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.72rem; font-weight:600;
  background:color-mix(in srgb, var(--accent) 14%, transparent); color:var(--accent); margin-right:.3rem; }}
.bench {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.7rem; margin:.3rem 0 .6rem; }}
.bench div {{ border:1px solid var(--border); border-radius:14px; background:var(--surface); padding:.7rem .9rem; }}
.bench span {{ display:block; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:600; }}
.bench b {{ font-family:var(--display); font-size:1.25rem; color:var(--text); }}
.bench em {{ font-style:normal; font-size:.78rem; color:var(--muted); }}

/* ---------- sidebar ---------- */
.brand {{ display:flex; align-items:center; gap:.75rem; margin:.2rem 0 .9rem; }}
.brand svg {{ width:42px; height:42px; flex:0 0 auto; }}
.brand .name {{ font-family:var(--display); font-weight:700; font-size:1.08rem; line-height:1.15; color:var(--text); }}
.brand .tagline {{ font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }}
.side-card {{ border:1px solid var(--border); border-radius:14px; padding:.75rem .9rem; margin:.5rem 0; background:color-mix(in srgb, var(--surface) 80%, transparent); font-size:.84rem; color:var(--text); }}
.side-card .k {{ font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:600; }}
.side-card .v {{ font-family:var(--display); font-size:1.15rem; font-weight:700; color:var(--accent); }}
.steps {{ counter-reset: step; list-style:none; padding:0; margin:.2rem 0; }}
.steps li {{ counter-increment: step; position:relative; padding:.35rem 0 .35rem 2rem; font-size:.84rem; color:var(--muted); }}
.steps li::before {{ content: counter(step); position:absolute; left:0; top:.3rem; width:1.35rem; height:1.35rem; border-radius:50%;
  display:grid; place-items:center; font-size:.7rem; font-weight:700; background:color-mix(in srgb, var(--accent) 16%, transparent); color:var(--accent); }}
.footer {{ text-align:center; color:var(--muted); font-size:.8rem; padding:1.5rem 0 .5rem; }}
.footer a {{ color:var(--accent); text-decoration:none; }}

/* ---------- native widget polish ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap:.3rem; border-bottom:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ height:auto; padding:.8rem 1.15rem; border-radius:12px 12px 0 0; font-weight:600; letter-spacing:.01em; }}
.stTabs [data-baseweb="tab"]:hover {{ background:color-mix(in srgb, var(--accent) 8%, transparent); }}
.stTabs [aria-selected="true"] {{ color:var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--accent) !important; height:3px; border-radius:3px; }}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {{ color:var(--accent-text) !important; font-weight:700; border:0;
  background:linear-gradient(135deg,var(--accent),var(--accent2)) !important; box-shadow:0 6px 20px color-mix(in srgb, var(--accent) 30%, transparent); }}
button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {{ color:var(--accent-text) !important; }}
.stButton button, .stDownloadButton button {{ transition: transform .15s ease, box-shadow .15s ease; }}
.stButton button:hover, .stDownloadButton button:hover {{ transform: translateY(-1px); }}
[data-testid="stMetric"] {{ background:var(--surface); }}
[data-testid="stExpander"] details {{ border-radius:16px; background:var(--surface); }}
[data-testid="stAlert"] {{ border-radius:14px; }}
[data-testid="stDataFrame"] {{ border-radius:16px; overflow:hidden; }}
[class*="st-key-panel"] {{ border-radius:22px !important; background:var(--surface); box-shadow:0 1px 0 color-mix(in srgb, var(--text) 4%, transparent); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ font-size:.88rem; }}
hr {{ border-color:var(--border) !important; }}

/* ---------- animation ---------- */
@keyframes rise {{ from {{ opacity:0; transform: translateY(10px); }} to {{ opacity:1; transform:none; }} }}
@keyframes ring {{ from {{ stroke-dasharray: 0 400; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
</style>
"""
    # Inject as raw HTML (no markdown processing needed inside <style>)
    st.markdown(css, unsafe_allow_html=True)


def render_html(markup: str) -> None:
    """Render custom HTML (including inline SVG).

    st.html() strips <svg>, so components go through st.markdown(unsafe_allow_html=True). The markup is flattened onto
    one line first: that keeps it a single raw-HTML block, so Markdown never re-interprets underscores, asterisks or
    indentation inside it.
    """
    # Collapse line breaks and indentation into single spaces
    flat = re.sub(r"\s*\n\s*", " ", markup.strip())
    # Render as raw HTML
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------
# SVG art
# ---------------------------------------------------------------------------------------------
def logo_svg() -> str:
    """Small brand mark: a gold rounded square with a house and a rising line."""
    return """
<svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F4D58D"/><stop offset="1" stop-color="#C98A1E"/></linearGradient></defs>
  <rect width="48" height="48" rx="13" fill="url(#lg)"/>
  <path d="M10 25 24 13l14 12" fill="none" stroke="#1A1405" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M15 24v12h18V24" fill="none" stroke="#1A1405" stroke-width="3" stroke-linejoin="round"/>
  <path d="M19 33l4-5 3 3 5-7" fill="none" stroke="#1A1405" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def hero_art() -> str:
    """Decorative skyline with a rising trend line for the hero banner."""
    return """
<svg viewBox="0 0 320 210" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="b1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#E3B25A" stop-opacity=".95"/><stop offset="1" stop-color="#E3B25A" stop-opacity=".12"/></linearGradient>
    <linearGradient id="b2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#60A5FA" stop-opacity=".75"/><stop offset="1" stop-color="#60A5FA" stop-opacity=".08"/></linearGradient>
    <linearGradient id="ln" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#34D399"/><stop offset="1" stop-color="#F4D58D"/></linearGradient>
  </defs>
  <g opacity=".95">
    <rect x="18"  y="110" width="34" height="90"  rx="5" fill="url(#b2)"/>
    <rect x="62"  y="78"  width="40" height="122" rx="5" fill="url(#b1)"/>
    <rect x="112" y="124" width="32" height="76"  rx="5" fill="url(#b2)"/>
    <rect x="154" y="52"  width="44" height="148" rx="5" fill="url(#b1)"/>
    <rect x="208" y="92"  width="34" height="108" rx="5" fill="url(#b2)"/>
    <rect x="252" y="66"  width="40" height="134" rx="5" fill="url(#b1)"/>
  </g>
  <g fill="#fff" opacity=".35">
    <rect x="72" y="92" width="6" height="6" rx="1"/><rect x="86" y="92" width="6" height="6" rx="1"/><rect x="72" y="108" width="6" height="6" rx="1"/><rect x="86" y="108" width="6" height="6" rx="1"/>
    <rect x="166" y="68" width="6" height="6" rx="1"/><rect x="180" y="68" width="6" height="6" rx="1"/><rect x="166" y="84" width="6" height="6" rx="1"/><rect x="180" y="84" width="6" height="6" rx="1"/>
    <rect x="264" y="80" width="6" height="6" rx="1"/><rect x="278" y="80" width="6" height="6" rx="1"/><rect x="264" y="96" width="6" height="6" rx="1"/>
  </g>
  <path d="M10 168 C60 150 90 160 130 118 S200 70 232 54 S280 26 306 14" fill="none" stroke="url(#ln)" stroke-width="4" stroke-linecap="round"/>
  <circle cx="306" cy="14" r="6" fill="#F4D58D"/><circle cx="306" cy="14" r="12" fill="#F4D58D" opacity=".2"/>
  <line x1="6" y1="200" x2="314" y2="200" stroke="currentColor" stroke-opacity=".25"/>
</svg>"""


# ---------------------------------------------------------------------------------------------
# HTML components
# ---------------------------------------------------------------------------------------------
def hero(chips: list[tuple[str, str]]) -> None:
    """Top banner. `chips` is a list of (bold part, regular part) tuples."""
    chip_html = "".join(
        f'<span class="chip"><b>{escape(a)}</b> {escape(b)}</span>' for a, b in chips
    )
    render_html(f"""
<div class="hero">
  <div>
    <div class="eyebrow">AI &middot; Real estate &middot; Investment intelligence</div>
    <div class="title">Real Estate<br><span class="grad">Investment Advisor</span></div>
    <p class="lede">Find out whether a property is a good investment and what it could be worth in five years &mdash;
    backed by machine learning, transparent scoring rules and a fixed-rate benchmark you can compare against.</p>
    <div class="chips">{chip_html}</div>
  </div>
  <div class="art">{hero_art()}</div>
</div>""")


def notice(text: str) -> None:
    """Slim disclaimer bar."""
    render_html(f'<div class="notice"><span>&#9432;</span><span>{text}</span></div>')


def section(label: str, title: str, sub: str = "") -> None:
    """Section header with eyebrow label, serif title and optional subtitle."""
    sub_html = f'<p class="section-sub">{sub}</p>' if sub else ""
    render_html(
        f'<div><div class="section-label">{escape(label)}</div><div class="section-title">{escape(title)}</div>{sub_html}</div>'
    )


def mini_label(text: str) -> None:
    """Small uppercase label used inside input panels."""
    render_html(f'<div class="mini-label">{escape(text)}</div>')


def kpi_grid(items: list[dict]) -> None:
    """Row of KPI cards. Each item: label, value, optional sub and tone ('up', 'down' or '')."""
    cards = ""
    for i, it in enumerate(items):
        sub = (
            f'<div class="sub {it.get("tone", "")}">{it["sub"]}</div>'
            if it.get("sub")
            else ""
        )
        cards += (
            f'<div class="kpi" style="animation-delay:{i * 70}ms"><div class="label">{escape(it["label"])}</div>'
            f'<div class="value">{it["value"]}</div>{sub}</div>'
        )
    render_html(f'<div class="kpi-grid">{cards}</div>')


def fmt_prob(p: float) -> str:
    """Show probabilities of exactly 0 or 1 as <1% / >99% (a model is never literally certain)."""
    return ">99%" if p > 0.995 else "<1%" if p < 0.005 else f"{p:.0%}"


def verdict_card(good: bool, probability: float) -> None:
    """Large verdict banner with an animated confidence ring."""
    circumference = 2 * 3.14159265 * 52
    label = fmt_prob(probability)
    filled = max(min(probability, 1.0), 0.0) * circumference
    if good:
        tag, headline = "Recommended", "Good investment"
        desc = (
            "This property clears the investment bar: it is attractively priced against its local market and "
            "scores well on infrastructure and expected growth."
        )
    else:
        tag, headline = "Not recommended", "Not a good investment"
        desc = (
            "This property does not clear the investment bar. Review the four factors below to see which "
            "ones are holding it back."
        )
    render_html(f"""
<div class="verdict {'good' if good else 'bad'}">
  <svg class="ring" viewBox="0 0 120 120" role="img" aria-label="Good-investment probability {label}">
    <circle class="track" cx="60" cy="60" r="52" fill="none" stroke-width="9"/>
    <circle class="bar" cx="60" cy="60" r="52" fill="none" stroke-width="9" stroke-linecap="round"
            stroke-dasharray="{filled:.1f} {circumference:.1f}" transform="rotate(-90 60 60)"/>
    <text class="pct" x="60" y="62" text-anchor="middle">{label}</text>
    <text class="cap" x="60" y="80" text-anchor="middle">Good-investment<tspan x="60" dy="10">probability</tspan></text>
  </svg>
  <div>
    <span class="tag">{'&#10003;' if good else '&#10005;'} {tag}</span>
    <div class="headline">{headline}</div>
    <div class="desc">{desc}</div>
  </div>
</div>""")


def factor_list(items: list[tuple[bool, str, str]]) -> None:
    """Checklist of the four investment factors: (passed, title, detail)."""
    rows = "".join(
        f'<div class="factor {"ok" if ok else "no"}"><span class="dot">{"&#10003;" if ok else "&ndash;"}</span>'
        f"<div>{escape(title)}<small>{escape(detail)}</small></div></div>"
        for ok, title, detail in items
    )
    render_html(rows)


def callout(text: str, kind: str = "") -> None:
    """Highlighted note. `kind` can be '', 'warn' or 'good'. Text may contain basic HTML."""
    render_html(f'<div class="callout {kind}">{text}</div>')


def bench_row(items: list[tuple[str, str, str]]) -> None:
    """Small benchmark tiles: (label, value, footnote)."""
    tiles = "".join(
        f"<div><span>{escape(a)}</span><b>{b}</b><br><em>{escape(c)}</em></div>"
        for a, b, c in items
    )
    render_html(f'<div class="bench">{tiles}</div>')


def sidebar_brand() -> None:
    """Brand header at the top of the sidebar."""
    render_html(
        f'<div class="brand">{logo_svg()}<div><div class="name">Investment<br>Advisor</div>'
        f'<div class="tagline">Property intelligence</div></div></div>'
    )


def side_card(key: str, value: str) -> None:
    """Compact stat tile for the sidebar."""
    render_html(
        f'<div class="side-card"><div class="k">{escape(key)}</div><div class="v">{value}</div></div>'
    )


def footer(github_url: str) -> None:
    """Page footer."""
    render_html(
        f'<div class="footer">Real Estate Investment Advisor &middot; Educational project &middot; '
        f'Predictions are not financial advice &middot; <a href="{github_url}" target="_blank">GitHub</a></div>'
    )


# ---------------------------------------------------------------------------------------------
# Plotly styling
# ---------------------------------------------------------------------------------------------
def style_fig(
    fig,
    theme: str | None = None,
    height: int = 360,
    legend: bool = True,
    title: str | None = None,
):
    """Apply the design system to a Plotly figure (transparent background, themed fonts, grid and colours)."""
    p = palette(theme)
    fig.update_layout(
        height=height,
        colorway=p["colorway"],
        margin=dict(l=12, r=16, t=56 if title else 24, b=14),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', sans-serif", color=p["muted"], size=12),
        title=(
            dict(
                text=title,
                x=0.01,
                xanchor="left",
                font=dict(
                    family="Playfair Display, Georgia, serif", size=18, color=p["text"]
                ),
            )
            if title
            else None
        ),
        showlegend=legend,
        legend=dict(
            orientation="h", y=1.08, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(
            bgcolor=p["surface2"],
            bordercolor=p["border"],
            font=dict(color=p["text"], family="Inter, sans-serif"),
        ),
    )
    # automargin grows the plot margins when tick labels or axis titles need more room (prevents clipping)
    fig.update_xaxes(
        gridcolor=p["grid"],
        zerolinecolor=p["grid"],
        linecolor=p["grid"],
        tickfont=dict(color=p["muted"]),
        automargin=True,
        title_standoff=10,
    )
    fig.update_yaxes(
        gridcolor=p["grid"],
        zerolinecolor=p["grid"],
        linecolor=p["grid"],
        tickfont=dict(color=p["muted"]),
        automargin=True,
        title_standoff=10,
    )
    return fig


def fmt_lakhs(value: float) -> str:
    """Format a lakh amount as lakhs or crores (100 lakhs = 1 crore)."""
    return f"₹{value / 100:,.2f} Cr" if value >= 100 else f"₹{value:,.1f} L"
