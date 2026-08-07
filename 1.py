import streamlit as st
import math as mt
import re
from streamlit.components.v1 import html as html_component

# --- Must stay the first Streamlit command ---
st.set_page_config(page_title="Shape Calculator", page_icon="📐", layout="centered")

# ===========================
# STYLING
# ===========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bungee&family=Nunito:wght@400;600;700;800;900&display=swap');

*{
    font-family:'Nunito', -apple-system, 'Segoe UI', sans-serif;
}
*:focus-visible{
    outline:3px solid #2563eb;
    outline-offset:2px;
}

/* ---------- APP BACKGROUND (dots drawn by JS canvas, mouse-reactive) ---------- */
[data-testid="stAppViewContainer"]{
    background-color:#f6f7fb;
}

/* ---------- DOT CANVAS IFRAME (pinned fullscreen, below content) ---------- */
iframe[srcdoc]{
    position:fixed !important;
    inset:0 !important;
    width:100vw !important;
    height:100vh !important;
    border:none !important;
    z-index:0 !important;
    pointer-events:none !important;
}

/* ---------- MAIN CARD (translucent so the interactive dot field shows through) ---------- */
.main .block-container{
    position:relative;
    z-index:2;
    background:rgba(255,255,255,.78);
    backdrop-filter:blur(3px);
    padding:2.4rem 2.4rem 2.8rem;
    border:3px solid #0f172a;
    border-radius:20px;
    box-shadow:8px 8px 0 #0f172a;
    margin:2rem auto;
    max-width:900px;
}
/* Result cards and info alerts stay readable on top of the dots */
div[data-testid="stAlert"]{
    background:rgba(220,252,231,.92) !important;
}
div[data-testid="stAlert"][data-baseweb="notification"]{
    background:rgba(220,252,231,.92) !important;
}

/* ---------- TEXT ---------- */
h1{ color:#1e3a8a; }
h2,h3,p,label,span{ color:#1e293b; }
[data-testid="stCaptionContainer"]{ color:#64748b; }
hr{ border-color:#cbd5e1 !important; }

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"]{
    position:relative;
    z-index:3;
    background:#0f172a;
    border-right:4px solid #0f172a;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{
    color:#e2e8f0 !important;
}
.side-head{
    font-family:'Bungee', Impact, 'Arial Black', sans-serif;
    font-size:1.1rem;
    letter-spacing:.5px;
    text-align:center;
    background:#1e3a8a;
    color:#ffffff !important;
    border:2px solid #334155;
    border-radius:12px;
    padding:.55rem .75rem;
    box-shadow:4px 4px 0 #020617;
}
[data-testid="stSidebar"] div[data-testid="stAlert"]{
    background:#1e293b !important;
    color:#bae6fd !important;
    border:2px solid #334155 !important;
    box-shadow:none;
}
[data-testid="stSidebar"] div[data-testid="stAlert"] *{ color:#bae6fd !important; }

/* ---------- SELECTBOX ---------- */
.stSelectbox > div > div{
    background:#ffffff !important;
    border:3px solid #0f172a;
    border-radius:12px;
    box-shadow:3px 3px 0 #020617;
}
.stSelectbox div[data-baseweb="select"] *{ color:#0f172a !important; }
div[role="listbox"]{
    background:#ffffff !important;
    border:3px solid #0f172a;
    border-radius:12px !important;
}
div[role="option"]{
    background:#ffffff !important;
    color:#0f172a !important;
    font-weight:700;
}
div[role="option"]:hover{ background:#e0e7ff !important; }
div[role="option"][aria-selected="true"]{ background:#bfdbfe !important; }

/* ---------- INPUTS ---------- */
div[data-baseweb="input"]{
    background:#ffffff;
    border:3px solid #0f172a;
    border-radius:12px;
    box-shadow:3px 3px 0 #d7dce8;
}
[data-testid="stNumberInput"]:focus-within div[data-baseweb="input"],
[data-testid="stTextInput"]:focus-within div[data-baseweb="input"]{
    border-color:#2563eb;
    box-shadow:3px 3px 0 #93c5fd, 0 0 0 3px #bfdbfe;
}

/* ---------- BUTTONS (animated: gentle bob, rainbow sheen, shine sweep, press bounce) ---------- */
.stButton > button,
.stButton button,
button[data-testid="stBaseButton-primary"]{
    width:100%;
    position:relative;
    overflow:hidden;
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    background-size:300% 300%;
    animation:btn-glow 3.5s ease infinite;
    color:#ffffff !important;
    border:3px solid #0f172a;
    border-radius:12px;
    padding:.6rem .4rem;
    font-weight:900;
    font-size:1.02rem;
    box-shadow:4px 4px 0 #0f172a;
    transition:transform .12s ease, box-shadow .12s ease;
}
@keyframes btn-glow{
    0%,100%{ background-position:0% 50%; }
    50%{ background-position:100% 50%; }
}
/* Shine sweep across the button */
.stButton > button::after,
.stButton button::after,
button[data-testid="stBaseButton-primary"]::after{
    content:"";
    position:absolute;
    top:0;
    left:-120%;
    width:60%;
    height:100%;
    background:linear-gradient(115deg, transparent 0%, rgba(255,255,255,.45) 50%, transparent 100%);
    transform:skewX(-20deg);
    animation:btn-sweep 2.8s ease-in-out infinite;
}
@keyframes btn-sweep{
    0%{ left:-120%; }
    60%,100%{ left:130%; }
}
/* Gentle hover bob + lift */
.stButton > button:hover,
button[data-testid="stBaseButton-primary"]:hover{
    animation:btn-glow 1.2s ease infinite, btn-nudge .35s ease infinite;
    transform:translateY(-2px);
    box-shadow:6px 6px 0 #0f172a;
}
@keyframes btn-nudge{
    0%,100%{ transform:translateY(-2px); }
    50%{ transform:translateY(-1px) rotate(-.4deg); }
}
/* Press: squash down hard */
.stButton > button:active,
button[data-testid="stBaseButton-primary"]:active{
    animation:none !important;
    transform:translate(4px,4px) scale(.97);
    box-shadow:1px 1px 0 #0f172a;
}
/* Rainbow border pulse on focus-visible */
.stButton > button:focus-visible,
button[data-testid="stBaseButton-primary"]:focus-visible{
    outline:none;
    border-color:#fbbf24;
    animation:btn-glow 1.2s ease infinite;
}

/* ---------- ALERTS (success/info/error keep their own colors) ---------- */
div[data-testid="stAlert"]{
    border:2.5px solid #0f172a !important;
    border-radius:12px !important;
    box-shadow:3px 3px 0 rgba(15,23,42,.18);
    font-weight:700;
}

/* ---------- HERO ---------- */
.hero{
    display:flex;
    align-items:center;
    gap:1rem;
    background:#1e3a8a;
    border:3px solid #0f172a;
    border-radius:18px;
    padding:1.1rem 1.4rem;
    box-shadow:6px 6px 0 #0f172a;
}
.hero-emoji{ font-size:3rem; animation:float 3s ease-in-out infinite; }
.hero-title{
    font-family:'Bungee', Impact, 'Arial Black', sans-serif;
    font-size:2rem;
    line-height:1;
    margin:0;
    color:#ffffff;
    letter-spacing:1px;
}
.hero-title span{ color:#fbbf24; text-shadow:2px 2px 0 #0f172a; }
.hero-sub{ margin:.35rem 0 0; color:#bfdbfe; font-weight:800; font-size:.95rem; }
@keyframes float{
    0%,100%{ transform:translateY(0); }
    50%{ transform:translateY(-6px); }
}
.chips{ display:flex; gap:.5rem; flex-wrap:wrap; margin:1.2rem 0 1.5rem; }
.chip{
    background:#ffffff;
    border:2.5px solid #0f172a;
    border-radius:999px;
    padding:.25rem .8rem;
    font-weight:800;
    font-size:.8rem;
    box-shadow:3px 3px 0 #0f172a;
}

/* ---------- SHAPE DIAGRAM ---------- */
.shape-box{
    display:flex;
    justify-content:center;
    margin:.75rem 0 .25rem;
}
.shape-box svg{ filter:drop-shadow(5px 5px 0 rgba(15,23,42,.35)); max-width:100%; height:auto; }
.shape-caption{
    text-align:center;
    color:#64748b;
    font-weight:600;
    font-size:.9rem;
    margin-bottom:1rem;
}

/* ---------- RESULT CARDS ---------- */
.results-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:1rem;
    margin-top:.75rem;
}
@media (max-width:600px){
    .results-grid{ grid-template-columns:1fr; }
}
.r-card{
    border:3px solid #0f172a;
    border-radius:16px;
    padding:1rem 1.2rem;
    box-shadow:5px 5px 0 rgba(15,23,42,.9);
}
.r-card.span2{ grid-column:1 / -1; }
.r-green{ background:#dcfce7; border-left:10px solid #22c55e; }
.r-amber{ background:#fef3c7; border-left:10px solid #f59e0b; }
.r-blue{ background:#dbeafe; border-left:10px solid #3b82f6; }
.r-emoji{ font-size:1.6rem; line-height:1; }
.r-label{
    font-weight:900;
    text-transform:uppercase;
    font-size:.78rem;
    letter-spacing:.5px;
    color:#334155;
    margin-top:.2rem;
}
.r-unit{ color:#64748b; font-weight:800; text-transform:none; }
.r-value{
    font-family:'Consolas','JetBrains Mono',monospace;
    font-weight:900;
    font-size:2rem;
    color:#0f172a;
    line-height:1.1;
    margin:.15rem 0 .35rem;
    word-break:break-all;
}
.r-formula{
    font-size:.85rem;
    color:#475569;
    font-weight:700;
    background:rgba(255,255,255,.65);
    padding:.25rem .55rem;
    border-radius:8px;
    display:inline-block;
}

/* ---------- CALCULATOR ---------- */
.calc-answer .r-value{ font-size:2.4rem; }
.calc-hint{
    color:#94a3b8;
    font-weight:700;
    text-align:center;
    padding:.9rem;
    font-size:.9rem;
}
.calc-tip{
    background:#f1f5f9;
    border:2px dashed #94a3b8;
    border-radius:12px;
    padding:.6rem .9rem;
    color:#475569;
    font-size:.85rem;
    font-weight:700;
    margin-top:1rem;
}

/* ---------- FOOTER ---------- */
.footer{
    margin-top:2.2rem;
    text-align:center;
    color:#94a3b8;
    font-weight:700;
    font-size:.82rem;
    border-top:3px dashed #cbd5e1;
    padding-top:1rem;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# MOUSE-REACTIVE DOT BACKGROUND
# (full-viewport canvas: dots swell, warm up and ripple near the cursor)
# ===========================
html_component(
    """
    <canvas id="dot-canvas" style="display:block;"></canvas>
    <script>
    (function(){
      if(window.__dotFieldStarted){ return; }
      window.__dotFieldStarted = true;

      const canvas = document.getElementById('dot-canvas');
      const ctx = canvas.getContext('2d');

      // The iframe is pinned fullscreen + pointer-events:none,
      // so listen on the PARENT window for real mouse movement.
      let parent = null;
      try { parent = window.parent; } catch(e){}
      const win = (parent && parent.document) ? parent : window;

      const GAP = 28;
      const MOUSE_RADIUS = 140;
      let dots = [];
      let w = 0, h = 0, dpr = 1;
      const mouse = { x: -9999, y: -9999 };

      function makeGrid(){
        dots = [];
        for(let x = GAP / 2; x < w + GAP; x += GAP){
          for(let y = GAP / 2; y < h + GAP; y += GAP){
            dots.push({ x, y, base: 2.4, seed: Math.random() * Math.PI * 2 });
          }
        }
      }

      function resize(){
        dpr = Math.max(1, win.devicePixelRatio || 1);
        w = win.innerWidth;
        h = win.innerHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        makeGrid();
      }

      win.addEventListener('resize', resize);

      if(parent && parent !== window){
        parent.addEventListener('mousemove', (e) => {
          mouse.x = e.clientX;
          mouse.y = e.clientY;
        });
        parent.addEventListener('mouseleave', () => {
          mouse.x = -9999;
          mouse.y = -9999;
        });
      }

      const baseColor = [125, 138, 158];
      const hotColor  = [37, 99, 235];
      function lerp(a, b, t){ return a + (b - a) * t; }

      function draw(now){
        const t = now * 0.001;
        ctx.clearRect(0, 0, w, h);

        for(const d of dots){
          const dx = d.x - mouse.x;
          const dy = d.y - mouse.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          let influence = 1 - dist / MOUSE_RADIUS;
          if(influence < 0){ influence = 0; }
          influence = influence * influence;

          // Dots swell and heat up (slate -> blue) near the cursor
          const r = d.base + influence * 6.5;
          // Gentle idle shimmer so the field breathes on its own
          const shimmer = 0.6 + 0.4 * Math.sin(t * 1.4 + d.seed);
          const cr = lerp(baseColor[0], hotColor[0], influence);
          const cg = lerp(baseColor[1], hotColor[1], influence);
          const cb = lerp(baseColor[2], hotColor[2], influence);

          ctx.beginPath();
          ctx.arc(d.x, d.y, r * shimmer, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${cr|0},${cg|0},${cb|0},${0.55 + influence * 0.35})`;
          ctx.fill();
        }

        // Expanding ripple ring that follows the cursor
        if(mouse.x > -9999){
          const age = (t * 2) % 1;
          const pulse = 10 + age * 46;
          ctx.beginPath();
          ctx.arc(mouse.x, mouse.y, pulse, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(37,99,235,${(1 - age) * 0.45})`;
          ctx.lineWidth = 2.5;
          ctx.stroke();
        }

        requestAnimationFrame(draw);
      }

      resize();
      requestAnimationFrame(draw);
    })();
    </script>
    """,
    height=100,
)


# ===========================
# HELPERS
# ===========================
def fmt(value):
    """Pretty-print a number: thousands separators, trimmed decimals."""
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value)):,}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{value:,}"


def field(label, hint):
    return st.number_input(f"{label} ({hint})", min_value=0.0, value=0.0, step=0.1)


def results_markup(results, unit):
    """Render result cards. Odd card counts widen the first card to full row."""
    cards = []
    total = len(results)
    for i, (label, emoji, value, formula, color) in enumerate(results):
        if "Volume" in label:
            suffix = f"{unit}³"
        elif "Area" in label or "Circumference" in label:
            suffix = f"{unit}²"
        else:
            suffix = unit
        span = ' span2' if (total % 2 == 1 and i == 0) else ''
        cards.append(
            f'<div class="r-card r-{color}{span}">'
            f'<div class="r-emoji">{emoji}</div>'
            f'<div class="r-label">{label} <span class="r-unit">({suffix})</span></div>'
            f'<div class="r-value">{fmt(value)}</div>'
            f'<div class="r-formula">= {formula}</div>'
            f'</div>'
        )
    return f'<div class="results-grid">{"".join(cards)}</div>'


def compute_expression(expr):
    """Safe evaluator with implicit multiplication, powers, and percent."""
    expr = expr.strip()
    if not expr:
        return None

    s = expr
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("^", "**")
    # Percent: treat 50% as 50/100
    s = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", s)
    # Implicit multiplication
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"\)(\d)", r")*\1", s)
    s = re.sub(r"\)\(", r")*(", s)

    if not re.fullmatch(r"[0-9+\-*/().\s]+", s):
        raise ValueError("invalid characters")
    return eval(s, {"__builtins__": None}, {})


# ===========================
# 2D SHAPE DIAGRAMS (SVG)
# ===========================
SVG_CIRCLE = """
<svg viewBox="0 0 200 200" width="170" height="170" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="100" r="72" fill="#93c5fd" stroke="#0f172a" stroke-width="5"/>
  <line x1="100" y1="100" x2="100" y2="28" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="100" cy="100" r="7" fill="#0f172a"/>
  <circle cx="100" cy="28" r="7" fill="#0f172a"/>
  <text x="112" y="70" font-family="Arial" font-size="26" font-weight="bold" fill="#0f172a">r</text>
</svg>"""

SVG_RECT = """
<svg viewBox="0 0 200 200" width="200" height="200" xmlns="http://www.w3.org/2000/svg">
  <rect x="35" y="60" width="130" height="85" fill="#f9a8d4" stroke="#0f172a" stroke-width="5" rx="2"/>
  <line x1="35" y1="52" x2="165" y2="52" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="35" cy="52" r="6" fill="#0f172a"/>
  <circle cx="165" cy="52" r="6" fill="#0f172a"/>
  <text x="78" y="44" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">l</text>
  <line x1="173" y1="60" x2="173" y2="145" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="173" cy="60" r="6" fill="#0f172a"/>
  <circle cx="173" cy="145" r="6" fill="#0f172a"/>
  <text x="179" y="108" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">w</text>
</svg>"""

SVG_SQUARE = """
<svg viewBox="0 0 200 200" width="160" height="160" xmlns="http://www.w3.org/2000/svg">
  <rect x="45" y="45" width="110" height="110" fill="#fde047" stroke="#0f172a" stroke-width="5"/>
  <text x="100" y="110" text-anchor="middle" font-family="Arial" font-size="30" font-weight="bold" fill="#0f172a">s</text>
</svg>"""

SVG_TRIANGLE = """
<svg viewBox="0 0 200 200" width="200" height="200" xmlns="http://www.w3.org/2000/svg">
  <polygon points="100,25 175,155 25,155" fill="#86efac" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <line x1="100" y1="155" x2="100" y2="25" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="100" cy="155" r="6" fill="#0f172a"/>
  <circle cx="100" cy="25" r="6" fill="#0f172a"/>
  <text x="70" y="88" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">h</text>
  <line x1="25" y1="165" x2="175" y2="165" stroke="#0f172a" stroke-width="3"/>
  <circle cx="25" cy="165" r="6" fill="#0f172a"/>
  <circle cx="175" cy="165" r="6" fill="#0f172a"/>
  <text x="82" y="185" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">b</text>
</svg>"""

SVG_PARALLELOGRAM = """
<svg viewBox="0 0 200 200" width="210" height="200" xmlns="http://www.w3.org/2000/svg">
  <polygon points="50,115 175,115 152,160 27,160" fill="#fdba74" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <line x1="38" y1="160" x2="38" y2="115" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="38" cy="160" r="6" fill="#0f172a"/>
  <circle cx="38" cy="115" r="6" fill="#0f172a"/>
  <text x="12" y="142" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">h</text>
  <line x1="50" y1="107" x2="175" y2="107" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="50" cy="107" r="6" fill="#0f172a"/>
  <circle cx="175" cy="107" r="6" fill="#0f172a"/>
  <text x="95" y="100" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">b</text>
</svg>"""

SVG_TRAPEZOID = """
<svg viewBox="0 0 200 200" width="200" height="200" xmlns="http://www.w3.org/2000/svg">
  <polygon points="60,85 145,85 172,155 30,155" fill="#c4b5fd" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <line x1="60" y1="78" x2="145" y2="78" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="60" cy="78" r="6" fill="#0f172a"/>
  <circle cx="145" cy="78" r="6" fill="#0f172a"/>
  <text x="82" y="70" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">b₁</text>
  <line x1="30" y1="163" x2="172" y2="163" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="30" cy="163" r="6" fill="#0f172a"/>
  <circle cx="172" cy="163" r="6" fill="#0f172a"/>
  <text x="82" y="184" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">b₂</text>
</svg>"""

SVG_RHOMBUS = """
<svg viewBox="0 0 200 200" width="180" height="180" xmlns="http://www.w3.org/2000/svg">
  <polygon points="100,35 162,100 100,165 38,100" fill="#67e8f9" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <line x1="100" y1="35" x2="100" y2="165" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <line x1="38" y1="100" x2="162" y2="100" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <text x="106" y="62" font-family="Arial" font-size="22" font-weight="bold" fill="#0f172a">d₁</text>
  <text x="78" y="82" font-family="Arial" font-size="22" font-weight="bold" fill="#0f172a">d₂</text>
</svg>"""


# ===========================
# 3D SHAPE DIAGRAMS (SVG)
# ===========================
SVG_CUBE = """
<svg viewBox="0 0 200 200" width="185" height="185" xmlns="http://www.w3.org/2000/svg">
  <polygon points="100,32 156,61 100,90 44,61" fill="#fde047" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <polygon points="44,61 100,90 100,158 44,129" fill="#fbbf24" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <polygon points="100,90 156,61 156,129 100,158" fill="#f59e0b" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <text x="78" y="52" font-family="Arial" font-size="26" font-weight="bold" fill="#0f172a">s</text>
</svg>"""

SVG_CUBOID = """
<svg viewBox="0 0 220 200" width="210" height="190" xmlns="http://www.w3.org/2000/svg">
  <polygon points="70,40 160,40 184,66 94,66" fill="#fde047" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <polygon points="94,66 184,66 184,142 94,142" fill="#fbbf24" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <polygon points="70,40 94,66 94,142 70,118" fill="#f59e0b" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <text x="62" y="136" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">l</text>
  <text x="118" y="150" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">w</text>
  <text x="45" y="90" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">h</text>
</svg>"""

SVG_SPHERE = """
<svg viewBox="0 0 200 200" width="180" height="180" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="100" r="74" fill="#93c5fd" stroke="#0f172a" stroke-width="5"/>
  <ellipse cx="100" cy="100" rx="27" ry="74" fill="#bfdbfe" stroke="#0f172a" stroke-width="3"/>
  <line x1="100" y1="100" x2="100" y2="26" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="100" cy="100" r="7" fill="#0f172a"/>
  <circle cx="100" cy="26" r="7" fill="#0f172a"/>
  <text x="110" y="68" font-family="Arial" font-size="26" font-weight="bold" fill="#0f172a">r</text>
</svg>"""

SVG_CYLINDER = """
<svg viewBox="0 0 200 200" width="190" height="190" xmlns="http://www.w3.org/2000/svg">
  <rect x="38" y="52" width="124" height="94" fill="#fb7185"/>
  <ellipse cx="100" cy="146" rx="62" ry="20" fill="#e11d48" stroke="#0f172a" stroke-width="5"/>
  <ellipse cx="100" cy="52" rx="62" ry="20" fill="#fda4af" stroke="#0f172a" stroke-width="5"/>
  <line x1="100" y1="52" x2="162" y2="52" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <text x="118" y="42" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">r</text>
  <line x1="30" y1="52" x2="30" y2="146" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="30" cy="52" r="6" fill="#0f172a"/>
  <circle cx="30" cy="146" r="6" fill="#0f172a"/>
  <text x="6" y="105" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">h</text>
</svg>"""

SVG_CONE = """
<svg viewBox="0 0 200 200" width="195" height="195" xmlns="http://www.w3.org/2000/svg">
  <line x1="100" y1="24" x2="100" y2="150" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="100" cy="24" r="6" fill="#0f172a"/>
  <circle cx="100" cy="150" r="6" fill="#0f172a"/>
  <text x="108" y="92" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">h</text>
  <polygon points="100,24 176,150 24,150" fill="#fdba74" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <ellipse cx="100" cy="150" rx="78" ry="24" fill="#f97316" stroke="#0f172a" stroke-width="5"/>
  <text x="106" y="182" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">r</text>
</svg>"""

SVG_PYRAMID = """
<svg viewBox="0 0 200 200" width="185" height="185" xmlns="http://www.w3.org/2000/svg">
  <line x1="100" y1="26" x2="100" y2="100" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <polygon points="100,66 156,100 100,134 44,100" fill="#c4b5fd" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <polygon points="100,26 44,100 100,134" fill="#a78bfa" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <polygon points="100,26 156,100 100,134" fill="#8b5cf6" stroke="#0f172a" stroke-width="5" stroke-linejoin="round"/>
  <text x="58" y="116" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">s</text>
  <text x="108" y="56" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">h</text>
</svg>"""

SVG_TORUS = """
<svg viewBox="0 0 200 200" width="190" height="190" xmlns="http://www.w3.org/2000/svg">
  <circle cx="100" cy="108" r="72" fill="#f9a8d4" stroke="#0f172a" stroke-width="5"/>
  <circle cx="100" cy="108" r="30" fill="#ffffff" stroke="#0f172a" stroke-width="5"/>
  <path d="M 100 42 A 62 62 0 0 1 158 80 A 52 52 0 0 0 100 54 Z" fill="#fbcfe8"/>
  <line x1="100" y1="108" x2="172" y2="108" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="100" cy="108" r="6" fill="#0f172a"/>
  <circle cx="172" cy="108" r="6" fill="#0f172a"/>
  <text x="126" y="100" font-family="Arial" font-size="24" font-weight="bold" fill="#0f172a">R</text>
  <line x1="100" y1="36" x2="100" y2="102" stroke="#0f172a" stroke-width="3" stroke-dasharray="7 6"/>
  <circle cx="100" cy="36" r="6" fill="#0f172a"/>
  <circle cx="100" cy="102" r="6" fill="#0f172a"/>
  <text x="108" y="64" font-family="Arial" font-size="22" font-weight="bold" fill="#0f172a">r</text>
</svg>"""


# ===========================
# 2D SHAPES
# ===========================
SHAPES_2D = {
    "⚪ Circle": {
        "svg": SVG_CIRCLE,
        "desc": "All points at a fixed distance from a center",
        "fields": [("Radius", "r")],
        "calc": lambda v: [
            ("Area", "📏", mt.pi * v[0] ** 2, f"π × {fmt(v[0])}²", "green"),
            ("Circumference", "⭕", 2 * mt.pi * v[0], f"2π × {fmt(v[0])}", "amber"),
        ],
    },
    "▭ Rectangle": {
        "svg": SVG_RECT,
        "desc": "Four right angles, opposite sides equal",
        "fields": [("Length", "l"), ("Width", "w")],
        "calc": lambda v: [
            ("Area", "📏", v[0] * v[1], f"{fmt(v[0])} × {fmt(v[1])}", "green"),
            ("Perimeter", "🔄", 2 * (v[0] + v[1]), f"2 × ({fmt(v[0])} + {fmt(v[1])})", "amber"),
        ],
    },
    "🟨 Square": {
        "svg": SVG_SQUARE,
        "desc": "Four equal sides, four right angles",
        "fields": [("Side", "s")],
        "calc": lambda v: [
            ("Area", "📏", v[0] ** 2, f"{fmt(v[0])}²", "green"),
            ("Perimeter", "🔄", 4 * v[0], f"4 × {fmt(v[0])}", "amber"),
        ],
    },
    "🔺 Triangle": {
        "svg": SVG_TRIANGLE,
        "desc": "Three sides — area uses base & height",
        "fields": [("Base", "b"), ("Height", "h"), ("Side 1", "s₁"), ("Side 2", "s₂")],
        "calc": lambda v: [
            ("Area", "📏", 0.5 * v[0] * v[1], f"½ × {fmt(v[0])} × {fmt(v[1])}", "green"),
            ("Perimeter", "🔄", v[0] + v[2] + v[3], f"{fmt(v[0])} + {fmt(v[2])} + {fmt(v[3])}", "amber"),
        ],
    },
    "▱ Parallelogram": {
        "svg": SVG_PARALLELOGRAM,
        "desc": "Opposite sides parallel and equal",
        "fields": [("Base", "b"), ("Height", "h"), ("Side", "s")],
        "calc": lambda v: [
            ("Area", "📏", v[0] * v[1], f"{fmt(v[0])} × {fmt(v[1])}", "green"),
            ("Perimeter", "🔄", 2 * (v[0] + v[2]), f"2 × ({fmt(v[0])} + {fmt(v[2])})", "amber"),
        ],
    },
    "⬠ Trapezoid": {
        "svg": SVG_TRAPEZOID,
        "desc": "One pair of parallel sides",
        "fields": [("Base 1", "b₁"), ("Base 2", "b₂"), ("Height", "h"), ("Side 1", "s₁"), ("Side 2", "s₂")],
        "calc": lambda v: [
            ("Area", "📏", ((v[0] + v[1]) / 2) * v[2], f"(({fmt(v[0])} + {fmt(v[1])}) / 2) × {fmt(v[2])}", "green"),
            ("Perimeter", "🔄", v[0] + v[1] + v[3] + v[4], f"{fmt(v[0])} + {fmt(v[1])} + {fmt(v[3])} + {fmt(v[4])}", "amber"),
        ],
    },
    "💎 Rhombus": {
        "svg": SVG_RHOMBUS,
        "desc": "Four equal sides — area uses the diagonals",
        "fields": [("Diagonal 1", "d₁"), ("Diagonal 2", "d₂"), ("Side", "s")],
        "calc": lambda v: [
            ("Area", "📏", (v[0] * v[1]) / 2, f"({fmt(v[0])} × {fmt(v[1])}) / 2", "green"),
            ("Perimeter", "🔄", 4 * v[2], f"4 × {fmt(v[2])}", "amber"),
        ],
    },
}


# ===========================
# 3D SHAPES
# ===========================
def _slant(radius, height):
    return mt.sqrt(radius ** 2 + height ** 2)


SHAPES_3D = {
    "🧊 Cube": {
        "svg": SVG_CUBE,
        "desc": "A box where every edge is the same length",
        "fields": [("Side", "s")],
        "calc": lambda v: [
            ("Volume", "🧊", v[0] ** 3, f"{fmt(v[0])}³", "green"),
            ("Surface Area", "🟨", 6 * v[0] ** 2, f"6 × {fmt(v[0])}²", "amber"),
            ("Space Diagonal", "📏", v[0] * mt.sqrt(3), f"{fmt(v[0])} × √3", "blue"),
        ],
    },
    "📦 Cuboid": {
        "svg": SVG_CUBOID,
        "desc": "A box — length, width and height can all differ",
        "fields": [("Length", "l"), ("Width", "w"), ("Height", "h")],
        "calc": lambda v: [
            ("Volume", "📦", v[0] * v[1] * v[2], f"{fmt(v[0])} × {fmt(v[1])} × {fmt(v[2])}", "green"),
            ("Surface Area", "🟨", 2 * (v[0] * v[1] + v[0] * v[2] + v[1] * v[2]),
                f"2 × ({fmt(v[0] * v[1])} + {fmt(v[0] * v[2])} + {fmt(v[1] * v[2])})", "amber"),
            ("Space Diagonal", "📏", mt.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2),
                f"√({fmt(v[0])}² + {fmt(v[1])}² + {fmt(v[2])}²)", "blue"),
        ],
    },
    "🌐 Sphere": {
        "svg": SVG_SPHERE,
        "desc": "Perfectly round — every point on the surface is r away from the center",
        "fields": [("Radius", "r")],
        "calc": lambda v: [
            ("Volume", "🌐", (4 / 3) * mt.pi * v[0] ** 3, f"4/3 × π × {fmt(v[0])}³", "green"),
            ("Surface Area", "🟠", 4 * mt.pi * v[0] ** 2, f"4π × {fmt(v[0])}²", "amber"),
        ],
    },
    "🛢️ Cylinder": {
        "svg": SVG_CYLINDER,
        "desc": "Two circular bases connected by a curved side",
        "fields": [("Radius", "r"), ("Height", "h")],
        "calc": lambda v: [
            ("Volume", "🛢️", mt.pi * v[0] ** 2 * v[1], f"π × {fmt(v[0])}² × {fmt(v[1])}", "green"),
            ("Surface Area", "🟠", 2 * mt.pi * v[0] * (v[0] + v[1]), f"2π × {fmt(v[0])} × ({fmt(v[0])} + {fmt(v[1])})", "amber"),
        ],
    },
    "🍦 Cone": {
        "svg": SVG_CONE,
        "desc": "A circular base that tapers up to a point",
        "fields": [("Radius", "r"), ("Height", "h")],
        "calc": lambda v: [
            ("Volume", "🍦", (mt.pi * v[0] ** 2 * v[1]) / 3, f"⅓ × π × {fmt(v[0])}² × {fmt(v[1])}", "green"),
            ("Surface Area", "🟠", mt.pi * v[0] * (v[0] + _slant(v[0], v[1])),
                f"π × {fmt(v[0])} × ({fmt(v[0])} + {fmt(_slant(v[0], v[1]))})", "amber"),
            ("Slant Height", "📏", _slant(v[0], v[1]), f"√({fmt(v[0])}² + {fmt(v[1])}²)", "blue"),
        ],
    },
    "⛺ Square Pyramid": {
        "svg": SVG_PYRAMID,
        "desc": "A square base with four triangular faces meeting at an apex",
        "fields": [("Base Side", "s"), ("Height", "h")],
        "calc": lambda v: [
            ("Volume", "⛺", (v[0] ** 2 * v[1]) / 3, f"⅓ × {fmt(v[0])}² × {fmt(v[1])}", "green"),
            ("Surface Area", "🟠", v[0] ** 2 + 2 * v[0] * _slant(v[0] / 2, v[1]),
                f"{fmt(v[0])}² + 2 × {fmt(v[0])} × {fmt(_slant(v[0] / 2, v[1]))}", "amber"),
            ("Face Slant", "📏", _slant(v[0] / 2, v[1]), f"√(({fmt(v[0])} / 2)² + {fmt(v[1])}²)", "blue"),
        ],
    },
    "🍩 Torus": {
        "svg": SVG_TORUS,
        "desc": "A donut — tube radius r spun around a center radius R",
        "fields": [("Major Radius R", "R"), ("Minor Radius r", "r")],
        "check": lambda v: "Major radius (R) must be ≥ minor radius (r) — otherwise it's a spindle torus."
        if v[0] < v[1] else None,
        "calc": lambda v: [
            ("Volume", "🍩", 2 * mt.pi ** 2 * v[0] * v[1] ** 2, f"2π² × {fmt(v[0])} × {fmt(v[1])}²", "green"),
            ("Surface Area", "🟠", 4 * mt.pi ** 2 * v[0] * v[1], f"4π² × {fmt(v[0])} × {fmt(v[1])}", "amber"),
        ],
    },
}


# ===========================
# CALCULATOR LOGIC
# ===========================
def clear_expr():
    st.session_state.calc_expr = ""


def backspace():
    st.session_state.calc_expr = st.session_state.get("calc_expr", "")[:-1]


def make_press(char):
    def _press():
        st.session_state.calc_expr = st.session_state.get("calc_expr", "") + char
    return _press


def do_equal():
    try:
        value = compute_expression(st.session_state.get("calc_expr", ""))
    except Exception:
        return
    if value is not None:
        st.session_state.calc_expr = fmt(value)


def calculator_section():
    st.subheader("🧮 Calculator")
    st.caption("Implicit multiplication works: `2(3+4)` = 14. Use `^` for powers and `%` for percent.")

    if "calc_expr" not in st.session_state:
        st.session_state.calc_expr = ""

    st.text_input(
        "Expression",
        key="calc_expr",
        placeholder="e.g. 2(3+4) + 10%",
    )

    pad = [
        [("C", "clear"), ("⌫", "back"), ("(", "("), (")", ")")],
        [("7", "7"), ("8", "8"), ("9", "9"), ("÷", "÷")],
        [("4", "4"), ("5", "5"), ("6", "6"), ("×", "×")],
        [("1", "1"), ("2", "2"), ("3", "3"), ("−", "−")],
        [("0", "0"), (".", "."), ("%", "%"), ("^", "^")],
    ]
    for row in pad:
        cols = st.columns(4)
        for col, (label, val) in zip(cols, row):
            with col:
                if val == "clear":
                    st.button(label, key=f"pad-{label}", on_click=clear_expr)
                elif val == "back":
                    st.button(label, key=f"pad-{label}", on_click=backspace)
                else:
                    st.button(label, key=f"pad-{label}", on_click=make_press(val))

    st.button("=", key="calc-eq", on_click=do_equal, use_container_width=True)

    expr = st.session_state.get("calc_expr", "")
    if expr.strip():
        try:
            value = compute_expression(expr)
            if value is not None:
                st.markdown(
                    f'<div class="r-card r-green calc-answer">'
                    f'<div class="r-emoji">🎯</div>'
                    f'<div class="r-label">Result</div>'
                    f'<div class="r-value">{fmt(value)}</div>'
                    f'<div class="r-formula">= {expr.strip()}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except ZeroDivisionError:
            st.error("❌ Cannot divide by zero.")
        except Exception:
            st.markdown(
                '<div class="calc-hint">Keep typing… or press = to snap the answer into the box.</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="calc-tip">💡 Examples: <b>2(3+4)</b> · <b>(5+1)2</b> · <b>2^10</b> · <b>15% of 200 → 15%×200</b></div>',
        unsafe_allow_html=True,
    )


# ===========================
# PAGE
# ===========================
st.markdown(
    """
    <div class="hero">
        <div class="hero-emoji">📐</div>
        <div>
            <div class="hero-title">SHAPE <span>CALCULATOR</span></div>
            <div class="hero-sub">Area • Perimeter • Volume & Surface — computed instantly.</div>
        </div>
    </div>
    <div class="chips">
        <span class="chip">⚡ Live results</span>
        <span class="chip">📐 7 planar shapes</span>
        <span class="chip">🧊 7 solid shapes</span>
        <span class="chip">🧮 Smart calculator</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="side-head">📐 SHAPE CALC</div>', unsafe_allow_html=True)
    st.caption("Geometry Made Easy")

    mode = st.radio("🧭 Mode", ["2D Shapes", "3D Shapes", "🧮 Calculator"], index=0)

    st.divider()
    unit = st.selectbox("📏 Unit", ["cm", "m", "in", "ft"], index=0)
    st.divider()

    st.info(
        """
        **Features**
        - 📐 Area & Perimeter
        - 🧊 Volume & Surface Area
        - 🧮 Calculator with keypad
        """
    )
    st.divider()
    st.caption("Made with ❤️ using Streamlit")

if mode == "🧮 Calculator":
    calculator_section()
else:
    pool = SHAPES_2D if mode == "2D Shapes" else SHAPES_3D
    option = st.selectbox("🧩 Choose a Shape", list(pool.keys()), key="shape_select")
    shape = pool[option]

    st.subheader(option)
    st.markdown(f'<div class="shape-box">{shape["svg"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="shape-caption">{shape["desc"]} · results update as you type</div>',
        unsafe_allow_html=True,
    )

    values = [field(label, hint) for label, hint in shape["fields"]]

    check = shape.get("check")
    error = check(values) if check else None

    if error:
        st.error(f"❌ {error}")
    elif all(v == 0 for v in values):
        st.info("👆 Enter a value above — results appear instantly.")
    else:
        st.markdown(results_markup(shape["calc"](values), unit), unsafe_allow_html=True)

st.markdown(
    '<div class="footer">© Shape Calculator — made with ❤️ in Python + Streamlit</div>',
    unsafe_allow_html=True,
)
