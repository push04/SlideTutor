from typing import List, Dict, Optional, Tuple, Any

# Keep palette to 3–5 colors total and high contrast, no purple
THEME_CSS = """
:root{
  --bg: #0B1220;         /* background (neutral) */
  --panel: #101A2D;      /* panel (neutral) */
  --text: #E6EDF6;       /* foreground (neutral) */
  --primary: #0EA5E9;    /* brand blue */
  --accent: #22C55E;     /* accent green */
  --radius: 12px;
  --shadow-soft: 0 8px 36px rgba(2,6,12,0.45);
  --max-width: 1180px;
  --top-offset: 84px;     /* avoid Streamlit header overlap */
  --side-gutter: 24px;    /* page gutters */
}

html, body, .stApp {
  background: linear-gradient(180deg, var(--bg) 0%, #0A1628 100%);
  color: var(--text);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  line-height: 1.45;
  box-sizing: border-box;
  overflow-x: hidden;
}

.stApp .block-container {
  padding: var(--top-offset) var(--side-gutter) 36px !important;
  max-width: var(--max-width);
  margin-left: auto;
  margin-right: auto;
  width: calc(100% - (var(--side-gutter) * 2));
}

.small-muted { color: rgba(230,237,246,0.65); font-size: 0.9rem; }

.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.015) 100%);
  border: 1px solid rgba(230,237,246,0.08);
  border-radius: var(--radius);
  box-shadow: var(--shadow-soft);
  padding: 16px;
  margin: 12px 0;
}

.card.tight { padding: 12px; }
.card .card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 8px; }

.hr {
  height: 1px;
  background: linear-gradient(90deg, rgba(230,237,246,0.1), rgba(230,237,246,0.02));
  border: 0; margin: 14px 0;
}

.kpis {
  display: grid;
  grid-template-columns: repeat(3, minmax(0,1fr));
  gap: 12px;
}
.kpi {
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(230,237,246,0.08);
  border-radius: var(--radius);
  padding: 12px;
}
.kpi .label { color: rgba(230,237,246,0.7); font-size: 0.9rem; }
.kpi .value { font-size: 1.6rem; font-weight: 700; }

.actions {
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}

.stButton > button {
  background: var(--primary);
  color: var(--bg);
  border: 1px solid rgba(230,237,246,0.08);
  border-radius: calc(var(--radius) - 4px);
  padding: 8px 14px;
  font-weight: 600;
  transition: transform 120ms ease, filter 120ms ease;
}
.stButton > button:hover { filter: brightness(0.92); transform: translateY(-1px); }
.stButton > button:focus { outline: 2px solid rgba(14,165,233,0.55); outline-offset: 2px; }

.btn-link {
  display: inline-flex; align-items: center; gap: 6px;
  color: var(--primary); text-decoration: none; font-weight: 600;
}
.btn-link:hover { text-decoration: underline; }

.chat {
  display: flex; flex-direction: column; gap: 10px;
}
.msg {
  max-width: 82%;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(230,237,246,0.06);
  background: rgba(255,255,255,0.03);
}
.msg.user {
  align-self: flex-end;
  background: var(--primary);
  color: var(--bg);
}
.msg.ai {
  align-self: flex-start;
  background: rgba(255,255,255,0.05);
  color: var(--text);
}
.msg .role { font-size: 0.8rem; opacity: 0.8; margin-bottom: 4px; }

.badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,0.06);
  color: var(--text);
  border: 1px solid rgba(230,237,246,0.08);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.85rem;
}

a, a:visited { color: var(--primary); }
"""

def apply_theme(st) -> None:
    """
    Injects the theme CSS. Safe to call multiple times. Requires st.markdown.
    """
    try:
        st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)
    except Exception:
        pass

def header(st, title: str, subtitle: Optional[str] = None, icon: Optional[str] = "🎓") -> None:
    """
    Renders a compact, professional header/brand area.
    """
    try:
        icon_html = f"<div style='font-size:2rem;line-height:1'>{icon}</div>" if icon else ""
        sub_html = f"<div class='small-muted'>{subtitle}</div>" if subtitle else ""
        st.markdown(
            f"<div role='banner' style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>"
            f"{icon_html}"
            f"<div><div style='font-weight:700;font-size:1.35rem'>{title}</div>{sub_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.title(title)
        if subtitle:
            st.caption(subtitle)

def hero(st, title: str, subtitle: str, actions: Optional[List[Tuple[str, str]]] = None) -> Optional[str]:
    """
    Renders a simple hero with optional primary actions.
    actions: list of (label, key), returns the key of the pressed button or None.
    """
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin:0 0 8px 0;'>{title}</h2>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted' style='margin-bottom:12px'>{subtitle}</div>", unsafe_allow_html=True)

    clicked: Optional[str] = None
    if actions:
        cols = st.columns([1] * len(actions))
        for i, (label, key) in enumerate(actions):
            if cols[i].button(label, key=f"hero_{key}"):
                clicked = key
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked

def kpis(st, items: List[Tuple[str, str]]) -> None:
    """
    Renders three KPI tiles (label, value). More than 3 will auto-wrap.
    """
    st.markdown("<div class='kpis'>", unsafe_allow_html=True)
    for label, value in items:
        st.markdown(
            f"<div class='kpi'><div class='label'>{label}</div><div class='value'>{value}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def section(st, title: str, description: Optional[str] = None) -> None:
    st.markdown(f"### {title}")
    if description:
        st.markdown(f"<div class='small-muted' style='margin-top:-6px'>{description}</div>", unsafe_allow_html=True)

def card_md(st, title: Optional[str] = None, md: Optional[str] = None, tight: bool = False) -> None:
    """
    Render a simple markdown card.
    """
    klass = "card tight" if tight else "card"
    st.markdown(f"<div class='{klass}'>", unsafe_allow_html=True)
    if title:
        st.markdown(f"<div class='card-title'>{title}</div>", unsafe_allow_html=True)
    if md:
        st.markdown(md, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def chat_message(st, text: str, role: str = "assistant") -> None:
    """
    Render a single chat message bubble.
    role ∈ {'user','assistant','system'}. System renders like AI.
    """
    role = (role or "assistant").lower()
    css_role = "user" if role == "user" else "ai"
    label = "You" if role == "user" else ("System" if role == "system" else "Assistant")
    st.markdown(
        f"<div class='chat'><div class='msg {css_role}'><div class='role'>{label}</div>{text}</div></div>",
        unsafe_allow_html=True,
    )

def button_row(st, buttons: List[Tuple[str, str]]) -> Optional[str]:
    """
    Render a row of equal-width buttons. Returns the key of the clicked button or None.
    """
    if not buttons:
        return None
    cols = st.columns([1] * len(buttons))
    clicked: Optional[str] = None
    for i, (label, key) in enumerate(buttons):
        if cols[i].button(label, key=f"row_{key}"):
            clicked = key
    return clicked

def alert(st, kind: str, text: str) -> None:
    """
    Simple alert using built-in Streamlit calls with consistent spacing.
    kind ∈ {'info','success','warning','error'}
    """
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    elif kind == "error":
        st.error(text)
    else:
        st.info(text)

def hr(st) -> None:
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

def recent_upload_preview(st, uploads: List[Dict[str, Any]], limit: int = 3) -> None:
    """
    Displays compact previews for recent uploads (filename, slide count, first slide text/image if present).
    """
    if not uploads:
        st.markdown("<div class='small-muted'>No uploads yet — use the Upload tab.</div>", unsafe_allow_html=True)
        return

    for up in list(uploads)[-limit:][::-1]:
        fname = up.get("filename", "(file)")
        slides = up.get("slides_data") or []
        slide_count = up.get("slide_count", 0) or len(slides)
        st.markdown(f"**{fname}** — slides: {slide_count}")
        if slides:
            s0 = slides[0] or {}
            txt = (s0.get("text") or "")[:800]
            if txt:
                st.write(txt + ("..." if len(s0.get("text") or "") > 800 else ""))
        st.markdown("---")

def uploader(st, label: str = "Upload PDFs or PPTX", types: Optional[List[str]] = None, key: str = "uploader"):
    """
    A wrapped file uploader with sensible defaults and short helper text.
    """
    if types is None:
        types = ["pdf", "pptx"]
    files = st.file_uploader(label, type=types, accept_multiple_files=True, key=key)
    st.caption("We process text and images; OCR is best-effort if available.")
    return files

def pagination(st, total: int, key_prefix: str = "pg") -> int:
    """
    Basic page selector for lists/grids. Returns 0-based index.
    """
    if total <= 1:
        return 0
    idx = st.session_state.get(f"{key_prefix}_idx", 0)
    cols = st.columns([1, 1, 1])
    prev = cols[0].button("Prev", key=f"{key_prefix}_prev", disabled=(idx <= 0))
    cols[1].markdown(f"<div class='badge'>Page {idx+1} / {total}</div>", unsafe_allow_html=True)
    nxt = cols[2].button("Next", key=f"{key_prefix}_next", disabled=(idx >= total-1))
    if prev: idx = max(0, idx - 1)
    if nxt: idx = min(total - 1, idx + 1)
    st.session_state[f"{key_prefix}_idx"] = idx
    return idx

def apply_minimal_layout(st) -> None:
    """
    Optional layout refinements for margins/containers on pages with heavy content.
    """
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
