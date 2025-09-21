# SlideTutor — Improved version with EasyOCR (no Tesseract binary required)
# UI/UX refreshed: single-page (no sidebar), top tabs, modern CSS, subtle animations
# Save this file and run with: streamlit run slidetutor_improved_easyocr_ui.py

import os
import io
import sys
import time
import json
import math
import base64
import tempfile
import traceback
import sqlite3
from typing import List, Dict, Tuple, Optional, Any

try:
    import streamlit as st
except Exception as e:
    raise RuntimeError("streamlit must be installed. pip install streamlit") from e

# Optional libraries
_HAS_FAISS = True
try:
    import faiss
except Exception:
    _HAS_FAISS = False

_HAS_PPTX = True
try:
    from pptx import Presentation
except Exception:
    _HAS_PPTX = False

_HAS_PYMUPDF = True
try:
    import fitz
except Exception:
    _HAS_PYMUPDF = False

# Use EasyOCR (pure-python OCR) instead of pytesseract system binary
_HAS_EASYOCR = True
try:
    import easyocr
    from PIL import Image
except Exception:
    _HAS_EASYOCR = False
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("Pillow must be installed. pip install pillow")

_HAS_SENTENCE_TRANSFORMERS = True
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False
    try:
        import numpy as np
    except Exception:
        raise RuntimeError("numpy is required. pip install numpy")

import requests

_HAS_GTTS = True
try:
    from gtts import gTTS
except Exception:
    _HAS_GTTS = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# ------------------------------
# Configuration
# ------------------------------
DEFAULT_OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_KEY)
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
DB_PATH = os.getenv("SLIDETUTOR_DB", "slidetutor.sqlite3")

APP_TITLE = "SlideTutor — Student PPT/PDF Tutor (Improved)"
APP_SUBTITLE = "Upload slides and get deep multi-level lessons, quizzes, flashcards and spaced repetition."

# ------------------------------
# Helpers
# ------------------------------

def log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


def safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# ------------------------------
# DB
# ------------------------------

def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            uploaded_at INTEGER,
            meta TEXT
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER,
            question TEXT,
            answer TEXT,
            easiness REAL DEFAULT 2.5,
            interval INTEGER DEFAULT 1,
            repetitions INTEGER DEFAULT 0,
            next_review INTEGER
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER,
            question TEXT,
            options TEXT,
            correct_index INTEGER,
            created_at INTEGER
        )
        """
    )
    conn.commit()
    return conn

_db_conn = init_db()

def upload_db_id(upload: Dict) -> int:
    """
    Return the DB uploads.id for this in-memory upload object.
    If the upload has a 'db_id' (set after insertion) return it,
    otherwise fall back to the in-memory 'id' value (timestamp).
    """
    try:
        return int(upload.get("db_id") or upload.get("id"))
    except Exception:
        return int(upload.get("id"))

# ------------------------------
# Embedding Index (improvements)
# ------------------------------
class VectorIndexFallback:
    def __init__(self, embeddings: np.ndarray, texts: List[str]):
        self.embeddings = embeddings.astype(np.float32) if embeddings is not None and embeddings.size else np.zeros((0, 1), dtype=np.float32)
        self.texts = texts or []
        self._norms = None
        if self.embeddings.size:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            self._norms = norms

    def search(self, q_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        k = max(1, int(k))
        if self.embeddings.size == 0:
            return np.full((1, k), 1e9), np.full((1, k), -1, dtype=int)
        q = q_emb
        if _HAS_SKLEARN:
            sims = cosine_similarity(q, self.embeddings)
        else:
            q_norm = np.linalg.norm(q, axis=1, keepdims=True)
            q_norm[q_norm == 0] = 1e-9
            sims = (q @ self.embeddings.T) / (q_norm * self._norms.T)
        idxs = np.argsort(-sims, axis=1)[:, :k]
        dists = -np.take_along_axis(sims, idxs, axis=1)
        return dists, idxs


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, texts: List[str]):
        self.embeddings = embeddings
        self.texts = texts
        if _HAS_FAISS and embeddings is not None and embeddings.shape[0] > 0:
            d = embeddings.shape[1]
            try:
                normed = embeddings.copy()
                norms = np.linalg.norm(normed, axis=1, keepdims=True)
                norms[norms == 0] = 1e-9
                normed = normed / norms
                self.index = faiss.IndexFlatIP(d)
                self.index.add(normed.astype(np.float32))
                self._use_faiss = True
            except Exception as e:
                log("FAISS build failed, falling back:", e)
                self.index = VectorIndexFallback(embeddings, texts)
                self._use_faiss = False
        else:
            self.index = VectorIndexFallback(embeddings, texts)
            self._use_faiss = False

    def search(self, q_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self._use_faiss:
            q = q_emb.copy()
            qnorm = np.linalg.norm(q, axis=1, keepdims=True)
            qnorm[qnorm == 0] = 1e-9
            q = q / qnorm
            D, I = self.index.search(q.astype(np.float32), k)
            return -D, I
        else:
            return self.index.search(q_emb, k)


@st.cache_resource
def load_sentence_transformer(model_name: str = EMBEDDING_MODEL_NAME):
    if not _HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError("sentence-transformers is required for embeddings. pip install sentence-transformers")
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return arr.astype(np.float32)

# ------------------------------
# Parsers
# ------------------------------

def extract_from_pptx_bytes(file_bytes: bytes) -> List[Dict]:
    slides = []
    if not _HAS_PPTX:
        return [{"index": 0, "text": "[python-pptx not installed]", "images": []}]
    try:
        prs = Presentation(io.BytesIO(file_bytes))
        for i, slide in enumerate(prs.slides):
            texts = []
            images = []
            for shape in slide.shapes:
                try:
                    if hasattr(shape, "text") and shape.text and shape.text.strip():
                        texts.append(shape.text.strip())
                except Exception:
                    pass
                try:
                    if hasattr(shape, "image"):
                        img = shape.image
                        if img is not None:
                            image_bytes = img.blob
                            if image_bytes:
                                images.append(image_bytes)
                except Exception:
                    pass
            slides.append({"index": i, "text": "\n".join(texts).strip(), "images": images})
        return slides
    except Exception as e:
        log("pptx parse error:", e)
        return [{"index": 0, "text": f"[pptx parse error] {e}", "images": []}]


def extract_from_pdf_bytes(file_bytes: bytes) -> List[Dict]:
    slides = []
    if not _HAS_PYMUPDF:
        return [{"index": 0, "text": "[pymupdf not installed, can't parse PDF]", "images": []}]
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = page.get_text("text").strip()
            images = []
            # try extracting embedded images
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    if image_bytes:
                        images.append(image_bytes)
                except Exception:
                    pass

            # If page has no selectable text and no embedded images, render page to an image
            if (not text) and (not images):
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    try:
                        img_bytes = pix.tobytes(output="png")
                    except TypeError:
                        img_bytes = pix.tobytes()
                    images.append(img_bytes)
                except Exception as e:
                    log(f"PDF page rendering failed for page {i}: {e}")

            slides.append({"index": i, "text": text, "images": images})
        doc.close()
        return slides
    except Exception as e:
        log("pdf parse error:", e)
        return [{"index": 0, "text": f"[pdf parse error] {e}", "images": []}]


# EasyOCR reader factory (cached)
@st.cache_resource
def get_easyocr_reader(lang_list: List[str] = ['en']):
    if not _HAS_EASYOCR:
        return None
    try:
        reader = easyocr.Reader(lang_list, gpu=False)
        return reader
    except Exception as e:
        log("easyocr init failed:", e)
        return None


def ocr_image_bytes_list(image_bytes_list: List[bytes]) -> List[str]:
    results = []
    reader = get_easyocr_reader(['en'])
    for b in image_bytes_list:
        if not b:
            results.append("")
            continue
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            if _HAS_EASYOCR and reader is not None:
                arr = np.array(img)
                try:
                    raw = reader.readtext(arr, detail=0, paragraph=True)
                except TypeError:
                    raw = reader.readtext(arr, detail=0)
                if isinstance(raw, list):
                    txt = "\n".join([r for r in raw if r])
                else:
                    txt = str(raw)
                results.append(txt.strip())
            else:
                results.append("")
        except Exception as e:
            log("OCR error:", e)
            results.append("")
    return results


def chunk_text(text: str, max_chars: int = 700) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    parts = []
    cur = ""
    for p in paragraphs:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur = cur + "\n" + p
        else:
            parts.append(cur)
            cur = p
    if cur:
        parts.append(cur)
    final_parts = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
        else:
            sents = part.split(". ")
            cur2 = ""
            for s in sents:
                piece = s if s.endswith('.') else s + '.'
                if not cur2:
                    cur2 = piece
                elif len(cur2) + 1 + len(piece) <= max_chars:
                    cur2 = cur2 + ' ' + piece
                else:
                    final_parts.append(cur2.strip())
                    cur2 = piece
            if cur2:
                final_parts.append(cur2.strip())
    return final_parts

# ------------------------------
# OpenRouter helpers
# ------------------------------

def call_openrouter_chat(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini",
                         max_tokens: int = 512, temperature: float = 0.2) -> str:
    api_key = st.session_state.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_KEY)
    if not api_key:
        return "[OpenRouter error] No API key configured. Set it in Settings or env var OPENROUTER_API_KEY."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=30)
        text = resp.text
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            msg = first.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if not content:
                content = first.get("text")
            if not content:
                content = json.dumps(first)
            return content
        if "result" in data:
            return str(data["result"])
        return "[openrouter] empty response"
    except Exception as e:
        try:
            return f"[OpenRouter error] {e} — resp_text={text}"
        except Exception:
            return f"[OpenRouter error] {e}"


def extract_json_from_text(text: str) -> Optional[Any]:
    if not text:
        return None
    cleaned = text.strip()
    for fence in ['```json', '```']:
        cleaned = cleaned.replace(fence, '')
    import re
    m = re.search(r'([\{\[])', cleaned)
    if not m:
        return None
    start = m.start(1)
    for end in range(len(cleaned), start, -1):
        try:
            candidate = cleaned[start:end]
            parsed = json.loads(candidate)
            return parsed
        except Exception:
            continue
    arr_match = re.search(r'\[.*\]', cleaned, flags=re.S)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except Exception:
            pass
    obj_match = re.search(r'\{.*\}', cleaned, flags=re.S)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass
    return None

# ------------------------------
# Generators
# ------------------------------

def generate_multilevel_lesson(slide_text: str, related_texts: str = "") -> str:
    system = (
        "You are a patient expert teacher. Produce three sections titled 'Beginner', 'Intermediate', and 'Advanced'. "
        "For each section: give a clear explanation, one small worked example, and 2-3 concise tips. Finally, produce "
        "3 multiple-choice questions (with answers) labelled under 'MCQs:'. Keep the tone friendly and short."
    )
    user_prompt = f"SLIDE_TEXT:\n{slide_text}\n\nRELATED:\n{related_texts}\n\nRespond with sections as requested."
    return call_openrouter_chat(system, user_prompt, max_tokens=900, temperature=0.15)


def generate_deep_lesson(slide_text: str, related_texts: str = "") -> str:
    system = (
        "You are an expert university instructor teaching the topic in depth to a BTech 2nd-year mechanical engineering student. "
        "Start from first principles in very simple language, then progressively increase difficulty. For each level include: (1) concept explanation, "
        "(2) 2 worked examples (one numerical, one conceptual), (3) common mistakes, (4) 5 short practice MCQs with answers, and (5) 8 concise flashcard-style Q/A pairs. "
        "Label sections clearly: 'Foundations', 'Developing', 'Advanced'. Keep numerical steps explicit and provide units where applicable."
    )
    user_prompt = f"SLIDE_TEXT:\n{slide_text}\n\nRELATED:\n{related_texts}\n\nRespond with detailed lesson as requested."
    return call_openrouter_chat(system, user_prompt, model="gpt-4o-mini", max_tokens=2500, temperature=0.05)


def generate_mcq_set_from_text(text: str, qcount: int = 5) -> List[Dict]:
    system = "You are an assistant that generates high-quality multiple-choice questions from source text. Reply JSON only."
    user_prompt = (
        f"Create {qcount} multiple-choice questions based solely on the following text. For each question provide: "
        f"'question' (string), 'options' (array of 4 strings), and 'answer_index' (0-based integer). Reply ONLY with a JSON array.\n\nTEXT:\n{text}\n"
    )
    resp = call_openrouter_chat(system, user_prompt, max_tokens=700, temperature=0.0)
    parsed = extract_json_from_text(resp)
    if isinstance(parsed, list):
        return parsed
    return [{"question": "Summarize the main idea in one sentence.", "options": ["A", "B", "C", "D"], "answer_index": 0}]


def generate_flashcards_from_text(text: str, n: int = 10) -> List[Dict]:
    system = "Extract concise Q/A flashcards from the source text. Reply only in JSON array."
    user_prompt = f"Extract up to {n} short Q/A flashcards from the text below. Keep Q under 80 chars and A under 200 chars. Reply ONLY in JSON.\n\nTEXT:\n{text}"
    resp = call_openrouter_chat(system, user_prompt, max_tokens=700, temperature=0.0)
    parsed = extract_json_from_text(resp)
    if isinstance(parsed, list):
        return parsed
    return []

# ------------------------------
# SM-2
# ------------------------------

def sm2_update_card(easiness: float, interval: int, repetitions: int, quality: int) -> Tuple[float, int, int, int]:
    q = quality
    if q < 3:
        repetitions = 0
        interval = 1
    else:
        repetitions += 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = max(1, int(round(interval * easiness)))
    easiness = max(1.3, easiness + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    next_review = int(time.time()) + int(interval) * 24 * 3600
    return easiness, interval, repetitions, next_review

# ------------------------------
# Export helpers
# ------------------------------

def anki_export_csv_for_upload(upload_id: int, conn: sqlite3.Connection) -> Tuple[str, bytes]:
    c = conn.cursor()
    c.execute("SELECT question, answer FROM flashcards WHERE upload_id = ?", (upload_id,))
    rows = c.fetchall() or []
    lines = []
    for q, a in rows:
        q2 = (q or "").replace("\t", " ").replace("\n", " ")
        a2 = (a or "").replace("\t", " ").replace("\n", " ")
        lines.append(f"{q2}\t{a2}")
    csv_bytes = "\n".join(lines).encode("utf-8")
    fname = f"slide_tutor_upload_{upload_id}_flashcards.txt"
    return fname, csv_bytes



def text_to_speech_download(text: str, lang: str = "en") -> Tuple[str, bytes]:
    if not _HAS_GTTS:
        raise RuntimeError("gTTS not installed (pip install gTTS)")
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    tts.save(tmp.name)
    with open(tmp.name, "rb") as fh:
        b = fh.read()
    os.unlink(tmp.name)
    return "lesson_audio.mp3", b

# ------------------------------
# Streamlit UI — single page with top tabs and improved CSS
# ------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
if "OPENROUTER_API_KEY" not in st.session_state:
    st.session_state["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_KEY)

# modern CSS + animations
st.markdown("""
<style>
:root{
  --bg:#071022;
  --card: rgba(255,255,255,0.03);
  --muted:#9aa6b2;
  --accent1: linear-gradient(135deg,#7b61ff,#3ad29f);
}
html, body, .stApp { background: linear-gradient(180deg, rgba(8,12,20,1), rgba(3,6,12,1)); color: #dbe9f7; }
/* Animated decorative blobs */
.blob{ position:absolute; width:360px; height:360px; filter: blur(70px); opacity:0.18; z-index:0; transform:translate3d(0,0,0); }
.blob.one{ background: radial-gradient(circle at 20% 30%, #7b61ff, transparent 40%); top:-60px; left:-60px; animation: floaty 9s ease-in-out infinite; }
.blob.two{ background: radial-gradient(circle at 80% 70%, #19d3a7, transparent 40%); right:-80px; bottom:-80px; animation: floaty 12s ease-in-out infinite reverse; }
@keyframes floaty{ 0%{transform:translateY(0) rotate(0deg);} 50%{transform:translateY(16px) rotate(6deg);} 100%{transform:translateY(0) rotate(0deg);} }

/* Top nav as tabs */
.topbar{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:18px 28px; position:relative; z-index:2; }
.brand{ display:flex; align-items:center; gap:14px; }
.logo{ width:48px; height:48px; border-radius:10px; background: linear-gradient(135deg,#7b61ff,#3ad29f); display:flex; align-items:center; justify-content:center; font-weight:700; color:white; box-shadow: 0 6px 20px rgba(0,0,0,0.5); }
.title{ font-size:20px; font-weight:700; }
.subtitle{ color:var(--muted); font-size:12px; margin-top:2px; }

.tabs-row{ display:flex; gap:8px; align-items:center; }
.tab-btn{ padding:10px 14px; border-radius:10px; background:transparent; color:var(--muted); border:1px solid transparent; transition:all .18s ease; font-weight:600; }
.tab-btn.active{ background:var(--card); color:#eaf6ff; border-color: rgba(255,255,255,0.04); box-shadow: 0 6px 18px rgba(6,8,12,0.7); }
.tab-btn:hover{ transform:translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.6); }

/* card style */
.card{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:18px; border-radius:14px; border:1px solid rgba(255,255,255,0.03); }
.hero{ padding:28px; border-radius:14px; margin-bottom:16px; }
.small-muted{ color:var(--muted); font-size:13px; }
.slide-box{ background:#071019; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }
.mono{ font-family: monospace; }

/* style streamlit buttons a bit */
div.stButton > button, button[kind="primary"]{ background: linear-gradient(90deg,#7b61ff,#3ad29f) !important; color: #021018 !important; font-weight:700; border:none !important; padding:10px 16px !important; border-radius:10px !important; }

/* small responsive tweaks */
@media (max-width: 900px){ .topbar{ flex-direction:column; align-items:flex-start; gap:8px; } .tabs-row{ flex-wrap:wrap; } }
</style>
""", unsafe_allow_html=True)

# Decorative blobs (absolute positioned)
st.markdown("<div class='blob one'></div><div class='blob two'></div>", unsafe_allow_html=True)

# Top bar (brand + tabs rendered using streamlit components)
with st.container():
    cols = st.columns([0.4, 3, 2])
    with cols[0]:
        st.markdown("<div class='logo'>ST</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div class='title'>%s</div><div class='subtitle'>%s</div>" % (APP_TITLE, APP_SUBTITLE), unsafe_allow_html=True)
    with cols[2]:
        st.markdown("<div style='text-align:right'><small class='small-muted'>Student edition • Improved</small></div>", unsafe_allow_html=True)

# top tabs — all on homepage, no sidebar
TAB_NAMES = ["Home", "Upload & Process", "Lessons", "Chat Q&A", "Quizzes", "Flashcards", "Export", "Progress", "Settings"]
# ---- Replace st.tabs with a controllable top nav so quick-action buttons can switch tabs ----
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Home"

# Render top tab buttons (simple, accessible — you can style them with CSS above)
tab_cols = st.columns(len(TAB_NAMES))
for i, name in enumerate(TAB_NAMES):
    if tab_cols[i].button(name, key=f"top_tab_{name}"):
        st.session_state["active_tab"] = name

active_tab = st.session_state["active_tab"]
# ------------------------------------------------------------------------------------------------


# Shared session uploads container
if "uploads" not in st.session_state:
    st.session_state["uploads"] = []

# helper to build index (unchanged)
def build_index_for_upload(upload: Dict):
    model = load_sentence_transformer()
    chunks = upload.get("chunks", [])
    if not chunks:
        upload["embeddings"] = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
        upload["index"] = VectorIndex(upload["embeddings"], chunks)
        return upload
    emb = embed_texts(model, chunks)
    upload["embeddings"] = emb
    upload["index"] = VectorIndex(emb, chunks)
    return upload

# ------------------------------
# Home tab
# ------------------------------
if active_tab == "Home":

    st.markdown("<div class='card hero'>", unsafe_allow_html=True)
    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown("<h2 style='margin:0;padding:0'>Welcome to <span style='color:#a79bff'>SlideTutor</span></h2>", unsafe_allow_html=True)
        st.markdown("<p class='small-muted'>Upload any PPTX or PDF and generate lessons, quizzes, flashcards, and more — all on this page.</p>", unsafe_allow_html=True)
        st.markdown("<ul class='small-muted'><li>Scanned PDFs supported (EasyOCR)</li><li>Auto-generated quizzes & flashcards</li><li>Spaced repetition (SM-2)</li></ul>", unsafe_allow_html=True)
        st.write("")
        st.markdown("<div class='small-muted'>Quick actions</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Upload & Process", key="quick_upload"):
                st.session_state["active_tab"] = "Upload & Process"
        with c2:
            if st.button("Generate Lesson", key="quick_lesson"):
                st.session_state["active_tab"] = "Lessons"
        with c3:
            if st.button("Practice Flashcards", key="quick_flash"):
                st.session_state["active_tab"] = "Flashcards"

    with cols[1]:
        st.markdown("<div class='card' style='padding:12px;text-align:center'>\n<h3 style='margin-top:2px'>Usage Tip</h3>\n<p class='small-muted' style='font-size:13px'>Start by uploading your file in the 'Upload & Process' tab. Then explore Lessons / Quizzes / Flashcards generated automatically.</p>\n</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# ------------------------------
# Upload & Process (robust replacement)
# ------------------------------
elif active_tab == "Upload & Process":

    st.header("Upload PPTX / PDF (Student Upload)")

    uploaded_file = st.file_uploader("Choose a PPTX or PDF file", type=["pptx", "pdf"], accept_multiple_files=False, key="uploader_file")
    if uploaded_file is not None:
        # read bytes safely
        try:
            raw_bytes = uploaded_file.read()
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            raw_bytes = b""

        # basic metadata
        fname = getattr(uploaded_file, "name", "uploaded_file")
        try:
            file_size = getattr(uploaded_file, "size", len(raw_bytes))
        except Exception:
            file_size = len(raw_bytes)

        # warn on very large uploads (but still attempt)
        MB = 1024 * 1024
        if file_size > 100 * MB:
            st.warning("Large file (>100MB) detected. Processing may be slow or run out of memory.")

        st.info(f"Processing {fname} ...")

        # wrap overall processing to avoid crashing UI
        try:
            # extract slides depending on type
            slides = []
            if fname.lower().endswith(".pptx"):
                if not _HAS_PPTX:
                    st.error("python-pptx not installed; cannot parse PPTX. Install python-pptx to enable this feature.")
                    slides = [{"index": 0, "text": "[python-pptx not installed]", "images": []}]
                else:
                    try:
                        slides = extract_from_pptx_bytes(raw_bytes)
                    except Exception as e:
                        log("pptx extraction error:", e)
                        slides = [{"index": 0, "text": f"[pptx parse error] {e}", "images": []}]
            elif fname.lower().endswith(".pdf"):
                if not _HAS_PYMUPDF:
                    st.error("pymupdf not installed; cannot parse PDF. Install pymupdf to enable full PDF parsing.")
                    slides = [{"index": 0, "text": "[pymupdf not installed, can't parse PDF]", "images": []}]
                else:
                    try:
                        slides = extract_from_pdf_bytes(raw_bytes)
                    except Exception as e:
                        log("pdf extraction error:", e)
                        slides = [{"index": 0, "text": f"[pdf parse error] {e}", "images": []}]
            else:
                slides = [{"index": 0, "text": "[Unsupported file type]", "images": []}]

            # OCR any slide images (only if easyocr available; keep resilient)
            with st.spinner("Running OCR on slide images (if any)..."):
                for si, s in enumerate(slides):
                    imgs = s.get("images") or []
                    if imgs:
                        try:
                            ocr_texts = ocr_image_bytes_list(imgs)
                            appended = "\n".join([t for t in ocr_texts if t])
                            if appended:
                                s["text"] = (s.get("text", "") + "\n\n" + appended).strip()
                        except Exception as e:
                            log(f"OCR failure on slide {si}:", e)

            # chunk text into manageable pieces
            chunks = []
            mapping = []
            for s in slides:
                try:
                    parts = chunk_text(s.get("text", ""))
                except Exception as e:
                    log("chunk_text failed for slide", s.get("index"), e)
                    parts = [s.get("text", "") or ""]
                for p in parts:
                    chunks.append(p)
                    mapping.append({"slide": s["index"], "text": p})

            # build upload object; use millisecond timestamp as id (keeps compatibility)
            upload_obj = {
                "id": int(time.time() * 1000),
                "filename": fname,
                "uploaded_at": int(time.time()),
                "slides": slides,
                "chunks": chunks,
                "mapping": mapping
            }

            # attempt to build embeddings/index but fail gracefully if models missing
            with st.spinner("Creating embeddings and index (this may take a while)..."):
                try:
                    upload_obj = build_index_for_upload(upload_obj)
                except Exception as e:
                    log("Index build failed; continuing without embeddings:", e)
                    # ensure a fallback index exists
                    upload_obj["embeddings"] = np.zeros((0, 1), dtype=np.float32)
                    upload_obj["index"] = VectorIndex(upload_obj["embeddings"], upload_obj.get("chunks", []))

            # persist upload metadata in DB and capture DB id (defensive)
            try:
                cur = _db_conn.cursor()
                meta = {"n_slides": len(slides), "n_chunks": len(chunks), "file_size": file_size}
                cur.execute(
                    "INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)",
                    (fname, upload_obj["uploaded_at"], json.dumps(meta))
                )
                _db_conn.commit()
                db_id = cur.lastrowid
                upload_obj["db_id"] = int(db_id)
            except Exception as e:
                log("Could not save upload metadata to DB:", e)
                # still keep going; mark db_id None
                upload_obj["db_id"] = None

            # store in session (in-memory) but avoid duplicates (same filename & size)
            try:
                # remove previous identical upload (filename + size) to avoid clutter
                existing = None
                for u in st.session_state["uploads"]:
                    if u.get("filename") == upload_obj["filename"] and len(u.get("slides", [])) == len(upload_obj.get("slides", [])):
                        existing = u
                        break
                if existing:
                    # replace existing
                    st.session_state["uploads"].remove(existing)
                st.session_state["uploads"].append(upload_obj)
            except Exception as e:
                log("Could not append upload to session:", e)
                st.session_state["uploads"].append(upload_obj)

            # keep the user on the Upload tab so they can preview and act
            st.session_state["active_tab"] = "Upload & Process"

            st.success(f"Upload processed: {len(slides)} slides/pages, {len(chunks)} chunks.")

            # show a compact info row and download option
            info_cols = st.columns([3, 1])
            with info_cols[0]:
                st.markdown(f"**Filename:** {fname}  •  **Slides:** {len(slides)}  •  **Chunks:** {len(chunks)}")
            with info_cols[1]:
                try:
                    json_bytes = json.dumps(slides, ensure_ascii=False, indent=2).encode("utf-8")
                    st.download_button("Download extracted slides (JSON)", json_bytes, file_name=f"{fname}_extracted.json", mime="application/json", key=f"dljson_{upload_obj['id']}")
                except Exception as e:
                    log("Could not prepare download button:", e)

            # ---------- Preview: Text chunks or Slide viewer ----------
            # unique key uses db_id or id to avoid duplication
            preview_key = f"view_mode_{upload_obj.get('db_id') or upload_obj.get('id')}"
            view_mode = st.radio("Preview mode", ["Text (chunks)", "Slide Viewer"], index=0, key=preview_key)
            if view_mode == "Text (chunks)":
                with st.expander("Preview first 10 chunks (expand to view)"):
                    st.write("Preview first 10 chunks:")
                    for i, c in enumerate(chunks[:10]):
                        st.code(c[:1200] + ("..." if len(c) > 1200 else ""))
            else:
                st.write("Slide viewer — first 10 slides")
                for s in upload_obj.get("slides", [])[:10]:
                    st.markdown(f"**Slide {s['index']}**")
                    imgs = s.get("images") or []
                    if imgs:
                        # try to render each image robustly
                        rendered = False
                        for img_idx, img_bytes in enumerate(imgs[:2]):  # try first two images per slide
                            if not img_bytes:
                                continue
                            try:
                                # st.image accepts bytes or PIL.Image; try bytes directly first
                                st.image(img_bytes, use_column_width=True, caption=f"Slide {s['index']} — image {img_idx}", clamp=True)
                                rendered = True
                                break
                            except Exception:
                                try:
                                    from PIL import Image as PILImage
                                    pil_img = PILImage.open(io.BytesIO(img_bytes))
                                    st.image(pil_img, use_column_width=True, caption=f"Slide {s['index']} — image {img_idx}")
                                    rendered = True
                                    break
                                except Exception as e:
                                    log(f"Failed to render slide image (slide {s['index']}, img {img_idx}):", e)
                                    rendered = False
                        if not rendered:
                            st.info("Could not render slide image(s); showing extracted text instead.")
                            st.write(s.get("text", ""))
                    else:
                        # no image: show extracted text as fallback
                        st.info("No image available for this slide — showing extracted text")
                        st.write(s.get("text", ""))

            # ---------------------------------------------------------

        except Exception as e:
            # top-level error handling so UI doesn't crash
            st.error(f"Unexpected error during upload processing: {e}")
            log("Unhandled upload processing error:", e, traceback.format_exc())


# ------------------------------
# Lessons
# ------------------------------
elif active_tab == "Lessons":

    st.header("Generate Multi-level Lessons")
    if not st.session_state["uploads"]:
        st.info("No uploads yet. Go to Upload & Process.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_lessons")
        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if upload is None:
            st.warning("Selected upload not found.")
        else:
            slides = upload.get("slides", [])
            max_idx = max([s["index"] for s in slides]) if slides else 0
            slide_idx = st.number_input("Slide/Page index", min_value=0, max_value=max_idx, value=0)
            slide_text = next((s.get("text", "") for s in slides if s["index"] == slide_idx), "")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader(f"Slide/Page {slide_idx} — preview")
            st.write(slide_text if len(slide_text) < 4000 else slide_text[:4000] + "...")
            st.markdown("</div>", unsafe_allow_html=True)

            deep = st.checkbox("Produce a deeply detailed lesson (longer, step-by-step)", value=False)
            auto_create = st.checkbox("Also auto-create quiz + flashcards and save to DB", value=True)

            if st.button("Generate Lesson (Beginner→Advanced)"):
                idx = upload.get("index")
                related = ""
                snippets = []
                if idx and upload.get("chunks"):
                    try:
                        model = load_sentence_transformer()
                        q_emb = embed_texts(model, [slide_text])
                        D, I = idx.search(q_emb, TOP_K)
                        for j in I[0]:
                            if isinstance(j, int) and 0 <= j < len(upload["chunks"]):
                                slide_num = upload["mapping"][j]["slide"] if j < len(upload["mapping"]) else None
                                prefix = f"[Slide {slide_num}] " if slide_num is not None else ""
                                snippets.append(prefix + upload["chunks"][j])
                    except Exception as e:
                        log("Index search failed:", e)
                related = "\n\n".join(snippets)
                with st.spinner("Generating lesson via OpenRouter..."):
                    if deep:
                        lesson = generate_deep_lesson(slide_text, related)
                    else:
                        lesson = generate_multilevel_lesson(slide_text, related)
                st.subheader("Generated Lesson")
                st.markdown(lesson)

                if auto_create:
                    st.info("Attempting to auto-generate MCQs and flashcards from lesson and saving to DB.")
                    mcqs = generate_mcq_set_from_text(lesson, qcount=8)
                    fcards = generate_flashcards_from_text(lesson, n=12)
                    cur = _db_conn.cursor()
                    try:
                        db_uid = upload_db_id(upload)
                        for q in mcqs:
                            cur.execute("INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                                        (db_uid, q.get("question", ""), json.dumps(q.get("options", [])), int(q.get("answer_index", 0)), int(time.time())))
                        inserted = 0
                        for card in fcards:
                            qtext = card.get("q") or card.get("question") or ""
                            atext = card.get("a") or card.get("answer") or ""
                            if qtext and atext:
                                cur.execute("INSERT INTO flashcards (upload_id, question, answer, easiness, interval, repetitions, next_review) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                            (db_uid, qtext, atext, 2.5, 1, 0, int(time.time())))
                                inserted += 1
                        _db_conn.commit()
                        st.success(f"Saved {len(mcqs)} MCQs and {inserted} flashcards to DB (if any).")
                    except Exception as e:
                        st.warning(f"Could not save generated artifacts: {e}")

                if _HAS_GTTS:
                    if st.button("Export lesson as MP3 (TTS)"):
                        try:
                            fname, mp3bytes = text_to_speech_download(lesson)
                            st.download_button("Download lesson audio", mp3bytes, file_name=fname, mime="audio/mpeg")
                        except Exception as e:
                            st.error(f"TTS failed: {e}")

# ------------------------------
# Chat Q&A
# ------------------------------
elif active_tab == "Chat Q&A":

    st.header("Ask questions about your upload (Retrieval + LLM)")
    if not st.session_state["uploads"]:
        st.info("No uploads yet. Upload files first.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_chat")

        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if upload:
            question = st.text_area("Ask a question about the slides/pages")
            if st.button("Get Answer") and question.strip():
                model = load_sentence_transformer()
                q_emb = embed_texts(model, [question])
                idx = upload.get("index")
                top_ctx = []
                if idx:
                    D, I = idx.search(q_emb, TOP_K)
                    for j in I[0]:
                        if isinstance(j, int) and 0 <= j < len(upload["chunks"]):
                            slide_num = upload["mapping"][j]["slide"] if j < len(upload["mapping"]) else None
                            prefix = f"[Slide {slide_num}] " if slide_num is not None else ""
                            top_ctx.append(prefix + upload["chunks"][j])
                top_ctx_text = "\n\n".join(top_ctx)
                system = "You are a helpful tutor. Use the provided context to answer concisely; cite slide/page indices if possible."
                prompt = f"CONTEXT:\n{top_ctx_text}\n\nQUESTION:\n{question}\n\nAnswer concisely and provide one short example or analogy."
                with st.spinner("Querying OpenRouter..."):
                    ans = call_openrouter_chat(system, prompt, max_tokens=450)
                st.subheader("Answer")
                st.write(ans)
                if top_ctx:
                    st.markdown("---")
                    st.markdown("**Context used (excerpt):**")
                    for j, s in enumerate(top_ctx[:TOP_K]):
                        st.code(s[:800] + ("..." if len(s) > 800 else ""))

# ------------------------------
# Quizzes
# ------------------------------
elif active_tab == "Quizzes":

    st.header("Auto-generated Quizzes")
    if not st.session_state["uploads"]:
        st.info("No uploads yet.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_quizzes")

        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if upload:
            slides = upload.get("slides", [])
            max_idx = max([s["index"] for s in slides]) if slides else 0
            slide_idx = st.number_input("Slide/Page index to generate quiz from", min_value=0, max_value=max_idx, value=0)
            text = next((s.get("text", "") for s in slides if s["index"] == slide_idx), "")
            if st.button("Generate Quiz (MCQs)"):
                with st.spinner("Creating MCQs..."):
                    qset = generate_mcq_set_from_text(text, qcount=5)
                st.success("Quiz ready — try it below")
                for qi, q in enumerate(qset):
                    st.markdown(f"**Q{qi + 1}.** {q.get('question', '')}")
                    opts = q.get("options", [])
                    if not isinstance(opts, list) or len(opts) == 0:
                        opts = ["A", "B", "C", "D"]
                    choice = st.radio(f"Select Q{qi + 1}", opts, key=f"quiz_{sel_id}_{slide_idx}_{qi}")
                    if st.button(f"Submit Q{qi + 1}", key=f"submit_{sel_id}_{slide_idx}_{qi}"):
                        chosen_idx = opts.index(choice) if choice in opts else 0
                        correct_idx = q.get("answer_index", 0)
                        if chosen_idx == correct_idx:
                            st.success("Correct")
                        else:
                            st.error(f"Incorrect — correct answer: {opts[correct_idx]}")
                try:
                    cur = _db_conn.cursor()
                    db_uid = upload_db_id(upload)
                    for q in qset:
                        cur.execute("INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                                    (db_uid, q.get("question", ""), json.dumps(q.get("options", [])), int(q.get("answer_index", 0)), int(time.time())))
                    _db_conn.commit()

                except Exception as e:
                    st.warning(f"Could not save quiz to DB: {e}")

# ------------------------------
# Flashcards
# ------------------------------
elif active_tab == "Flashcards":

    st.header("Flashcards & Spaced Repetition")
    if not st.session_state["uploads"]:
        st.info("No uploads yet.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload to work with", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_flashcards")

        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if upload:
            slides = upload.get("slides", [])
            max_idx = max([s["index"] for s in slides]) if slides else 0
            slide_idx = st.number_input("Slide/Page index to extract flashcards from", min_value=0, max_value=max_idx, value=0)
            text = next((s.get("text", "") for s in slides if s["index"] == slide_idx), "")
            if st.button("Generate Flashcards from this slide/page"):
                with st.spinner("Generating flashcards..."):
                    cards = generate_flashcards_from_text(text, n=12)
                if not cards:
                    st.warning("No flashcards generated (OpenRouter might have returned unexpected output).")
                else:
                    cur = _db_conn.cursor()
                    inserted = 0
                    db_uid = upload_db_id(upload)
                    try:
                        for card in cards:
                            qtext = card.get("q") or card.get("question") or ""
                            atext = card.get("a") or card.get("answer") or ""
                            if qtext and atext:
                                cur.execute(
                                    "INSERT INTO flashcards (upload_id, question, answer, easiness, interval, repetitions, next_review) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (db_uid, qtext, atext, 2.5, 1, 0, int(time.time()))
                                )
                                inserted += 1
                        _db_conn.commit()

                        if inserted:
                            st.success(f"Inserted {inserted} flashcards into your deck.")
                            # preview first few inserted cards
                            st.markdown("**Preview (first 5):**")
                            for i, card in enumerate(cards[:5]):
                                qtext = card.get("q") or card.get("question") or ""
                                atext = card.get("a") or card.get("answer") or ""
                                st.markdown(f"**{i+1}.** {qtext}")
                                st.markdown(f"<div class='small-muted'>Answer: {atext}</div>", unsafe_allow_html=True)
                        else:
                            st.info("No valid Q/A pairs were found in the generated output.")
                    except Exception as e:
                        st.error(f"Failed to save flashcards: {e}")
                        log("flashcard save error:", e)
