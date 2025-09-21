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


st.markdown("""
<style>
/* =========================
   ADVANCED UI/UX THEME KIT
   - Highly configurable CSS variables
   - Glassmorphism + layered elevation
   - Subtle animated accents
   - Accessibility: focus-visible & reduced-motion
   - Streamlit-safe overrides (low-side effects)
   ========================= */

/* Uncomment to load fonts from Google (only if allowed in your environment)
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Poppins:wght@500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
*/

/* ---------- THEME TOKENS ---------- */
:root{
  /* Colors */
  --bg: #071025;
  --bg-2: #06101A;
  --card: rgba(14,27,42,0.86);
  --glass: rgba(255,255,255,0.04);
  --muted: #9AA6B2;
  --text: #E6F0FA;
  --accent: #2AB7A9;
  --accent-2: #4D7CFE;
  --danger: #FF6B6B;
  --success: #4ADE80;

  /* Typography */
  --font-sans: "Inter", "Poppins", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace;
  --base-size: 14px;
  --heading-multiplier: 1.15;
  --radius: 12px;

  /* Spacing */
  --gap-xxs: 6px;
  --gap-xs: 8px;
  --gap-sm: 12px;
  --gap-md: 20px;
  --gap-lg: 28px;
  --gap-xl: 44px;

  /* Elevation (soft shadow tokens) */
  --elev-1: 0 6px 18px rgba(2,6,12,0.32);
  --elev-2: 0 12px 28px rgba(2,6,12,0.38);
  --elev-3: 0 20px 56px rgba(2,6,12,0.48);

  /* Animations */
  --fast: 120ms;
  --mid: 240ms;
  --slow: 420ms;
  --ease: cubic-bezier(.2,.9,.2,1);
  --accent-animation-duration: 6s;
}

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {
  :root { --accent-animation-duration: 0.001ms; }
  * { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; scroll-behavior: auto !important; }
}

/* Light mode fallback for users that prefer it */
@media (prefers-color-scheme: light) {
  :root {
    --bg: #F6F8FB;
    --bg-2: #EEF2F8;
    --card: rgba(255,255,255,0.96);
    --glass: rgba(11,22,48,0.03);
    --muted: #516177;
    --text: #071025;
  }
}

/* ---------- SCOPE WRAPPER ----------
   To avoid collisions with other CSS (Streamlit internals),
   place your app content inside:
     st.markdown('<div class="app-theme">', unsafe_allow_html=True)
     ... your Streamlit content ...
     st.markdown('</div>', unsafe_allow_html=True)

   If you prefer global application, remove the `.app-theme` prefix from selectors.
------------------------------------- */
.app-theme, .app-theme * { box-sizing: border-box; font-family: var(--font-sans); color: var(--text); }

/* Page background + base */
.app-theme body, .app-theme .stApp {
  background: linear-gradient(180deg, var(--bg), var(--bg-2)) fixed;
  margin: 0;
  padding: 0;
  font-size: var(--base-size);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  min-height: 100vh;
}

/* Animated ambient gradient (subtle, behind everything) */
.app-theme::before{
  content: "";
  position: fixed;
  inset: -20% -10% auto -10%;
  height: 60vh;
  background: radial-gradient(800px 400px at 10% 10%, rgba(77,124,254,0.12), transparent 10%),
              radial-gradient(700px 500px at 90% 90%, rgba(42,183,169,0.08), transparent 8%);
  pointer-events: none;
  filter: blur(36px);
  z-index: 0;
  transform: translateZ(0);
  animation: ambientShift var(--accent-animation-duration) linear infinite;
  mix-blend-mode: soft-light;
}
@keyframes ambientShift {
  0% { transform: translateX(0) translateY(0) rotate(0deg); }
  50% { transform: translateX(20px) translateY(-12px) rotate(0.3deg); }
  100% { transform: translateX(0) translateY(0) rotate(0deg); }
}

/* ---------- LAYOUT HELPERS ---------- */
.app-theme .container { position: relative; z-index: 2; padding: var(--gap-md); }
.app-theme .row { display:flex; gap:var(--gap-md); align-items:center; }
.app-theme .col { display:block; }
.app-theme .stack { display:flex; flex-direction:column; gap:var(--gap-sm); }

/* Visually-hidden helper for accessibility */
.app-theme .visually-hidden {
  position:absolute !important; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; border:0;
}

/* ---------- TOPBAR / BRAND ---------- */
.app-theme .topbar {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:var(--gap-sm);
  padding:14px 18px;
  position: relative;
  z-index: 3;
}
.app-theme .brand { display:flex; gap:12px; align-items:center; }
.app-theme .logo {
  width:48px; height:48px;
  border-radius: 12px;
  display:flex; align-items:center; justify-content:center; font-weight:700; color:white;
  background: linear-gradient(135deg, var(--accent-2), var(--accent));
  box-shadow: var(--elev-2);
  transform-origin: center;
  transition: transform var(--fast) var(--ease);
}
.app-theme .logo:active { transform: scale(.98) rotate(-1deg); }
.app-theme .brand .title { font-weight:700; font-size: calc(var(--base-size) * 1.35); line-height:1; }
.app-theme .brand .subtitle { color: var(--muted); font-size: calc(var(--base-size) * 0.92); margin-top: 2px; }

/* ---------- NAV / TABS ---------- */
.app-theme .tabs { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.app-theme .tab {
  padding:8px 12px;
  border-radius: 10px;
  background: transparent;
  color: var(--muted);
  border: 1px solid transparent;
  font-weight:600;
  cursor: pointer;
  transform-origin: center;
  transition: transform var(--fast) var(--ease), box-shadow var(--fast) var(--ease), background var(--fast) var(--ease);
}
.app-theme .tab:hover { transform: translateY(-3px); box-shadow: var(--elev-1); }
.app-theme .tab.active {
  background: linear-gradient(180deg, rgba(255,255,255,0.018), rgba(255,255,255,0.01));
  color: var(--text);
  border-color: rgba(255,255,255,0.035);
  box-shadow: var(--elev-2);
}

/* ---------- CARD SYSTEM (GLASS + LAYERS) ---------- */
.app-theme .card {
  position: relative;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: var(--radius);
  padding: 18px;
  border: 1px solid var(--glass);
  box-shadow: var(--elev-1);
  overflow: hidden;
  transform: translateZ(0);
  transition: transform var(--mid) var(--ease), box-shadow var(--mid) var(--ease);
}
.app-theme .card:hover { transform: translateY(-6px); box-shadow: var(--elev-3); }

/* Glass blur accent (backdrop-filter if supported) */
.app-theme .card::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(90deg, rgba(255,255,255,0.015), transparent 30%);
  mix-blend-mode: overlay;
  opacity: 0.9;
}
@supports ((-webkit-backdrop-filter: blur(6px)) or (backdrop-filter: blur(6px))) {
  .app-theme .card { background: rgba(255,255,255,0.02); -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); }
}

/* Subtle floating highlight along top edge */
.app-theme .card .edge-highlight {
  position:absolute; left:0; right:0; top:0; height:3px;
  background: linear-gradient(90deg, transparent, rgba(77,124,254,0.16), rgba(42,183,169,0.12), transparent);
  opacity: 0.9;
  transform-origin: left center;
  animation: edgeSweep 3.2s linear infinite;
}
@keyframes edgeSweep {
  0% { transform: translateX(-105%); opacity: 0; }
  10% { opacity: 1; }
  50% { transform: translateX(0%); opacity: 1; }
  90% { opacity: 0; }
  100% { transform: translateX(105%); opacity: 0; }
}

/* ---------- BUTTONS (PRIMARY / SECONDARY / GHOST) ---------- */
.app-theme .btn {
  display:inline-flex; align-items:center; gap:10px; justify-content:center;
  padding:10px 16px; border-radius: 12px; font-weight:700; cursor:pointer; border: none;
  background: linear-gradient(90deg, var(--accent-2), var(--accent));
  color: #ffffff; box-shadow: var(--elev-2);
  transition: transform var(--fast) var(--ease), box-shadow var(--fast) var(--ease), opacity var(--fast) var(--ease);
}
.app-theme .btn:hover { transform: translateY(-4px); box-shadow: var(--elev-3); }
.app-theme .btn:active { transform: translateY(-1px); }
.app-theme .btn.secondary {
  background: transparent; color: var(--text); border: 1px solid rgba(255,255,255,0.04); box-shadow: none;
}
.app-theme .btn.ghost { background: transparent; border: none; color: var(--muted); }

/* Button micro-ripple (CSS only using pseudo-element) */
.app-theme .btn::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: linear-gradient(180deg, rgba(255,255,255,0.03), transparent 30%);
  opacity: 0;
  transition: opacity var(--fast) var(--ease);
  pointer-events: none;
}
.app-theme .btn:hover::after { opacity: 1; }

/* ---------- FORMS / INPUTS ---------- */
.app-theme input[type="text"], .app-theme input[type="number"], .app-theme textarea, .app-theme select {
  width:100%;
  padding: 12px 14px;
  border-radius: 10px;
  font-size: 0.95rem;
  color: var(--text);
  background: rgba(255,255,255,0.01);
  border: 1px solid rgba(255,255,255,0.02);
  transition: box-shadow var(--fast) var(--ease), border-color var(--fast) var(--ease), transform var(--fast) var(--ease);
}
.app-theme input:focus, .app-theme textarea:focus, .app-theme select:focus {
  outline: none;
  border-color: rgba(77,124,254,0.18);
  box-shadow: 0 10px 30px rgba(2,6,12,0.3), 0 0 0 6px rgba(77,124,254,0.06);
  transform: translateY(-1px);
}

/* Field label + helper text */
.app-theme label { display:block; font-weight:600; margin-bottom:6px; }
.app-theme .helper { font-size:0.88rem; color: var(--muted); }

/* ---------- SKELETON / SHIMMER ---------- */
.app-theme .skeleton {
  background: linear-gradient(90deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.035) 50%, rgba(255,255,255,0.02) 100%);
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}
.app-theme .skeleton::after {
  content: "";
  position: absolute; inset: 0;
  transform: translateX(-100%);
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.03) 50%, transparent 100%);
  animation: shimmer 1.6s linear infinite;
}
@keyframes shimmer {
  to { transform: translateX(100%); }
}

/* ---------- TOOLTIPS (CSS only) ---------- */
.app-theme .tooltip {
  position: relative; display:inline-block;
}
.app-theme .tooltip .tooltip-text {
  position: absolute; bottom: calc(100% + 8px); left: 50%; transform: translateX(-50%) translateY(6px);
  background: rgba(6,10,20,0.92); color: var(--text); padding:8px 10px; border-radius:8px; font-size:12px;
  white-space:nowrap; opacity:0; pointer-events:none; transition: opacity var(--fast) var(--ease), transform var(--fast) var(--ease);
  box-shadow: var(--elev-1);
  z-index: 50;
}
.app-theme .tooltip:hover .tooltip-text, .app-theme .tooltip:focus-within .tooltip-text {
  opacity:1; transform: translateX(-50%) translateY(0);
}

/* ---------- PROGRESS / LOADER ---------- */
.app-theme .progress {
  height: 10px; border-radius: 999px; background: rgba(255,255,255,0.02); overflow: hidden;
  box-shadow: inset 0 -2px 6px rgba(0,0,0,0.4);
}
.app-theme .progress > .bar {
  height:100%; width:0%; background: linear-gradient(90deg, var(--accent-2), var(--accent));
  transition: width var(--mid) var(--ease);
}

/* indeterminate loader */
.app-theme .indeterminate { position: relative; overflow:hidden; background: rgba(255,255,255,0.02); }
.app-theme .indeterminate::after {
  content: ""; position: absolute; inset:0; transform: translateX(-40%);
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.02) 30%, rgba(255,255,255,0.06) 50%, transparent 70%);
  animation: indeterminate 1.2s linear infinite;
}
@keyframes indeterminate { to { transform: translateX(100%); } }

/* ---------- TABLES & CODE ---------- */
.app-theme table { width:100%; border-collapse: collapse; font-size: 0.95rem; }
.app-theme th, .app-theme td { padding:10px 12px; border-bottom: 1px dashed rgba(255,255,255,0.03); text-align: left; }
.app-theme pre, .app-theme code {
  background: rgba(255,255,255,0.02); padding:10px; border-radius:8px; font-family: var(--font-mono); font-size: 0.92rem; overflow:auto;
}

/* ---------- SMALL UTILITIES ---------- */
.app-theme .muted { color: var(--muted); font-size: 0.92rem; }
.app-theme .small { font-size: 0.9rem; color: var(--muted); }
.app-theme .success { color: var(--success); }
.app-theme .danger { color: var(--danger); }

/* Visual separator */
.app-theme .divider { height:1px; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); margin: 12px 0; border-radius: 1px; }

/* ---------- ACCESSIBILITY ---------- */
/* prefer focus-visible (keyboard users) */
.app-theme :focus { outline: none; }
.app-theme :focus-visible {
  box-shadow: 0 0 0 5px rgba(77,124,254,0.08), 0 12px 30px rgba(2,6,12,0.25);
  border-radius: 8px;
}

/* High contrast mode for users who opt in */
@media (prefers-contrast: more) {
  :root { --glass: rgba(255,255,255,0.06); --muted: #BBD2E1; }
  .app-theme .card { border-color: rgba(255,255,255,0.06); }
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 900px) {
  .app-theme .row { flex-direction: column; gap: var(--gap-sm); }
  .app-theme .topbar { padding: 12px; }
  .app-theme .brand .title { font-size: calc(var(--base-size) * 1.15); }
  .app-theme .container { padding: var(--gap-sm); }
}

/* ---------- STREAMLIT OVERRIDES (non-invasive) ---------- */
/* Keep specificity modest so future Streamlit updates break less often. */
.app-theme div.stButton > button, .app-theme button[kind="primary"] {
  all: unset; box-sizing: border-box; display:inline-flex; align-items:center; justify-content:center; cursor:pointer;
  padding:10px 16px; border-radius:12px; font-weight:700;
  background: linear-gradient(90deg, var(--accent-2), var(--accent)); color: #fff; box-shadow: var(--elev-2);
  transition: transform var(--fast) var(--ease), box-shadow var(--fast) var(--ease);
  position: relative;
}
.app-theme div.stButton > button:hover { transform: translateY(-3px); }

/* Streamlit sidebar (best-effort) */
.app-theme .css-1lcbmhc, .app-theme .css-1d391kg, .app-theme .css-1v3fvcr /* common streamlit container classes */ {
  background: transparent;
}

/* ---------- CLEAN-UP: layer ordering ---------- */
.app-theme * { z-index: auto; }
.app-theme .topbar, .app-theme .card { z-index: 2; }

/* ---------- SMALL ANIMATIONS ---------- */
@keyframes floaty {
  0% { transform: translateY(0); }
  50% { transform: translateY(-6px); }
  100% { transform: translateY(0); }
}
.app-theme .float { animation: floaty 6s ease-in-out infinite; }

/* ---------- USAGE COMMENTS ----------
1) Wrap your Streamlit app content:
   st.markdown('<div class="app-theme">', unsafe_allow_html=True)
   ... your app UI ...
   st.markdown('</div>', unsafe_allow_html=True)

2) Optional fonts: if allowed, uncomment the @import lines to use Inter + Poppins + JetBrains Mono.
   For production, prefer hosting fonts locally or via a CSP-friendly mechanism.

3) To disable animations for a specific element, add: style="animation: none; transition: none;"
4) For modal/dialog styling or more advanced JS-driven interactions, add a small JS snippet (I can provide)
   — but this CSS provides a robust foundation without any JS.

5) If you need a dark/light toggle, the preferred approach is to toggle a class on the wrapper:
   <div class="app-theme light"> ... </div>
   and then apply `.app-theme.light { --bg: ...; }` overrides.

6) Streamlit classnames change sometimes — if a particular override fails, inspect the page classes
   and increase specificity for the few elements that need it (I kept specificity conservative intentionally).

---------- END THEME KIT ---------- */
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
    # Open upload tab by default so users can start immediately
    st.session_state["active_tab"] = "Upload & Process"


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
            preview_key = f"view_mode_{upload_obj.get('db_id') or upload_obj.get('id')}"
            view_mode = st.radio("Preview mode", ["Text (chunks)", "Slide Viewer"], index=0, key=preview_key)

            if view_mode == "Text (chunks)":
                with st.expander("Preview first 10 chunks (expand to view)"):
                    for i, c in enumerate(chunks[:10]):
                        st.code(c[:1200] + ("..." if len(c) > 1200 else ""))
            else:
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                # unique keys for this viewer instance
                uid = upload_obj.get("db_id") or upload_obj.get("id")
                viewer_index_key = f"viewer_index_{uid}"
                slider_key = f"viewer_slider_{uid}"
                prev_key = f"viewer_prev_{uid}"
                next_key = f"viewer_next_{uid}"

                # initialize viewer index
                if viewer_index_key not in st.session_state:
                    st.session_state[viewer_index_key] = 0

                slides_list = upload_obj.get("slides", []) or []
                num_slides = len(slides_list)

                # Navigation (Prev / label / Next)
                nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
                with nav_c1:
                    if st.button("◀ Prev", key=prev_key):
                        st.session_state[viewer_index_key] = max(0, st.session_state[viewer_index_key] - 1)
                with nav_c2:
                    st.markdown(f"### Slide {st.session_state[viewer_index_key] + 1} / {max(1, num_slides)}")
                with nav_c3:
                    if st.button("Next ▶", key=next_key):
                        st.session_state[viewer_index_key] = min(max(0, num_slides - 1), st.session_state[viewer_index_key] + 1)

                # main area + side controls
                main_col, side_col = st.columns([3, 1])
                cur_idx = int(st.session_state.get(viewer_index_key, 0))
                cur_idx = max(0, min(cur_idx, max(0, num_slides - 1)))
                st.session_state[viewer_index_key] = cur_idx  # clamp

                cur_slide = slides_list[cur_idx] if num_slides > 0 else {"index": 0, "text": "", "images": []}
                imgs = cur_slide.get("images") or []

                # Main image display — prefer PIL.Image and st.image (safer than long data URIs)
                with main_col:
                    if imgs:
                        img_bytes = imgs[0]
                        try:
                            from PIL import Image as PILImage
                            pil = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                            # downscale large images for performance
                            max_w = 1400
                            w, h = pil.size
                            if w > max_w:
                                pil.thumbnail((max_w, int(h * max_w / w)))
                            st.image(pil, use_column_width=True, caption=f"Slide {cur_slide.get('index') + 1}")
                        except Exception as e:
                            # fallback to raw bytes (some formats may already be PNG/JPEG)
                            try:
                                st.image(img_bytes, use_column_width=True, caption=f"Slide {cur_slide.get('index') + 1}")
                            except Exception:
                                st.info("Could not render image for this slide — showing extracted text instead.")
                                st.write(cur_slide.get("text", ""))
                    else:
                        st.info("No image available for this slide — showing extracted text")
                        st.write(cur_slide.get("text", ""))

                # Side controls: slider + compact thumbnails (clickable 'Open' buttons)
                with side_col:
                    st.markdown("**Controls**")
                    if num_slides > 0:
                        # slider 1-indexed for UX, but stored 0-indexed
                        s_val = st.slider("Go to slide", 1, num_slides, value=cur_idx + 1, key=slider_key)
                        new_idx = max(0, s_val - 1)
                        if new_idx != cur_idx:
                            st.session_state[viewer_index_key] = new_idx

                        st.markdown("---")
                        st.markdown("**Thumbnails**")

                        # thumbnails: show up to first 36 slides (to avoid huge layouts)
                        thumbs = slides_list[:min(36, num_slides)]
                        thumbs_per_row = 3
                        for i in range(0, len(thumbs), thumbs_per_row):
                            group = thumbs[i:i + thumbs_per_row]
                            cols = st.columns(len(group))
                            for ci, sthumb in enumerate(group):
                                t_idx = sthumb.get("index", i + ci)
                                t_img = (sthumb.get("images") or [None])[0]
                                with cols[ci]:
                                    if t_img:
                                        try:
                                            from PIL import Image as PILImage
                                            pil_t = PILImage.open(io.BytesIO(t_img)).convert("RGB")
                                            pil_t.thumbnail((260, 160))
                                            st.image(pil_t, use_column_width=True)
                                        except Exception:
                                            try:
                                                st.image(t_img, use_column_width=True)
                                            except Exception:
                                                st.markdown(f"<div class='small-muted'>Slide {t_idx+1}</div>", unsafe_allow_html=True)
                                    else:
                                        snippet = (sthumb.get("text") or "")[:100]
                                        st.markdown(f"<div class='small-muted' style='font-size:12px;padding:6px;border-radius:8px;border:1px solid rgba(255,255,255,0.02);'>{snippet}</div>", unsafe_allow_html=True)

                                    # unique open button per thumbnail
                                    open_key = f"open_thumb_{uid}_{t_idx}"
                                    if st.button("Open", key=open_key):
                                        st.session_state[viewer_index_key] = int(t_idx)
                    else:
                        st.info("No slides detected to display.")

                st.markdown("</div>", unsafe_allow_html=True)
            # ---------------------------------------------------------


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
# ------------------------------
# Flashcards
# ------------------------------
elif active_tab == "Flashcards":

    st.header("Flashcards & Spaced Repetition")
    if not st.session_state["uploads"]:
        st.info("No uploads yet. Go to Upload & Process.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload to work with", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_flashcards")

        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if not upload:
            st.warning("Selected upload not found.")
        else:
            slides = upload.get("slides", [])
            max_idx = max([s["index"] for s in slides]) if slides else 0
            slide_idx = st.number_input("Slide/Page index to extract flashcards from", min_value=0, max_value=max_idx, value=0, key=f"flash_idx_{sel_id}")
            text = next((s.get("text", "") for s in slides if s["index"] == slide_idx), "")

            # generate flashcards
            if st.button("Generate Flashcards from this slide/page", key=f"gen_flash_{sel_id}_{slide_idx}"):
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
                                qtxt = card.get("q") or card.get("question") or ""
                                atxt = card.get("a") or card.get("answer") or ""
                                st.markdown(f"**{i+1}.** {qtxt}")
                                st.markdown(f"<div class='small-muted'>Answer: {atxt}</div>", unsafe_allow_html=True)
                        else:
                            st.info("No valid Q/A pairs were found in the generated output.")
                    except Exception as e:
                        st.error(f"Failed to save flashcards: {e}")
                        log("flashcard save error:", e)

            st.markdown("---")
            st.subheader("Review due flashcards")
            now = int(time.time())
            cur = _db_conn.cursor()
            db_uid = upload_db_id(upload)
            cur.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards WHERE upload_id = ? AND (next_review IS NULL OR next_review <= ?) ORDER BY next_review ASC",
                        (db_uid, now))
            due_cards = cur.fetchall()
            if not due_cards:
                st.info("No cards due for this upload. Generate some or wait for scheduled review.")
            else:
                for row in due_cards:
                    fid, qtext, atext, eas, inter, reps, nxt = row
                    st.markdown(f"**Q:** {qtext}")
                    if st.button(f"Show Answer", key=f"show_{fid}"):
                        st.markdown(f"**A:** {atext}")
                        rating = st.slider("How well did you recall? (0-5)", 0, 5, 3, key=f"rating_{fid}")
                        if st.button("Submit Rating", key=f"submit_rating_{fid}"):
                            eas_new, interval_new, reps_new, next_review = sm2_update_card(eas, inter, reps, rating)
                            cur.execute("UPDATE flashcards SET easiness = ?, interval = ?, repetitions = ?, next_review = ? WHERE id = ?",
                                        (eas_new, interval_new, reps_new, next_review, fid))
                            _db_conn.commit()
                            st.success("Card updated; next review scheduled.")

# ------------------------------
# Export
# ------------------------------
elif active_tab == "Export":

    st.header("Export — Anki / Audio / Raw")
    if not st.session_state["uploads"]:
        st.info("No uploads available.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload to export from", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_export")
        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if not upload:
            st.warning("Selected upload not found.")
        else:
            st.markdown("**Flashcards / Anki**")
            if st.button("Export flashcards to Anki (TSV)", key=f"export_anki_{sel_id}"):
                try:
                    fname, data = anki_export_csv_for_upload(upload_db_id(upload), _db_conn)
                    st.download_button("Download Anki TSV", data, file_name=fname, mime="text/tab-separated-values", key=f"dl_anki_{sel_id}")
                except Exception as e:
                    st.error(f"Export failed: {e}")

            st.markdown("---")
            if _HAS_GTTS:
                if st.button("Export all generated lessons as MP3 (single)", key=f"export_mp3_{sel_id}"):
                    lesson_text = ""
                    for s in upload.get("slides", [])[:50]:
                        lesson_text += f"Slide {s['index']}. {s.get('text','')}\n\n"
                    try:
                        fname, data = text_to_speech_download(lesson_text)
                        st.download_button("Download lessons MP3", data, file_name=fname, mime="audio/mpeg", key=f"dl_mp3_{sel_id}")
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
            else:
                st.info("gTTS not available — TTS export disabled.")

            st.markdown("---")
            if st.button("Download extracted slides (JSON)", key=f"export_json_{sel_id}"):
                try:
                    ark = json.dumps(upload.get("slides", []), ensure_ascii=False, indent=2).encode("utf-8")
                    st.download_button("Download JSON", ark, file_name=f"{upload['filename']}_extracted.json", mime="application/json", key=f"dl_json_{sel_id}")
                except Exception as e:
                    st.error(f"Could not prepare download: {e}")

# ------------------------------
# Progress
# ------------------------------
elif active_tab == "Progress":

    st.header("Progress & Analytics")
    st.markdown("Overview of uploads and study artifacts")

    cur = _db_conn.cursor()
    cur.execute("SELECT id, filename, uploaded_at, meta FROM uploads ORDER BY uploaded_at DESC")
    rows = cur.fetchall()
    if not rows:
        st.info("No uploads logged in DB yet.")
    else:
        for r in rows:
            uid, fname, uploaded_at, meta = r
            st.markdown(f"**{fname}** — uploaded at {time.ctime(uploaded_at)}")
            meta_obj = safe_json_loads(meta) or {}
            st.write(meta_obj)
            c2 = _db_conn.cursor()
            c2.execute("SELECT COUNT(*) FROM flashcards WHERE upload_id = ?", (uid,))
            fc_count = c2.fetchone()[0]
            c2.execute("SELECT COUNT(*) FROM quizzes WHERE upload_id = ?", (uid,))
            q_count = c2.fetchone()[0]
            st.write(f"Flashcards: {fc_count} • Quizzes: {q_count}")
            st.markdown("---")

    c = _db_conn.cursor()
    c.execute("SELECT COUNT(*) FROM flashcards")
    total_fc = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM quizzes")
    total_q = c.fetchone()[0]
    st.write(f"Total flashcards in DB: {total_fc}")
    st.write(f"Total quizzes in DB: {total_q}")

# ------------------------------
# Settings
# ------------------------------
elif active_tab == "Settings":

    st.header("Settings & Diagnostics")
    if "OPENROUTER_API_KEY" not in st.session_state:
        st.session_state["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_KEY)

    key = st.text_input("OpenRouter API key (leave blank to use env/default)", value=st.session_state.get("OPENROUTER_API_KEY",""), type="password", key="api_key_input")
    if st.button("Save API Key (session only)", key="save_api_key"):
        st.session_state["OPENROUTER_API_KEY"] = key.strip() or st.session_state.get("OPENROUTER_API_KEY", "")
        st.success("API key set for this session.")

    st.markdown("Diagnostics:")
    st.write({
        "faiss_available": _HAS_FAISS,
        "pymupdf_available": _HAS_PYMUPDF,
        "easyocr_available": _HAS_EASYOCR,
        "sentence_transformers_available": _HAS_SENTENCE_TRANSFORMERS,
        "gtts_available": _HAS_GTTS
    })

    if st.button("Test OpenRouter (small ping)", key="test_openrouter"):
        test = call_openrouter_chat("You are a test bot.", "Say 'pong' in a plain short reply.", max_tokens=20)
        st.code(test)

    if st.button("Clear all session uploads (session state only)", key="clear_uploads_btn"):
        st.session_state["uploads"] = []
        st.success("Cleared session uploads.")
