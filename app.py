# SlideTutor — Improved version with EasyOCR (no Tesseract binary required)
# Save this file and run with: streamlit run slidetutor_improved_easyocr.py

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
# IMPORTANT: remove hard-coded API keys in production. Leave default empty and use env or settings.
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
        # Always return shapes (1, k) for compatibility
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
# Parsers (unchanged logic but kept defensive)
# ------------------------------
# [extract_from_pptx_bytes, extract_from_pdf_bytes, ocr_image_bytes_list, chunk_text remain similar]

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
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append(image_bytes)
                except Exception:
                    pass
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
        # gpu=False ensures CPU-only; change to True if GPU available
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
                    # detail=0 returns text only; paragraph=True attempts to group lines
                    raw = reader.readtext(arr, detail=0, paragraph=True)
                except TypeError:
                    # older easyocr versions may not support paragraph kw; fallback
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
# OpenRouter helpers (more robust)
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
        # prefer common response shapes
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            # many endpoints use message->content
            msg = first.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if not content:
                # some endpoints return 'text'
                content = first.get("text")
            if not content:
                # last resort: try data['output'][0]['content'] (some wrappers)
                content = json.dumps(first)
            return content
        # fallback: maybe top-level 'result' or 'output'
        if "result" in data:
            return str(data["result"])
        return "[openrouter] empty response"
    except Exception as e:
        try:
            return f"[OpenRouter error] {e} — resp_text={text}"
        except Exception:
            return f"[OpenRouter error] {e}"


def extract_json_from_text(text: str) -> Optional[Any]:
    # Try to extract first JSON array/object from a string, tolerant to code fences and markdown
    if not text:
        return None
    # remove common markdown code fences
    cleaned = text.strip()
    for fence in ['```json', '```']:
        cleaned = cleaned.replace(fence, '')
    # find first { or [ and attempt json.loads of progressive substring
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
    # last resort, attempt to find balanced JSON via regex for array
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
# Generators (improved parsing and deeper lesson option)
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
    # Longer detailed lesson, step-by-step from basics to advanced with progressive worked examples, derivations, and common misconceptions
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
    # fallback: simple heuristic generation (very small)
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
# SM-2 (unchanged)
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
    rows = c.fetchall()
    lines = []
    for q, a in rows:
        q2 = q.replace("\t", " ").replace("\n", " ")
        a2 = a.replace("\t", " ").replace("\n", " ")
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
# Streamlit UI (major UX improvements)
# ------------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
if "OPENROUTER_API_KEY" not in st.session_state:
    st.session_state["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_KEY)

st.markdown("""
    <style>
    html, body, .stApp { background-color: #0b0f13; color: #e6edf3; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:16px; border-radius:12px; }
    .small-muted { color:#9aa6b2; font-size:13px; }
    .slide-box { background:#071019; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); }
    .title { font-size:28px; font-weight:700; color:#fff; }
    .subtitle { color:#9aa6b2; margin-bottom:12px; }
    .mono { font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([8, 2])
with col1:
    st.markdown(f"<div class='title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align:right'><small class='small-muted'>Student edition • Improved</small></div>", unsafe_allow_html=True)

st.sidebar.title("SlideTutor")
nav = st.sidebar.radio("Navigation", ["Upload & Process", "Lessons", "Chat Q&A", "Quizzes", "Flashcards", "Export", "Progress", "Settings"])

if "uploads" not in st.session_state:
    st.session_state["uploads"] = []

def build_index_for_upload(upload: Dict):
    # builds embeddings and index and retains mapping
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

# Upload & Process
if nav == "Upload & Process":
    st.header("Upload PPTX / PDF (Student Upload)")
    uploaded_file = st.file_uploader("Choose a PPTX or PDF file", type=["pptx", "pdf"], accept_multiple_files=False)
    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        fname = uploaded_file.name
        st.info(f"Processing {fname} ...")
        try:
            if fname.lower().endswith(".pptx"):
                slides = extract_from_pptx_bytes(raw_bytes)
            elif fname.lower().endswith(".pdf"):
                slides = extract_from_pdf_bytes(raw_bytes)
            else:
                slides = [{"index": 0, "text": "[Unsupported file type]", "images": []}]
            for s in slides:
                imgs = s.get("images") or []
                if imgs:
                    ocr_texts = ocr_image_bytes_list(imgs)
                    appended = "\n".join([t for t in ocr_texts if t])
                    if appended:
                        s["text"] = (s.get("text", "") + "\n\n" + appended).strip()
            chunks = []
            mapping = []
            for s in slides:
                parts = chunk_text(s.get("text", ""))
                for p in parts:
                    chunks.append(p)
                    mapping.append({"slide": s["index"], "text": p})
            upload_obj = {
                "id": int(time.time() * 1000),
                "filename": fname,
                "uploaded_at": int(time.time()),
                "slides": slides,
                "chunks": chunks,
                "mapping": mapping
            }
            with st.spinner("Creating embeddings and index..."):
                upload_obj = build_index_for_upload(upload_obj)
            st.session_state["uploads"].append(upload_obj)
            cur = _db_conn.cursor()
            cur.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)", (fname, upload_obj["uploaded_at"], json.dumps({"n_slides": len(slides)})))
            _db_conn.commit()
            st.success(f"Upload processed: {len(slides)} slides/pages, {len(chunks)} chunks.")
            st.write("Preview first 3 chunks:")
            for i, c in enumerate(chunks[:3]):
                st.code(c[:800] + ("..." if len(c) > 800 else ""))
        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.exception(traceback.format_exc())

# Lessons
elif nav == "Lessons":
    st.header("Generate Multi-level Lessons")
    if not st.session_state["uploads"]:
        st.info("No uploads yet. Go to Upload & Process.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k])
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

            # options
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
                    # attempt to extract MCQs and flashcards from lesson
                    st.info("Attempting to auto-generate MCQs and flashcards from lesson and saving to DB.")
                    mcqs = generate_mcq_set_from_text(lesson, qcount=8)
                    fcards = generate_flashcards_from_text(lesson, n=12)
                    # save
                    cur = _db_conn.cursor()
                    try:
                        for q in mcqs:
                            cur.execute("INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                                        (upload["id"], q.get("question", ""), json.dumps(q.get("options", [])), int(q.get("answer_index", 0)), int(time.time())))
                        inserted = 0
                        for card in fcards:
                            qtext = card.get("q") or card.get("question") or ""
                            atext = card.get("a") or card.get("answer") or ""
                            if qtext and atext:
                                cur.execute("INSERT INTO flashcards (upload_id, question, answer, easiness, interval, repetitions, next_review) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                            (upload["id"], qtext, atext, 2.5, 1, 0, int(time.time())))
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

# Chat Q&A
elif nav == "Chat Q&A":
    st.header("Ask questions about your upload (Retrieval + LLM)")
    if not st.session_state["uploads"]:
        st.info("No uploads yet. Upload files first.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k])
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

# Quizzes (unchanged but improved save)
elif nav == "Quizzes":
    st.header("Auto-generated Quizzes")
    if not st.session_state["uploads"]:
        st.info("No uploads yet.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k])
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
                    for q in qset:
                        cur.execute("INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                                    (upload["id"], q.get("question", ""), json.dumps(q.get("options", [])), int(q.get("answer_index", 0)), int(time.time())))
                    _db_conn.commit()
                except Exception as e:
                    st.warning(f"Could not save quiz to DB: {e}")

# Flashcards (unchanged UI, but uses better generator)
elif nav == "Flashcards":
    st.header("Flashcards & Spaced Repetition")
    if not st.session_state["uploads"]:
        st.info("No uploads yet.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload to work with", options=list(options.keys()), format_func=lambda k: options[k])
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
                    for card in cards:
                        qtext = card.get("q") or card.get("question") or ""
                        atext = card.get("a") or card.get("answer") or ""
                        if qtext and atext:
                            cur.execute("INSERT INTO flashcards (upload_id, question, answer, easiness, interval, repetitions, next_review) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                        (upload["id"], qtext, atext, 2.5, 1, 0, int(time.time())))
                            inserted += 1
                    _db_conn.commit()
                    st.success(f"Inserted {inserted} flashcards into your deck.")
            st.markdown("---")
            st.subheader("Review due flashcards")
            now = int(time.time())
            cur = _db_conn.cursor()
            cur.execute("SELECT id, question, answer, easiness, interval, repetitions FROM flashcards WHERE upload_id = ? AND (next_review IS NULL OR next_review <= ?) ORDER BY next_review ASC",
                        (upload["id"], now))
            due_cards = cur.fetchall()
            if not due_cards:
                st.info("No cards due for this upload. Generate some or wait for scheduled review.")
            else:
                for row in due_cards:
                    fid, qtext, atext, eas, inter, reps = row
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

# Export
elif nav == "Export":
    st.header("Export — Anki / Audio / Raw")
    if not st.session_state["uploads"]:
        st.info("No uploads available.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload to export from", options=list(options.keys()), format_func=lambda k: options[k])
        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if upload:
            if st.button("Export flashcards to Anki (TSV)"):
                try:
                    fname, data = anki_export_csv_for_upload(upload["id"], _db_conn)
                    st.download_button("Download Anki TSV", data, file_name=fname, mime="text/tab-separated-values")
                except Exception as e:
                    st.error(f"Export failed: {e}")
            if _HAS_GTTS:
                if st.button("Export all generated lessons as MP3 (single)"):
                    lesson_text = ""
                    for s in upload.get("slides", [])[:10]:
                        lesson_text += f"Slide {s['index']}. {s.get('text','')}\n\n"
                    try:
                        fname, data = text_to_speech_download(lesson_text)
                        st.download_button("Download lessons MP3", data, file_name=fname, mime="audio/mpeg")
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
            if st.button("Download extracted slides (JSON)"):
                ark = json.dumps(upload.get("slides", []), ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button("Download JSON", ark, file_name=f"{upload['filename']}_extracted.json", mime="application/json")

# Progress
elif nav == "Progress":
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

# Settings
elif nav == "Settings":
    st.header("Settings & Diagnostics")
    if "OPENROUTER_API_KEY" not in st.session_state:
        st.session_state["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_KEY)
    key = st.text_input("OpenRouter API key (leave blank to use env/default)", value=st.session_state.get("OPENROUTER_API_KEY",""), type="password")
    if st.button("Save API Key (session only)"):
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
    if st.button("Test OpenRouter (small ping)"):
        test = call_openrouter_chat("You are a test bot.", "Say 'pong' in a plain short reply.", max_tokens=20)
        st.code(test)
    if st.button("Clear all uploads (session state only)"):
        st.session_state["uploads"] = []
        st.success("Cleared session uploads.")

st.markdown("<div style='position:fixed;bottom:8px;right:16px;color:#7f8c8d'>SlideTutor • Improved • Student-first</div>", unsafe_allow_html=True)

# End of improved app (EasyOCR version)
