# app.py -- SlideTutor AI (single-file)
# Run: streamlit run app.py

from __future__ import annotations
import os
import io
import re
import sys
import time
import json
import sqlite3
import logging
from typing import List, Dict, Tuple, Optional, Any
from types import ModuleType
from typing_extensions import TypedDict

# SlideTutor — single-file robust Streamlit app (no "import" statements at top)
# Paste into app.py and run: streamlit run app.py

# ----------------------------
# Load stdlib modules dynamically (no top-level import statements)
# ----------------------------
_os = __import__("os")
_io = __import__("io")
_re = __import__("re")
_sys = __import__("sys")
_time = __import__("time")
_json = __import__("json")
_logging = __import__("logging")
_sqlite3 = __import__("sqlite3")
_hashlib = __import__("hashlib")
_typing = __import__("typing")

# expose friendly names used later
os = _os
io = _io
re = _re
sys = _sys
time = _time
json = _json
logging = _logging
sqlite3 = _sqlite3
hashlib = _hashlib
# typing helpers (only used for comments / optional checks)
typing = _typing

# ----------------------------
# Helper: graceful optional imports via __import__ (returns module or None)
# ----------------------------
def _try_import(module_name, package_name=None):
    """Return (module_or_None, bool_installed)."""
    if package_name is None:
        package_name = module_name
    try:
        mod = __import__(module_name)
        return mod, True
    except Exception:
        logging.getLogger(__name__).warning("Optional module '%s' not found. To enable install: pip install %s", module_name, package_name)
        return None, False

# optional third-party modules
st, _HAS_STREAMLIT = _try_import("streamlit")
np, _HAS_NUMPY = _try_import("numpy")
requests, _HAS_REQUESTS = _try_import("requests")
pptx_module, _HAS_PPTX = _try_import("pptx", "python-pptx")
fitz, _HAS_PYMUPDF = _try_import("fitz", "PyMuPDF")
easyocr_module, _HAS_EASYOCR = _try_import("easyocr")
sentence_transformers_module, _HAS_SENTENCE_TRANSFORMERS = _try_import("sentence_transformers")
faiss, _HAS_FAISS = _try_import("faiss", "faiss-cpu")
PIL_module, _HAS_PIL = _try_import("PIL", "Pillow")

# map some names
easyocr = easyocr_module if _HAS_EASYOCR else None
SentenceTransformer = None
if _HAS_SENTENCE_TRANSFORMERS:
    try:
        SentenceTransformer = sentence_transformers_module.SentenceTransformer
    except Exception:
        SentenceTransformer = None

# require Streamlit + NumPy
if not _HAS_STREAMLIT:
    raise RuntimeError("Streamlit is required. Install: pip install streamlit")
if not _HAS_NUMPY:
    raise RuntimeError("NumPy is required. Install: pip install numpy")

# alias numpy as np (already assigned)
np = np

# ----------------------------
# Logging + configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("slidetutor")

# Config from environment / Streamlit secrets
DEFAULT_OPENROUTER_KEY = (
    (st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") and st.secrets is not None else None)
    or os.getenv("OPENROUTER_API_KEY", "")
)
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
DB_PATH = os.getenv("SLIDETUTOR_DB", "slidetutor.sqlite3")
APP_TITLE = "SlideTutor AI"
APP_SUBTITLE = "Your personal AI tutor for lecture slides and documents."

# ----------------------------
# Simple internal types via comments (avoid typing imports)
# SlideData: dict with keys: index(int), text(str), images(List[bytes]), ocr_text(str)
# ----------------------------

class APIError(Exception):
    pass

# ----------------------------
# Database (sqlite) helpers
# ----------------------------
@st.cache_resource
def get_db_connection(path: str = DB_PATH):
    """Open sqlite connection and ensure folder exists. Returns sqlite3.Connection."""
    try:
        db_dir = os.path.dirname(os.path.abspath(path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    except Exception as e:
        logger.debug("Could not ensure DB dir exists: %s", e)

    logger.info("Opening DB: %s", path)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON;")
    # return rows as tuples (we parse manually)
    return conn

def init_db(conn):
    """Create required tables if absent."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at INTEGER NOT NULL,
                meta TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                easiness REAL DEFAULT 2.5,
                interval INTEGER DEFAULT 1,
                repetitions INTEGER DEFAULT 0,
                next_review INTEGER,
                FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER,
                question TEXT NOT NULL,
                options TEXT NOT NULL,
                correct_index INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
            );
        """)
    logger.info("DB initialized.")

# initialize DB once
try:
    db_conn = get_db_connection()
    init_db(db_conn)
except Exception as e:
    logger.exception("Failed initializing DB: %s", e)
    raise

def save_upload_record(conn, filename, meta=None):
    ts = int(time.time())
    meta_json = json.dumps(meta or {})
    cur = conn.cursor()
    cur.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)", (filename, ts, meta_json))
    conn.commit()
    upload_id = cur.lastrowid
    logger.info("Saved upload record id=%s filename=%s", upload_id, filename)
    return upload_id

def fetch_uploads(conn, limit=50):
    cur = conn.cursor()
    cur.execute("SELECT id, filename, uploaded_at, meta FROM uploads ORDER BY uploaded_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    result = []
    for r in rows:
        try:
            meta = json.loads(r[3]) if r[3] else {}
        except Exception:
            meta = {}
        result.append({"id": r[0], "filename": r[1], "uploaded_at": r[2], "meta": meta})
    return result

def delete_upload(conn, upload_id):
    """Delete an upload and cascade flashcards/quizzes via foreign keys."""
    try:
        with conn:
            conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
        return True
    except Exception as e:
        logger.exception("Failed to delete upload %s: %s", upload_id, e)
        return False

# ----------------------------
# Utilities: stable widget keys & safe rerun
# ----------------------------
def _sanitize_key(s):
    if s is None:
        return "none"
    try:
        s = str(s)
        out = re.sub(r'[^0-9a-zA-Z]', '_', s)
        return out[:64]
    except Exception:
        return "key"

def widget_key(prefix, upload=None, index=None):
    uid = None
    if isinstance(upload, dict):
        uid = upload.get("db_id") or upload.get("filename")
    if uid is None:
        uid = index if index is not None else str(int(time.time()*1000))
    return f"{prefix}_{_sanitize_key(uid)}_{index if index is not None else ''}"

def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun") and callable(getattr(st, "experimental_rerun")):
            st.experimental_rerun()
        else:
            st.session_state["_rerun_token"] = st.session_state.get("_rerun_token", 0) + 1
    except Exception:
        st.session_state["_rerun_token"] = st.session_state.get("_rerun_token", 0) + 1

# ----------------------------
# Vector index (FAISS optional, numpy fallback)
# ----------------------------
_NORM_EPS = 1e-9

def _ensure_2d(arr):
    arr = np.asarray(arr)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr

def _safe_normalize(arr, axis=1):
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm = np.where(norm < _NORM_EPS, 1.0, norm)
    return arr / norm

class VectorIndex:
    """Wrapper around FAISS (if present) or NumPy matrix operations supporting cosine similarity."""
    def __init__(self, embeddings, texts):
        self.texts = list(texts or [])
        if embeddings is None:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            arr = np.asarray(embeddings, dtype=np.float32)
            if arr.ndim == 1:
                arr = _ensure_2d(arr)
            self.embeddings = arr
        self.dimension = self.embeddings.shape[1] if self.embeddings.size else 0
        self._normed_embeddings = _safe_normalize(self.embeddings) if self.embeddings.size else np.zeros_like(self.embeddings)
        self._use_faiss = False
        self.faiss_index = None
        if _HAS_FAISS and self.embeddings.size:
            try:
                idx = faiss.IndexFlatIP(self.dimension)
                idx.add(np.ascontiguousarray(self._normed_embeddings))
                self.faiss_index = idx
                self._use_faiss = True
                logger.info("FAISS index built with %d vectors (dim=%d)", self.embeddings.shape[0], self.dimension)
            except Exception as e:
                logger.warning("FAISS index build failed, falling back to NumPy: %s", e)
                self._use_faiss = False

    def search(self, query_embeddings, k):
        queries = _ensure_2d(np.asarray(query_embeddings, dtype=np.float32))
        if self._use_faiss and self.faiss_index is not None and self.dimension:
            try:
                qn = np.ascontiguousarray(_safe_normalize(queries))
                sims, idxs = self.faiss_index.search(qn, k)
                dists = 1.0 - sims.astype(np.float32)
                return dists, idxs.astype(np.int64)
            except Exception as e:
                logger.warning("FAISS search failed: %s -- falling back to numpy", e)
        return self._numpy_search(queries, k)

    def _numpy_search(self, queries, k):
        if self.embeddings.size == 0:
            return np.full((queries.shape[0], k), np.inf, dtype=np.float32), np.full((queries.shape[0], k), -1, dtype=np.int64)
        nq = _safe_normalize(queries)
        sims = nq @ self._normed_embeddings.T  # (nq, n)
        k_eff = min(k, sims.shape[1])
        idxs = np.argsort(-sims, axis=1)[:, :k_eff]
        final_sims = np.take_along_axis(sims, idxs, axis=1)
        if k_eff < k:
            pad_idxs = np.full((sims.shape[0], k - k_eff), -1, dtype=np.int64)
            pad_dists = np.full((sims.shape[0], k - k_eff), np.inf, dtype=np.float32)
            idxs = np.concatenate([idxs, pad_idxs], axis=1)
            final_sims = np.concatenate([final_sims, pad_dists], axis=1)
        return 1.0 - final_sims.astype(np.float32), idxs.astype(np.int64)

# ----------------------------
# Embeddings & chunking
# ----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Return a sentence-transformers model if available. Otherwise None (we use deterministic fallback)."""
    if not _HAS_SENTENCE_TRANSFORMERS or SentenceTransformer is None:
        logger.info("sentence-transformers not available; using deterministic fallback embeddings.")
        return None
    try:
        model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)
        return model
    except Exception as e:
        logger.exception("Failed to load embedding model: %s", e)
        return None

def _deterministic_fallback_embedding(texts, dim=384):
    """Create deterministic pseudo-embeddings using MD5 hashing; keeps same shape and dtype."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.md5(t.encode("utf-8")).digest()
        # expand hash to fill vector deterministically
        vals = []
        while len(vals) < dim:
            h = hashlib.md5(h).digest()
            vals.extend([b / 255.0 for b in h])
        arr = np.asarray(vals[:dim], dtype=np.float32)
        # normalize to unit length
        arr = arr / (np.linalg.norm(arr) + 1e-6)
        out[i] = arr
    return out

def embed_texts(texts, model):
    """Return 2D numpy array of embeddings. If a real model is not present, uses deterministic fallback."""
    if not texts:
        # return empty (0, dim) where dim guessed from model or default 384
        try:
            dim = model.get_sentence_embedding_dimension() if (model is not None and hasattr(model, "get_sentence_embedding_dimension")) else 384
        except Exception:
            dim = 384
        return np.zeros((0, dim), dtype=np.float32)
    if model is None:
        return _deterministic_fallback_embedding(texts, dim=384)
    try:
        enc = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return _ensure_2d(enc.astype(np.float32))
    except Exception as e:
        logger.exception("Embedding failed, falling back to deterministic: %s", e)
        return _deterministic_fallback_embedding(texts, dim=384)

def chunk_text(text, max_chars=1000):
    """Chunk text preserving paragraph boundaries where possible."""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current += (p + "\n")
        else:
            if current:
                chunks.append(current.strip())
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars].strip())
                current = ""
            else:
                current = p + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def extract_json_from_text(text):
    """Try to extract first JSON-like substring and parse. Cleans trailing commas."""
    if not text:
        return None
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if not match:
        return None
    s = match.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        cleaned = re.sub(r',\s*([\]}])', r'\1', s)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

# ----------------------------
# OCR / file parsing utilities
# ----------------------------
@st.cache_resource(show_spinner="Loading OCR model...")
def get_easyocr_reader(lang_list=None):
    if lang_list is None:
        lang_list = ["en"]
    if not _HAS_EASYOCR or easyocr is None:
        logger.info("EasyOCR not available; skipping OCR functionality.")
        return None
    try:
        reader = easyocr.Reader(lang_list, gpu=False)
        logger.info("EasyOCR initialized for %s", lang_list)
        return reader
    except Exception as e:
        logger.warning("EasyOCR initialization failed: %s", e)
        return None

def _bytes_to_pil_image(b):
    if not b:
        return None
    if _HAS_PIL and PIL_module is not None:
        try:
            Image = PIL_module.Image
            return Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            return None
    return None

def ocr_image_bytes(image_bytes, reader):
    """Perform OCR on raw image bytes using EasyOCR if available. Returns text (possibly empty)."""
    if not image_bytes or reader is None:
        return ""
    try:
        pil_img = _bytes_to_pil_image(image_bytes)
        if pil_img is not None:
            np_img = np.array(pil_img)
            out = reader.readtext(np_img, detail=0, paragraph=True)
        else:
            # Some EasyOCR versions accept numpy arrays or file paths only; attempt to pass bytes as last resort
            try:
                out = reader.readtext(image_bytes, detail=0, paragraph=True)
            except Exception:
                out = []
        if not out:
            return ""
        if isinstance(out, list):
            return "\n".join([str(x).strip() for x in out if str(x).strip()])[:10000]
        return str(out).strip()
    except Exception as e:
        logger.debug("OCR failed: %s", e)
        return ""

# PDF / PPTX extract helpers
def _extract_from_pptx(file_bytes):
    if not _HAS_PPTX or pptx_module is None:
        raise ModuleNotFoundError("python-pptx not installed. Install: pip install python-pptx")
    slides_content = []
    Presentation = pptx_module.Presentation
    prs = Presentation(io.BytesIO(file_bytes))
    for i, slide in enumerate(prs.slides):
        texts = []
        images = []
        for shape in slide.shapes:
            try:
                if getattr(shape, "has_text_frame", False) and getattr(shape, "text", None):
                    txt = (shape.text or "").strip()
                    if txt:
                        texts.append(txt)
            except Exception:
                pass
            try:
                img = getattr(shape, "image", None)
                if img is not None and getattr(img, "blob", None) is not None:
                    images.append(img.blob)
            except Exception:
                pass
        slides_content.append({"index": i, "text": "\n".join(texts), "images": images})
    return slides_content

def _extract_from_pdf(file_bytes):
    if not _HAS_PYMUPDF or fitz is None:
        raise ModuleNotFoundError("PyMuPDF not installed. Install: pip install PyMuPDF")
    slides_content = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        text = (page.get_text("text") or "").strip()
        images = []
        try:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    imgdict = doc.extract_image(xref)
                    if imgdict and "image" in imgdict:
                        images.append(imgdict["image"])
                except Exception:
                    logger.debug("Failed to extract image xref=%s on page=%s", xref, i)
        except Exception:
            logger.debug("Failed enumerating images on page %s", i)
        if not text and not images:
            try:
                pix = page.get_pixmap(dpi=150)
                images.append(pix.tobytes("png"))
            except Exception:
                pass
        slides_content.append({"index": i, "text": text, "images": images})
    try:
        doc.close()
    except Exception:
        pass
    return slides_content

def parse_and_extract_content(filename, file_bytes):
    """Return list of SlideData dicts for a .pptx or .pdf file."""
    ext = os.path.splitext(filename)[1].lower()
    raw_slides = []
    if ext == ".pptx":
        raw_slides = _extract_from_pptx(file_bytes)
    elif ext == ".pdf":
        raw_slides = _extract_from_pdf(file_bytes)
    else:
        raise ValueError("Unsupported file type: " + ext)

    reader = get_easyocr_reader()
    processed = []
    for s in raw_slides:
        ocr_texts = []
        for img in s.get("images", []) or []:
            try:
                t = ocr_image_bytes(img, reader)
                if t:
                    ocr_texts.append(t)
            except Exception:
                continue
        processed.append({
            "index": int(s["index"]),
            "text": s.get("text", "") or "",
            "images": s.get("images", []),
            "ocr_text": "\n".join(ocr_texts).strip()
        })
    return processed

# ----------------------------
# LLM / OpenRouter wrapper
# ----------------------------
def call_openrouter(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=1500, temperature=0.1):
    api_key = (st.session_state.get("OPENROUTER_API_KEY") if hasattr(st, "session_state") else None) or DEFAULT_OPENROUTER_KEY
    if not api_key:
        raise APIError("OpenRouter API key not configured. Set OPENROUTER_API_KEY env or provide in UI.")
    if not _HAS_REQUESTS or requests is None:
        raise APIError("requests library not available.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        body_json = resp.json()
        choices = body_json.get("choices") or []
        if choices:
            first = choices[0]
            content = (first.get("message") or {}).get("content") or first.get("text") or body_json.get("text")
            if not content:
                raise APIError("Empty content from OpenRouter response")
            return str(content).strip()
        if "output" in body_json:
            return str(body_json["output"]).strip()
        raise APIError("No choices returned from OpenRouter")
    except requests.exceptions.RequestException as e:
        logger.exception("OpenRouter request failed: %s", e)
        raise APIError(f"OpenRouter request failed: {e}")
    except ValueError as e:
        logger.exception("Failed decoding OpenRouter response: %s", e)
        raise APIError("Invalid response from OpenRouter")

# Prompts
PROMPT_LESSON_MULTILEVEL = ("You are an expert teacher. Create a multi-level lesson (Beginner, Intermediate, Advanced) "
                           "based on the provided text. For each level, give a clear explanation, a worked example, "
                           "and key tips. Finally, add a short 3-question quiz.")
PROMPT_MCQ_JSON = ("You are an AI that generates multiple-choice questions from text. Reply ONLY with a valid JSON "
                   "array of objects, where each object has keys 'question', 'options' (4 strings), and 'answer_index' (0-based).")
PROMPT_FLASHCARDS_JSON = ("You are an AI that creates flashcards from text. Reply ONLY with a valid JSON array of objects, "
                          "where each object has keys 'question' and 'answer'.")

def generate_multilevel_lesson(context):
    return call_openrouter(PROMPT_LESSON_MULTILEVEL, f"Generate a lesson from this text:\n\n{context}")

def generate_mcq_set_from_text(context, qcount=5):
    resp = call_openrouter(PROMPT_MCQ_JSON, f"Create exactly {qcount} MCQs from this text:\n\n{context}")
    parsed = extract_json_from_text(resp)
    return parsed or []

def generate_flashcards_from_text(context, n=10):
    resp = call_openrouter(PROMPT_FLASHCARDS_JSON, f"Create {n} flashcards from this text:\n\n{context}")
    parsed = extract_json_from_text(resp)
    return parsed or []

# ----------------------------
# RAG: build index and answer
# ----------------------------
def build_upload_index(slides, model):
    chunks = []
    origin_map = []
    for s in slides:
        combined = "\n".join([s.get("text", ""), s.get("ocr_text", "")]).strip()
        if not combined:
            continue
        pieces = chunk_text(combined, max_chars=1200)
        for pi, p in enumerate(pieces):
            chunks.append(p)
            origin_map.append((s["index"], pi))
    if not chunks:
        emb = np.zeros((0, 384), dtype=np.float32)
        idx = VectorIndex(emb, [])
        return {"chunks": [], "index": idx, "slides": slides, "meta": {"chunks": 0}}
    emb = embed_texts(chunks, model)
    idx = VectorIndex(emb, chunks)
    logger.info("Built upload index with %d chunks", len(chunks))
    return {"chunks": chunks, "index": idx, "slides": slides, "meta": {"chunks": len(chunks), "origin_map": origin_map}}

def answer_question_with_rag(query, uploads):
    if not query or not query.strip():
        return "Please provide a question."
    try:
        model = get_embedding_model()
    except Exception as e:
        return "Embeddings not available: " + str(e)
    q_emb = embed_texts([query], model)
    found_chunks = []
    for upload in uploads:
        idx = upload.get("index")
        chunks = upload.get("chunks") or []
        if not idx or getattr(idx, "dimension", 0) == 0:
            continue
        try:
            dists, ids = idx.search(q_emb, k=min(TOP_K, max(1, len(chunks))))
        except Exception as e:
            logger.warning("Index search failed for upload %s: %s", upload.get("filename"), e)
            continue
        for j in ids[0]:
            if j is None or int(j) < 0:
                continue
            try:
                found_chunks.append(chunks[int(j)])
            except Exception:
                continue
    if not found_chunks:
        return "Couldn't find relevant information in your documents."
    context = "\n---\n".join(found_chunks[: min(len(found_chunks), TOP_K * 3)])
    sys_prompt = "You are a precise assistant. Answer using ONLY the provided context. If not available, say you cannot answer."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    try:
        return call_openrouter(sys_prompt, user_prompt, max_tokens=600)
    except APIError as e:
        return f"LLM Error: {e}"

# ----------------------------
# SM-2 spaced repetition
# ----------------------------
def sm2_update_card(easiness, interval, reps, quality):
    if quality < 3:
        reps = 0
        interval = 1
    else:
        reps += 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = max(1, round(interval * easiness))
    easiness = max(1.3, easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    next_review = int(time.time()) + interval * 86400
    return easiness, interval, reps, next_review

# ----------------------------
# Exports
# ----------------------------
def anki_export_tsv(upload_id, conn):
    cursor = conn.execute("SELECT question, answer FROM flashcards WHERE upload_id = ?", (upload_id,))
    rows = cursor.fetchall()
    if not rows:
        return None
    buf = io.StringIO()
    for q, a in rows:
        qs = (q or "").replace("\t", " ").replace("\n", "<br>")
        as_ = (a or "").replace("\t", " ").replace("\n", "<br>")
        buf.write(f"{qs}\t{as_}\n")
    filename = f"anki_deck_upload_{upload_id}.txt"
    return filename, buf.getvalue().encode("utf-8")

# ----------------------------
# Minimal UI (Streamlit)
# ----------------------------
APP_CSS = """
/* small polished theme */
.stApp .block-container { padding: 24px 28px !important; }
"""

def main_ui():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

    # Settings: allow user to provide OpenRouter key at runtime
    with st.expander("Settings / API Key (optional)"):
        key = st.text_input("OpenRouter API key (optional, overrides env/secrets)", value="", type="password", key=widget_key("api_key"))
        if key:
            st.session_state["OPENROUTER_API_KEY"] = key

    col1, col2 = st.columns([1, 2])

    # ensure uploads_in_memory container exists
    if "uploads_in_memory" not in st.session_state:
        st.session_state["uploads_in_memory"] = []

    with col1:
        st.header("Upload file (.pptx or .pdf)")
        uploaded = st.file_uploader("Upload a .pptx or .pdf", type=["pptx", "pdf"], accept_multiple_files=False, key=widget_key("uploader"))
        if uploaded is not None:
            filename = uploaded.name
            try:
                file_bytes = uploaded.read()
            except Exception as e:
                st.error("Failed reading file bytes: " + str(e))
                file_bytes = None
            if file_bytes:
                st.write(f"Got file: **{filename}** — {len(file_bytes)} bytes")
                if st.button("Parse and index file", key=widget_key("parse", filename)):
                    try:
                        slides = parse_and_extract_content(filename, file_bytes)
                        st.success(f"Parsed {len(slides)} slides/pages.")
                        # Save DB record
                        try:
                            upload_id = save_upload_record(db_conn, filename, meta={"slides": len(slides)})
                        except Exception as e:
                            logger.exception("Failed to save upload record: %s", e)
                            st.error("Failed to save upload metadata.")
                            upload_id = None
                        # Build index (embedding model may be None -> fallback embeddings used)
                        try:
                            model = get_embedding_model()
                            upload_obj = build_upload_index(slides, model)
                            st.session_state["uploads_in_memory"].append({
                                "upload_id": upload_id,
                                "filename": filename,
                                "slides": slides,
                                "chunks": upload_obj["chunks"],
                                "index": upload_obj["index"],
                                "meta": upload_obj["meta"],
                            })
                            st.success("Index built and stored in session.")
                        except Exception as e:
                            logger.exception("Failed to build index: %s", e)
                            st.error("Failed to build embeddings/index: " + str(e))
                    except Exception as e:
                        st.exception("Failed to parse file: %s" % e)

        st.markdown("### Existing uploads (DB)")
        uploads_list = fetch_uploads(db_conn, limit=20)
        for u in uploads_list:
            col_a, col_b = st.columns([8,1])
            with col_a:
                st.write(f"- ID {u['id']} — {u['filename']} — {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(u['uploaded_at']))}")
            with col_b:
                if st.button("Delete", key=widget_key("del_upload", u)):
                    if delete_upload(db_conn, u["id"]):
                        st.success("Deleted upload id="+str(u["id"]))
                        safe_rerun()
                    else:
                        st.error("Delete failed for upload id="+str(u["id"]))

    with col2:
        st.header("Ask a question (RAG)")
        query = st.text_area("Question about your uploaded documents", height=140, key=widget_key("query"))
        if st.button("Answer question", key=widget_key("answer_btn")):
            uploads_in_memory = st.session_state.get("uploads_in_memory", [])
            if not uploads_in_memory:
                st.warning("No indexed uploads in this session. Upload & index a file first.")
            else:
                with st.spinner("Searching..."):
                    ans = answer_question_with_rag(query, uploads_in_memory)
                    st.markdown("### Answer")
                    st.write(ans)

        st.markdown("---")
        st.header("Create flashcards / quizzes")
        sample_text = st.text_area("Text to generate flashcards or MCQs from", height=160, key=widget_key("sample_text"))
        col3, col4 = st.columns([1, 1])
        with col3:
            if st.button("Generate 5 MCQs", key=widget_key("gen_mcq")):
                if not sample_text.strip():
                    st.warning("Provide some text above to generate MCQs.")
                else:
                    try:
                        mcqs = generate_mcq_set_from_text(sample_text, qcount=5)
                        st.write(mcqs)
                    except Exception as e:
                        st.error("Failed to generate MCQs: " + str(e))
        with col4:
            if st.button("Generate 10 flashcards", key=widget_key("gen_fc")):
                if not sample_text.strip():
                    st.warning("Provide some text above to generate flashcards.")
                else:
                    try:
                        fcs = generate_flashcards_from_text(sample_text, n=10)
                        st.write(fcs)
                    except Exception as e:
                        st.error("Failed to generate flashcards: " + str(e))

    st.sidebar.header("Session info")
    st.sidebar.write("Uploads indexed in session:")
    uims = st.session_state.get("uploads_in_memory", [])
    for u in uims:
        st.sidebar.write(f"- {u.get('filename')} (chunks={len(u.get('chunks', []))})")

if __name__ == "__main__":
    main_ui()

# SlideTutor — Robust single-file Streamlit app (NO top-level import statements)
# Save as app.py and run: streamlit run app.py

# ----------------------------
# Dynamic stdlib / optional imports (no "import" statements at top)
# ----------------------------
def _try_import(module_name, package_name=None):
    try:
        return __import__(module_name), True
    except Exception:
        return None, False

# stdlib aliases (loaded with __import__ to respect "no import statements" request)
_os = __import__("os")
_io = __import__("io")
_re = __import__("re")
_sys = __import__("sys")
_time = __import__("time")
_json = __import__("json")
_logging = __import__("logging")
_sqlite3 = __import__("sqlite3")
_hashlib = __import__("hashlib")

os = _os
io = _io
re = _re
sys = _sys
time = _time
json = _json
logging = _logging
sqlite3 = _sqlite3
hashlib = _hashlib

# optional third-party libraries (graceful)
st, _HAS_STREAMLIT = _try_import("streamlit")
np, _HAS_NUMPY = _try_import("numpy")
requests, _HAS_REQUESTS = _try_import("requests")
pptx_module, _HAS_PPTX = _try_import("pptx",)
fitz, _HAS_PYMUPDF = _try_import("fitz",)
easyocr_module, _HAS_EASYOCR = _try_import("easyocr",)
sentence_transformers_module, _HAS_SENTENCE_TRANSFORMERS = _try_import("sentence_transformers",)
faiss, _HAS_FAISS = _try_import("faiss",)
PIL_module, _HAS_PIL = _try_import("PIL",)

# sanity: require streamlit + numpy
if not _HAS_STREAMLIT:
    raise RuntimeError("Streamlit is required. Install: pip install streamlit")
if not _HAS_NUMPY:
    raise RuntimeError("NumPy is required. Install: pip install numpy")

# map optional names
easyocr = easyocr_module if _HAS_EASYOCR else None
SentenceTransformer = None
if _HAS_SENTENCE_TRANSFORMERS:
    try:
        SentenceTransformer = sentence_transformers_module.SentenceTransformer
    except Exception:
        SentenceTransformer = None

np = np  # alias

# ----------------------------
# Logging + app config
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("slidetutor")

DEFAULT_OPENROUTER_KEY = (
    (st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") and st.secrets is not None else None)
    or os.getenv("OPENROUTER_API_KEY", "")
)
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
DB_PATH = os.getenv("SLIDETUTOR_DB", "slidetutor.sqlite3")
APP_TITLE = "SlideTutor AI"
APP_SUBTITLE = "Beautiful. Simple. Smart — study from slides quickly."

# ----------------------------
# Database helpers
# ----------------------------
@st.cache_resource
def get_db_connection(path=DB_PATH):
    try:
        folder = os.path.dirname(os.path.abspath(path))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn):
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at INTEGER NOT NULL,
                meta TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                easiness REAL DEFAULT 2.5,
                interval INTEGER DEFAULT 1,
                repetitions INTEGER DEFAULT 0,
                next_review INTEGER,
                FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id INTEGER,
                question TEXT NOT NULL,
                options TEXT NOT NULL,
                correct_index INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
            );
        """)

# initialize DB
try:
    db_conn = get_db_connection()
    init_db(db_conn)
except Exception as e:
    logger.exception("DB init failed: %s", e)
    raise

# ----------------------------
# Utility helpers
# ----------------------------
def _sanitize_key(s):
    try:
        if s is None:
            return "none"
        s = str(s)
        return re.sub(r'[^0-9a-zA-Z]', '_', s)[:64]
    except Exception:
        return "key"

def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun") and callable(getattr(st, "experimental_rerun")):
            st.experimental_rerun()
        else:
            st.session_state["_rerun_token"] = st.session_state.get("_rerun_token", 0) + 1
    except Exception:
        st.session_state["_rerun_token"] = st.session_state.get("_rerun_token", 0) + 1

# ----------------------------
# VectorIndex: FAISS optional, numpy fallback
# ----------------------------
_NORM_EPS = 1e-9

def _ensure_2d(arr):
    arr = np.asarray(arr)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr

def _safe_normalize(arr, axis=1):
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm = np.where(norm < _NORM_EPS, 1.0, norm)
    return arr / norm

class VectorIndex:
    def __init__(self, embeddings, texts):
        self.texts = list(texts or [])
        if embeddings is None:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            arr = np.asarray(embeddings, dtype=np.float32)
            if arr.ndim == 1:
                arr = _ensure_2d(arr)
            self.embeddings = arr
        self.dimension = self.embeddings.shape[1] if self.embeddings.size else 0
        self._normed_embeddings = _safe_normalize(self.embeddings) if self.embeddings.size else np.zeros_like(self.embeddings)
        self._use_faiss = False
        self.faiss_index = None
        if _HAS_FAISS and self.embeddings.size:
            try:
                idx = faiss.IndexFlatIP(self.dimension)
                idx.add(np.ascontiguousarray(self._normed_embeddings))
                self.faiss_index = idx
                self._use_faiss = True
            except Exception as e:
                logger.warning("FAISS init failed, using numpy. %s", e)
                self._use_faiss = False

    def search(self, query_embeddings, k):
        queries = _ensure_2d(np.asarray(query_embeddings, dtype=np.float32))
        if self._use_faiss and self.faiss_index is not None and self.dimension:
            try:
                qn = np.ascontiguousarray(_safe_normalize(queries))
                sims, idxs = self.faiss_index.search(qn, k)
                dists = 1.0 - sims.astype(np.float32)
                return dists, idxs.astype(np.int64)
            except Exception as e:
                logger.warning("FAISS search failed: %s", e)
        return self._numpy_search(queries, k)

    def _numpy_search(self, queries, k):
        if self.embeddings.size == 0:
            return np.full((queries.shape[0], k), np.inf, dtype=np.float32), np.full((queries.shape[0], k), -1, dtype=np.int64)
        nq = _safe_normalize(queries)
        sims = nq @ self._normed_embeddings.T
        k_eff = min(k, sims.shape[1])
        idxs = np.argsort(-sims, axis=1)[:, :k_eff]
        final_sims = np.take_along_axis(sims, idxs, axis=1)
        if k_eff < k:
            pad_idxs = np.full((sims.shape[0], k - k_eff), -1, dtype=np.int64)
            pad_dists = np.full((sims.shape[0], k - k_eff), np.inf, dtype=np.float32)
            idxs = np.concatenate([idxs, pad_idxs], axis=1)
            final_sims = np.concatenate([final_sims, pad_dists], axis=1)
        return 1.0 - final_sims.astype(np.float32), idxs.astype(np.int64)

# ----------------------------
# Embeddings with deterministic fallback
# ----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    if not _HAS_SENTENCE_TRANSFORMERS or SentenceTransformer is None:
        return None
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.warning("Sentence-transformers load failed: %s", e)
        return None

def _deterministic_fallback_embedding(texts, dim=384):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.md5((t or "").encode("utf-8")).digest()
        vals = []
        while len(vals) < dim:
            h = hashlib.md5(h).digest()
            vals.extend([b / 255.0 for b in h])
        arr = np.asarray(vals[:dim], dtype=np.float32)
        arr = arr / (np.linalg.norm(arr) + 1e-6)
        out[i] = arr
    return out

def embed_texts(texts, model):
    if not texts:
        dim = 384
        try:
            dim = model.get_sentence_embedding_dimension() if model is not None and hasattr(model, "get_sentence_embedding_dimension") else dim
        except Exception:
            pass
        return np.zeros((0, dim), dtype=np.float32)
    if model is None:
        return _deterministic_fallback_embedding(texts, dim=384)
    try:
        enc = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return _ensure_2d(enc.astype(np.float32))
    except Exception as e:
        logger.exception("Embedding failed, fallback to deterministic: %s", e)
        return _deterministic_fallback_embedding(texts, dim=384)

# ----------------------------
# OCR & file parsing helpers
# ----------------------------
@st.cache_resource(show_spinner="Loading OCR model...")
def get_easyocr_reader(lang_list=None):
    if lang_list is None:
        lang_list = ["en"]
    if not _HAS_EASYOCR or easyocr is None:
        return None
    try:
        reader = easyocr.Reader(lang_list, gpu=False)
        return reader
    except Exception as e:
        logger.warning("EasyOCR init failed: %s", e)
        return None

def _bytes_to_pil_image(b):
    if not b:
        return None
    if _HAS_PIL and PIL_module is not None:
        try:
            Image = PIL_module.Image
            return Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            return None
    return None

def ocr_image_bytes(image_bytes, reader):
    if not image_bytes or reader is None:
        return ""
    try:
        pil_img = _bytes_to_pil_image(image_bytes)
        if pil_img is not None:
            arr = np.array(pil_img)
            out = reader.readtext(arr, detail=0, paragraph=True)
        else:
            try:
                out = reader.readtext(image_bytes, detail=0, paragraph=True)
            except Exception:
                out = []
        if not out:
            return ""
        if isinstance(out, list):
            return "\n".join([str(x).strip() for x in out if str(x).strip()])[:10000]
        return str(out).strip()
    except Exception as e:
        logger.debug("OCR failed: %s", e)
        return ""

def _extract_from_pptx(file_bytes):
    if not _HAS_PPTX or pptx_module is None:
        raise ModuleNotFoundError("python-pptx not installed.")
    slides = []
    Presentation = pptx_module.Presentation
    prs = Presentation(io.BytesIO(file_bytes))
    for i, slide in enumerate(prs.slides):
        texts = []
        images = []
        for shape in slide.shapes:
            try:
                if getattr(shape, "has_text_frame", False) and getattr(shape, "text", None):
                    txt = (shape.text or "").strip()
                    if txt:
                        texts.append(txt)
            except Exception:
                pass
            try:
                img = getattr(shape, "image", None)
                if img is not None and getattr(img, "blob", None) is not None:
                    images.append(img.blob)
            except Exception:
                pass
        slides.append({"index": i, "text": "\n".join(texts), "images": images})
    return slides

def _extract_from_pdf(file_bytes):
    if not _HAS_PYMUPDF or fitz is None:
        raise ModuleNotFoundError("PyMuPDF not installed.")
    slides = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        text = (page.get_text("text") or "").strip()
        images = []
        try:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    d = doc.extract_image(xref)
                    if d and "image" in d:
                        images.append(d["image"])
                except Exception:
                    logger.debug("Extract image failed xref=%s page=%s", xref, i)
        except Exception:
            logger.debug("get_images failed on page %s", i)
        if not text and not images:
            try:
                pix = page.get_pixmap(dpi=150)
                images.append(pix.tobytes("png"))
            except Exception:
                pass
        slides.append({"index": i, "text": text, "images": images})
    try:
        doc.close()
    except Exception:
        pass
    return slides

def parse_and_extract_content(filename, file_bytes):
    try:
        ext = os.path.splitext(filename)[1].lower()
    except Exception:
        logger.exception("Bad filename: %r", filename)
        return []
    try:
        if ext == ".pptx":
            raw = _extract_from_pptx(file_bytes)
        elif ext == ".pdf":
            raw = _extract_from_pdf(file_bytes)
        else:
            logger.warning("Unsupported extension: %s", ext)
            return []
    except Exception as e:
        try:
            st.error(f"Failed to parse file '{filename}': {e}")
        except Exception:
            pass
        logger.exception("Parse failed for %s: %s", filename, e)
        return []

    reader = get_easyocr_reader(["en"])
    processed = []
    total = max(1, len(raw))
    progress = None
    try:
        progress = st.progress(0, text="Extracting content...")
    except Exception:
        progress = None

    for i, s in enumerate(raw):
        try:
            imgs = s.get("images") or []
            ocr_texts = []
            for img in imgs:
                try:
                    if not img:
                        continue
                    t = ocr_image_bytes(img, reader)
                    if t:
                        ocr_texts.append(t)
                except Exception:
                    continue
            processed.append({
                "index": int(s.get("index", i)),
                "text": (s.get("text") or "").strip(),
                "images": imgs,
                "ocr_text": "\n".join(ocr_texts).strip()
            })
            if progress is not None:
                try:
                    progress.progress(min(1.0, float(i+1)/float(total)), text=f"Processing {i+1}/{total}")
                except Exception:
                    try:
                        progress.progress(min(1.0, float(i+1)/float(total)))
                    except Exception:
                        pass
        except Exception as e:
            logger.exception("Slide processing failed: %s", e)
            continue
    return processed

# ----------------------------
# Robust chunking & JSON extraction
# ----------------------------
def chunk_text(text, max_chars=1000):
    if not text or not isinstance(text, str):
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(p) <= max_chars:
            if not current:
                current = p + "\n"
            else:
                if len(current) + len(p) + 1 <= max_chars:
                    current += p + "\n"
                else:
                    chunks.append(current.strip())
                    current = p + "\n"
        else:
            try:
                sentences = re.split(r'(?<=[\.\?\!])\s+', p)
            except Exception:
                sentences = [p[i:i+max_chars] for i in range(0, len(p), max_chars)]
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(s) > max_chars:
                    for j in range(0, len(s), max_chars):
                        block = s[j:j+max_chars].strip()
                        if not block:
                            continue
                        if current and len(current) + len(block) + 1 <= max_chars:
                            current += block + " "
                        else:
                            if current:
                                chunks.append(current.strip())
                                current = ""
                            chunks.append(block)
                else:
                    if not current:
                        current = s + " "
                    elif len(current) + len(s) + 1 <= max_chars:
                        current += s + " "
                    else:
                        chunks.append(current.strip())
                        current = s + " "
            if current:
                chunks.append(current.strip())
                current = ""
    if current:
        chunks.append(current.strip())
    final = [c.strip() for c in chunks if c and c.strip()]
    return final

def extract_json_from_text(text):
    if not text or not isinstance(text, str):
        return None
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    candidate = None
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            candidate = None
    start_pos = None
    start_char = None
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            start_pos = i
            start_char = ch
            break
    if start_pos is None:
        return None
    opening = {"{": "}", "[": "]"}
    stack = [opening[start_char]]
    i = start_pos
    in_string = False
    escape = False
    candidate = None
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = False
        else:
            if ch == '"' or ch == "'":
                in_string = ch
            elif ch in ("{", "["):
                stack.append(opening[ch])
            elif ch in ("}", "]"):
                if not stack:
                    break
                expected = stack.pop()
                if ch != expected:
                    break
                if not stack:
                    candidate = text[start_pos:i+1]
                    break
        i += 1
    if not candidate:
        m2 = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if not m2:
            return None
        candidate = m2.group(0)
    def clean_trailing_commas(s):
        return re.sub(r',\s*(\}|])', r'\1', s)
    attempts = 0
    s = candidate
    while attempts < 3:
        try:
            return json.loads(s)
        except Exception:
            s = clean_trailing_commas(s)
            attempts += 1
    logger.debug("Failed parsing JSON candidate: %s", candidate[:200].replace("\n", " "))
    return None

# ----------------------------
# LLM / OpenRouter wrapper
# ----------------------------
def call_openrouter(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=1200, temperature=0.1, is_json=False):
    api_key = None
    try:
        api_key = st.session_state.get("OPENROUTER_API_KEY") or DEFAULT_OPENROUTER_KEY
    except Exception:
        api_key = DEFAULT_OPENROUTER_KEY
    if not api_key:
        raise Exception("OpenRouter API key not configured.")
    if not _HAS_REQUESTS or requests is None:
        raise Exception("requests not available.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}], "max_tokens": max_tokens, "temperature": temperature}
    if is_json:
        try:
            body["response_format"] = {"type": "json_object"}
        except Exception:
            pass
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
    except Exception as e:
        logger.exception("Network error: %s", e)
        raise Exception("Network error when contacting LLM: " + str(e))
    try:
        resp.raise_for_status()
    except Exception:
        raw = "<no-body>"
        try:
            raw = resp.text
        except Exception:
            pass
        logger.error("LLM HTTP %s: %s", resp.status_code, raw[:1000])
        raise Exception(f"LLM API returned HTTP {resp.status_code}: {raw[:1000]}")
    try:
        data = resp.json()
    except Exception:
        logger.exception("Failed to parse LLM response as JSON")
        raise Exception("Failed to decode LLM response as JSON.")
    content = None
    try:
        choices = data.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message") or {}
                content = msg.get("content") or first.get("text") or first.get("message", {}).get("content")
        if not content:
            if "output" in data:
                content = data["output"]
            elif "result" in data:
                content = data["result"]
    except Exception:
        content = None
    if not content:
        logger.error("LLM missing expected fields; full response (truncated): %s", json.dumps(data)[:2000])
        raise Exception("LLM response missing expected content.")
    if not isinstance(content, str):
        try:
            content = json.dumps(content)
        except Exception:
            content = str(content)
    return content.strip()

# ----------------------------
# Content generation wrappers
# ----------------------------
PROMPT_LESSON_MULTILEVEL = """You are a patient expert teacher. Produce three sections...
## Beginner
...
"""
PROMPT_MCQ_JSON = """You are an AI that generates MCQs. Reply only with a JSON array of objects with keys: question, options (4), answer_index (0-based)."""
PROMPT_FLASHCARDS_JSON = """You are an AI that generates flashcards. Reply only with a JSON array of objects with keys: question, answer."""

def generate_multilevel_lesson(text, extra=""):
    prompt = f"{PROMPT_LESSON_MULTILEVEL}\n\nTEXT:\n{text}\n\nEXTRA:\n{extra}"
    try:
        return call_openrouter("You are a helpful teacher.", prompt, max_tokens=1500, temperature=0.2)
    except Exception as e:
        logger.exception("Lesson generation failed: %s", e)
        return f"LLM error: {e}"

def generate_mcq_set_from_text(text, qcount=5):
    if not text:
        return []
    prompt = f"Create exactly {qcount} MCQs from the text below.\n\n{text}"
    try:
        resp = call_openrouter(PROMPT_MCQ_JSON, prompt, max_tokens=800, temperature=0.0, is_json=True)
        parsed = extract_json_from_text(resp)
        if isinstance(parsed, list):
            return parsed
        try:
            return json.loads(resp)
        except Exception:
            logger.debug("MCQ parse failed")
            return []
    except Exception as e:
        logger.warning("MCQ generation failed: %s", e)
        return []

def generate_flashcards_from_text(text, n=10):
    if not text:
        return []
    prompt = f"Extract up to {n} flashcards from the text below.\n\n{text}"
    try:
        resp = call_openrouter(PROMPT_FLASHCARDS_JSON, prompt, max_tokens=1000, temperature=0.0, is_json=True)
        parsed = extract_json_from_text(resp)
        if isinstance(parsed, list):
            return parsed
        try:
            return json.loads(resp)
        except Exception:
            logger.debug("Flashcards parse failed")
            return []
    except Exception as e:
        logger.warning("Flashcard generation failed: %s", e)
        return []

# ----------------------------
# RAG / indexing helpers
# ----------------------------
def build_vector_index(upload_obj):
    # Accepts an upload object in session (dict) and populates embeddings and index
    chunks = upload_obj.get("chunks") or []
    model = get_embedding_model()
    try:
        if not chunks:
            upload_obj["embeddings"] = np.zeros((0, 384), dtype=np.float32)
            upload_obj["index"] = VectorIndex(upload_obj["embeddings"], [])
            upload_obj["index_built"] = False
            return upload_obj
        upload_obj["status_msg"] = "Embedding chunks..."
        emb = embed_texts(chunks, model)
        upload_obj["embeddings"] = emb
        upload_obj["index"] = VectorIndex(emb, chunks)
        upload_obj["index_built"] = True
        upload_obj["status_msg"] = f"Index built ({len(chunks)} chunks)"
        return upload_obj
    except Exception as e:
        logger.exception("Failed building vector index: %s", e)
        upload_obj["index_built"] = False
        upload_obj["status_msg"] = "Index build failed"
        return upload_obj

def answer_question_with_rag(query, uploads):
    if not query or not query.strip():
        return "Please provide a question."
    model = get_embedding_model()
    q_emb = embed_texts([query], model)
    found = []
    for u in uploads:
        idx = u.get("index")
        chunks = u.get("chunks") or []
        if not idx or getattr(idx, "dimension", 0) == 0:
            continue
        try:
            dists, ids = idx.search(q_emb, k=min(TOP_K, max(1, len(chunks))))
        except Exception as e:
            logger.warning("Index search failed: %s", e)
            continue
        for j in ids[0]:
            if j is None or int(j) < 0:
                continue
            try:
                found.append(chunks[int(j)])
            except Exception:
                continue
    if not found:
        return "Couldn't find relevant information in your documents."
    context = "\n---\n".join(found[: min(len(found), TOP_K * 3)])
    sys_prompt = "You are a precise assistant. Answer using ONLY the provided context. If not possible, say you cannot answer."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    try:
        return call_openrouter(sys_prompt, user_prompt, max_tokens=600)
    except Exception as e:
        logger.exception("RAG LLM error: %s", e)
        return f"LLM Error: {e}"

# ----------------------------
# Spaced repetition (SM-2)
# ----------------------------
def sm2_update_card(easiness, interval, reps, quality):
    if quality < 3:
        reps = 0
        interval = 1
    else:
        reps += 1
        if reps == 1:
            interval = 1
        elif reps == 2:
            interval = 6
        else:
            interval = max(1, round(interval * easiness))
    easiness = max(1.3, easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    next_review = int(time.time()) + interval * 86400
    return easiness, interval, reps, next_review

# ----------------------------
# Exports
# ----------------------------
def anki_export_tsv(upload_id, conn):
    cursor = conn.execute("SELECT question, answer FROM flashcards WHERE upload_id = ?", (upload_id,))
    rows = cursor.fetchall()
    if not rows:
        return None
    buf = io.StringIO()
    for q, a in rows:
        qs = (q or "").replace("\t", " ").replace("\n", "<br>")
        as_ = (a or "").replace("\t", " ").replace("\n", "<br>")
        buf.write(f"{qs}\t{as_}\n")
    filename = f"anki_deck_upload_{upload_id}.txt"
    return filename, buf.getvalue().encode("utf-8")

# ----------------------------
# Session & UI helpers (clean, minimal UX)
# ----------------------------
def initialize_session_state():
    defaults = {
        "uploads": [],
        "OPENROUTER_API_KEY": st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") and st.secrets is not None else os.getenv("OPENROUTER_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def handle_file_processing(upload_idx):
    try:
        upload = st.session_state["uploads"][upload_idx]
    except Exception:
        return
    filename = upload.get("filename", "file")
    try:
        upload["status_msg"] = "Extracting content..."
        slides = parse_and_extract_content(filename, upload.get("file_bytes") or b"")
        upload["slides_data"] = slides
        upload["slide_count"] = len(slides)
        # Aggregate text
        parts = []
        for s in slides:
            t = (s.get("text", "") or "").strip()
            o = (s.get("ocr_text", "") or "").strip()
            combined = "\n".join([p for p in (t, o) if p]).strip()
            if combined:
                parts.append(combined)
        upload["full_text"] = "\n\n".join(parts).strip()
        upload["status_msg"] = "Chunking..."
        upload["chunks"] = chunk_text(upload["full_text"], max_chars=1200) or ([upload["full_text"]] if upload["full_text"] else [])
        # Build index (with fallback embeddings)
        upload = build_vector_index(upload)
        # Save metadata to DB
        try:
            conn = get_db_connection()
            with conn:
                cur = conn.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)",
                                   (filename, int(time.time()), json.dumps({"slide_count": upload.get("slide_count", 0)})))
                upload["db_id"] = cur.lastrowid
        except Exception as e:
            logger.warning("Saving upload record failed: %s", e)
            upload["db_id"] = None
        upload["status_msg"] = f"Ready ({len(upload.get('chunks', []))} chunks)"
    except Exception as e:
        logger.exception("Processing failed: %s", e)
        upload["status_msg"] = f"Error: {str(e)}"

# ----------------------------
# Polished UI (simple & professional)
# ----------------------------
APP_CSS = """
/* Minimal polished theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root {
  --bg: linear-gradient(180deg,#071029 0%,#071229 100%);
  --card: rgba(255,255,255,0.03);
  --muted: #94a3b8;
  --accent: linear-gradient(90deg,#7c3aed,#5b21b6);
  --radius: 12px;
}
body, .stApp, .main {
  font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, Arial;
  background: var(--bg);
  color: #e6eef8;
}
.block-container { padding: 24px 28px !important; }
.card {
  background: var(--card);
  padding: 18px;
  border-radius: var(--radius);
  box-shadow: 0 8px 30px rgba(2,6,23,0.6);
  border: 1px solid rgba(255,255,255,0.03);
}
h1 { margin: 0 0 4px 0; font-weight:700; }
.small-muted { color: var(--muted); font-size:0.95rem; margin-bottom:12px; }
.preview { background: rgba(255,255,255,0.02); padding:12px; border-radius:10px; margin-bottom:8px; }
.stButton>button { border-radius:10px; padding:8px 14px; background: linear-gradient(90deg,#7c3aed,#5b21b6); color:#fff; border:none; }
textarea, input, .stTextInput, .stTextArea { border-radius:10px; }
.sidebar .stButton>button { background: none !important; color: #cbd5e1 !important;}
"""

def render_home():
    st.markdown(f"<div class='card'><h1>{APP_TITLE}</h1><div class='small-muted'>{APP_SUBTITLE}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><strong>Quick tips</strong><ul><li>Upload PPTX / PDF on the Upload tab</li><li>Process to extract text & build index</li><li>Use Lessons, MCQs, Flashcards to study</li></ul></div>", unsafe_allow_html=True)

def render_upload_tab():
    st.header("Upload & Process")
    st.markdown("<div class='card'>Upload PPTX or PDF files. After upload click <em>Process</em> to extract text & OCR and optionally build embeddings.</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PPTX / PDF — multiple", type=["pptx", "pdf"], accept_multiple_files=True, key="file_uploader_main")
    if uploaded_files:
        existing = {u["filename"] for u in st.session_state.uploads}
        for f in uploaded_files:
            if f.name not in existing:
                st.session_state.uploads.append({
                    "filename": f.name,
                    "file_bytes": f.getvalue(),
                    "status_msg": "Uploaded",
                    "slides_data": [],
                    "chunks": [],
                    "index": None,
                    "index_built": False,
                    "db_id": None,
                    "full_text": ""
                })
    if not st.session_state.uploads:
        st.info("No uploads yet. Use the uploader above.")
        return
    for i, up in enumerate(list(st.session_state.uploads)):  # copy to avoid iteration issues
        with st.expander(f"{up['filename']} — {up.get('status_msg','Ready')}", expanded=False):
            cols = st.columns([3,1,1])
            cols[0].markdown(f"**Slides:** {up.get('slide_count', 0)}  —  **Chunks:** {len(up.get('chunks', []))}")
            if cols[1].button("Process", key=f"proc_{_sanitize_key(up['filename'])}"):
                handle_file_processing(i)
                safe_rerun()
            if cols[2].button("Build Index", key=f"idx_{_sanitize_key(up['filename'])}", disabled=bool(up.get("index_built", False))):
                build_vector_index(up)
                safe_rerun()
            # preview area
            if up.get("slides_data"):
                preview_btn_key = f"preview_{_sanitize_key(up['filename'])}"
                if st.button("Preview slides", key=preview_btn_key):
                    for s in up.get("slides_data", [])[:20]:
                        st.markdown(f"**Slide {s['index']+1}**")
                        if s.get("text"):
                            st.write(s["text"])
                        if s.get("ocr_text"):
                            st.caption("OCR text:")
                            st.write(s["ocr_text"])
                        st.markdown("---")
            # delete
            if st.button("Delete upload", key=f"del_{_sanitize_key(up['filename'])}"):
                try:
                    dbid = up.get("db_id")
                    if dbid:
                        try:
                            conn = get_db_connection()
                            with conn:
                                conn.execute("DELETE FROM uploads WHERE id = ?", (dbid,))
                        except Exception:
                            logger.exception("Failed removing DB record")
                    # remove from session
                    st.session_state.uploads = [u for u in st.session_state.uploads if not (u.get("db_id")==dbid and u.get("filename")==up.get("filename"))]
                    safe_rerun()
                    st.success("Deleted.")
                except Exception:
                    logger.exception("Delete failed")
                    st.error("Failed to delete.")

def render_lessons_tab():
    st.header("Generate Lesson")
    choices = ["<none>"] + [u["filename"] for u in st.session_state.uploads if u.get("full_text")]
    sel = st.selectbox("Choose processed upload", choices, key="lesson_select")
    if sel == "<none>":
        st.info("Process an upload first.")
        return
    upload = next((u for u in st.session_state.uploads if u["filename"]==sel), None)
    text = upload.get("full_text") or "\n".join(upload.get("chunks") or [])
    if not text.strip():
        st.warning("No extracted text to generate lesson.")
        return
    if st.button("Generate lesson"):
        with st.spinner("Generating lesson..."):
            res = generate_multilevel_lesson(text)
            st.success("Lesson generated")
            st.text_area("Lesson", res, height=400)

def render_mcq_tab():
    st.header("Generate MCQs")
    choices = ["<none>"] + [u["filename"] for u in st.session_state.uploads if u.get("full_text")]
    sel = st.selectbox("Select document", choices, key="mcq_select")
    if sel == "<none>":
        st.info("Process an upload first.")
        return
    upload = next((u for u in st.session_state.uploads if u["filename"]==sel), None)
    n = st.slider("How many MCQs", 1, 20, 5)
    if st.button("Generate MCQs"):
        with st.spinner("Generating MCQs..."):
            mcqs = generate_mcq_set_from_text(upload.get("full_text") or "", qcount=n)
            if mcqs:
                st.json(mcqs)
                # persist
                try:
                    conn = get_db_connection()
                    now = int(time.time())
                    with conn:
                        for obj in mcqs:
                            qtext = obj.get("question","")
                            opts = obj.get("options",[])
                            ans = int(obj.get("answer_index",0))
                            conn.execute("INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                                         (upload.get("db_id"), qtext, json.dumps(opts), ans, now))
                    st.success("MCQs saved to DB")
                except Exception:
                    logger.exception("Save MCQs failed")
                    st.warning("Could not save MCQs to DB.")
            else:
                st.warning("No MCQs returned.")

def render_flashcards_tab():
    st.header("Generate Flashcards")
    choices = ["<none>"] + [u["filename"] for u in st.session_state.uploads if u.get("full_text")]
    sel = st.selectbox("Select document", choices, key="fc_select")
    if sel == "<none>":
        st.info("Process an upload first.")
        return
    upload = next((u for u in st.session_state.uploads if u["filename"]==sel), None)
    n = st.number_input("Number of flashcards", min_value=1, max_value=100, value=10)
    if st.button("Generate flashcards"):
        with st.spinner("Generating flashcards..."):
            fcs = generate_flashcards_from_text(upload.get("full_text") or "", n=n)
            if fcs:
                st.json(fcs)
                try:
                    conn = get_db_connection()
                    with conn:
                        for obj in fcs:
                            qtext = (obj.get("question") or "")[:2000]
                            ans = (obj.get("answer") or "")[:2000]
                            conn.execute("INSERT INTO flashcards (upload_id, question, answer, next_review) VALUES (?, ?, ?, ?)",
                                         (upload.get("db_id"), qtext, ans, int(time.time())))
                    st.success("Saved flashcards to DB")
                except Exception:
                    logger.exception("Save flashcards failed")
                    st.warning("Could not save flashcards to DB.")
            else:
                st.warning("No flashcards returned.")
    st.markdown("### Review due flashcards (SM-2)")
    conn = get_db_connection()
    rows = conn.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards ORDER BY next_review ASC LIMIT 10").fetchall()
    if not rows:
        st.info("No flashcards to review.")
    else:
        for (cid, q, a, eas, ivl, reps, nxt) in rows:
            with st.expander(q[:120]):
                st.write(a)
                cols = st.columns([1,1,1,1])
                quality = cols[0].radio("Quality (0-5)", [0,1,2,3,4,5], key=f"q_{cid}", index=5)
                if cols[3].button("Mark review", key=f"m_{cid}"):
                    new_eas, new_ivl, new_reps, new_next = sm2_update_card(eas, ivl, reps, int(quality))
                    with conn:
                        conn.execute("UPDATE flashcards SET easiness=?, interval=?, repetitions=?, next_review=? WHERE id=?",
                                     (float(new_eas), int(new_ivl), int(new_reps), int(new_next), cid))
                    st.success("Updated")

def render_chat_tab():
    st.header("Ask Questions (RAG)")
    query = st.text_input("Ask a question about your documents", key="rag_query")
    processed = [u for u in st.session_state.uploads if u.get("index_built")]
    if st.button("Get answer"):
        if not processed:
            st.warning("No processed uploads with index. Process & build index first.")
            return
        with st.spinner("Searching..."):
            ans = answer_question_with_rag(query, processed)
            st.markdown("**Answer**")
            st.write(ans)

def render_quizzes_tab():
    st.header("Take a Quiz")
    conn = get_db_connection()
    upload_map = {u["db_id"]: u["filename"] for u in st.session_state.uploads if u.get("db_id")}
    if not upload_map:
        st.info("No quizzes available. Generate MCQs first.")
        return
    sel = st.selectbox("Select upload quizzes (or all)", ["<all>"] + list(upload_map.values()))
    if st.button("Start quiz"):
        if sel == "<all>":
            rows = conn.execute("SELECT question, options, correct_index FROM quizzes ORDER BY RANDOM() LIMIT 10").fetchall()
        else:
            uid = next((k for k,v in upload_map.items() if v==sel), None)
            rows = conn.execute("SELECT question, options, correct_index FROM quizzes WHERE upload_id=? ORDER BY RANDOM() LIMIT 10", (uid,)).fetchall()
        if not rows:
            st.info("No quiz questions found.")
            return
        score = 0
        for idx, (q, opts_json, correct_index) in enumerate(rows):
            opts = json.loads(opts_json)
            answer = st.radio(f"Q{idx+1}: {q}", opts, key=f"quiz_{idx}")
            if st.button(f"Submit Q{idx+1}", key=f"sub_{idx}"):
                if opts.index(answer) == correct_index:
                    st.success("Correct!")
                    score += 1
                else:
                    st.error(f"Wrong — correct: {opts[correct_index]}")
        st.write("Session score:", score)

def render_exports_tab():
    st.header("Exports")
    conn = get_db_connection()
    upload_map = {u["db_id"]: u["filename"] for u in st.session_state.uploads if u.get("db_id")}
    if not upload_map:
        st.info("No DB uploads yet.")
        return
    sel = st.selectbox("Select upload to export flashcards", list(upload_map.keys()), format_func=lambda x: upload_map.get(x))
    if st.button("Export Anki TSV"):
        res = anki_export_tsv(sel, conn)
        if not res:
            st.warning("No flashcards for this upload.")
        else:
            fname, b = res
            st.download_button("Download Anki TSV", data=b, file_name=fname, mime="text/tab-separated-values")

def render_settings_tab():
    st.header("Settings")
    key = st.text_input("OpenRouter API Key (session only)", value=st.session_state.get("OPENROUTER_API_KEY",""), type="password", key="ui_api_key")
    if key:
        st.session_state["OPENROUTER_API_KEY"] = key
        st.success("API key saved in session (not persisted).")
    st.write("Embedding model:", EMBEDDING_MODEL_NAME)
    if st.button("Clear session uploads"):
        st.session_state["uploads"] = []
        st.success("Cleared uploads in session")

# ----------------------------
# App entrypoint
# ----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
    initialize_session_state()

    st.sidebar.title(APP_TITLE)
    st.sidebar.markdown(APP_SUBTITLE)
    st.sidebar.write(f"Session uploads: {len(st.session_state.uploads)}")
    nav = st.sidebar.radio("Go to", ["Home", "Upload", "Lessons", "MCQs", "Flashcards", "Chat", "Quizzes", "Exports", "Settings"])

    if nav == "Home":
        render_home()
    elif nav == "Upload":
        render_upload_tab()
    elif nav == "Lessons":
        render_lessons_tab()
    elif nav == "MCQs":
        render_mcq_tab()
    elif nav == "Flashcards":
        render_flashcards_tab()
    elif nav == "Chat":
        render_chat_tab()
    elif nav == "Quizzes":
        render_quizzes_tab()
    elif nav == "Exports":
        render_exports_tab()
    elif nav == "Settings":
        render_settings_tab()

if __name__ == "__main__":
    main()


# SlideTutor — Robust single-file Streamlit app (NO top-level import statements)
# Save as app.py and run: streamlit run app.py
# This file uses dynamic __import__ calls and provides defensive fallbacks so it runs in
# environments with missing optional libraries (sentence-transformers, PyMuPDF, python-pptx, easyocr, gTTS, faiss).
# Do NOT add top-level import statements — this file intentionally avoids them.

# ----------------------------
# Minimal dynamic imports / alias stdlib
# ----------------------------
def _try_import(name):
    try:
        return __import__(name), True
    except Exception:
        return None, False

_os = __import__("os")
_io = __import__("io")
_re = __import__("re")
_sys = __import__("sys")
_time = __import__("time")
_json = __import__("json")
_logging = __import__("logging")
_sqlite3 = __import__("sqlite3")
_hashlib = __import__("hashlib")
_typing = __import__("typing")

os = _os
io = _io
re = _re
sys = _sys
time = _time
json = _json
logging = _logging
sqlite3 = _sqlite3
hashlib = _hashlib
typing = _typing

# Third-party optional
st, _HAS_STREAMLIT = _try_import("streamlit")
np, _HAS_NUMPY = _try_import("numpy")
requests, _HAS_REQUESTS = _try_import("requests")
pptx_module, _HAS_PPTX = _try_import("pptx")
fitz, _HAS_PYMUPDF = _try_import("fitz")
easyocr_module, _HAS_EASYOCR = _try_import("easyocr")
sentence_transformers_module, _HAS_SENTENCE_TRANSFORMERS = _try_import("sentence_transformers")
faiss, _HAS_FAISS = _try_import("faiss")
gtts_module, _HAS_GTTS = _try_import("gtts")
PIL_module, _HAS_PIL = _try_import("PIL")

# Map names and classes
easyocr = easyocr_module if _HAS_EASYOCR else None
SentenceTransformer = None
if _HAS_SENTENCE_TRANSFORMERS and sentence_transformers_module is not None:
    try:
        SentenceTransformer = sentence_transformers_module.SentenceTransformer
    except Exception:
        SentenceTransformer = None

# require streamlit + numpy
if not _HAS_STREAMLIT:
    raise RuntimeError("Streamlit is required. Install: pip install streamlit")
if not _HAS_NUMPY:
    raise RuntimeError("NumPy is required. Install: pip install numpy")

# numpy alias
np = np

# ----------------------------
# Logging & configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("slidetutor")

DEFAULT_OPENROUTER_KEY = (
    (st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") and st.secrets is not None else None)
    or os.getenv("OPENROUTER_API_KEY", "")
)
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
DB_PATH = os.getenv("SLIDETUTOR_DB", "slidetutor.sqlite3")
APP_TITLE = "SlideTutor AI"
APP_SUBTITLE = "Beautiful • Simple • Smart — Learn from slides quickly."

# ----------------------------
# Utility helpers
# ----------------------------
def _sanitize_key(s):
    try:
        if s is None:
            return "none"
        s = str(s)
        return re.sub(r'[^0-9a-zA-Z]', '_', s)[:64]
    except Exception:
        return "key"

def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun") and callable(getattr(st, "experimental_rerun")):
            st.experimental_rerun()
        else:
            st.session_state["_rerun_token"] = st.session_state.get("_rerun_token", 0) + 1
    except Exception:
        st.session_state["_rerun_token"] = st.session_state.get("_rerun_token", 0) + 1

# ----------------------------
# DB connection (safe)
# ----------------------------
@st.cache_resource
def get_db_connection(path: str = DB_PATH):
    try:
        folder = os.path.dirname(os.path.abspath(path))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
    except Exception:
        pass
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn: sqlite3.Connection):
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at INTEGER NOT NULL,
                meta TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                easiness REAL DEFAULT 2.5,
                interval INTEGER DEFAULT 1,
                repetitions INTEGER DEFAULT 0,
                next_review INTEGER,
                FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id TEXT,
                question TEXT NOT NULL,
                options TEXT NOT NULL,
                correct_index INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
            );
        """)

# Initialize DB
try:
    db_conn = get_db_connection()
    init_db(db_conn)
except Exception as e:
    logger.exception("DB init failed: %s", e)
    # proceed — session fallback will be used

# ----------------------------
# Vector index (FAISS optional) and embedding utilities
# ----------------------------
_NORM_EPS = 1e-9

def _ensure_2d(arr):
    arr = np.asarray(arr)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr

def _safe_normalize(arr, axis=1):
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm = np.where(norm < _NORM_EPS, 1.0, norm)
    return arr / norm

class VectorIndex:
    """Wrapper using FAISS if available, otherwise NumPy-based cosine search."""
    def __init__(self, embeddings, texts):
        self.texts = list(texts or [])
        if embeddings is None:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            arr = np.asarray(embeddings, dtype=np.float32)
            if arr.ndim == 1:
                arr = _ensure_2d(arr)
            self.embeddings = arr
        self.dimension = self.embeddings.shape[1] if self.embeddings.size else 0
        self._normed = _safe_normalize(self.embeddings) if self.embeddings.size else np.zeros_like(self.embeddings)
        self._use_faiss = False
        self._faiss_index = None
        if _HAS_FAISS and faiss is not None and self.embeddings.size:
            try:
                idx = faiss.IndexFlatIP(self.dimension)
                idx.add(np.ascontiguousarray(self._normed))
                self._faiss_index = idx
                self._use_faiss = True
                logger.info("Built FAISS index (%d vectors, dim=%d)", self.embeddings.shape[0], self.dimension)
            except Exception as e:
                logger.warning("FAISS init failed - falling back to NumPy: %s", e)
                self._use_faiss = False

    def search(self, query_embeddings, k):
        q = _ensure_2d(np.asarray(query_embeddings, dtype=np.float32))
        if self._use_faiss and self._faiss_index is not None and self.dimension:
            try:
                qn = np.ascontiguousarray(_safe_normalize(q))
                sims, ids = self._faiss_index.search(qn, k)
                dists = 1.0 - sims.astype(np.float32)
                return dists, ids.astype(np.int64)
            except Exception as e:
                logger.warning("FAISS search failed, fallback to numpy: %s", e)
        return self._numpy_search(q, k)

    def _numpy_search(self, queries, k):
        if self.embeddings.size == 0:
            return np.full((queries.shape[0], k), np.inf, dtype=np.float32), np.full((queries.shape[0], k), -1, dtype=np.int64)
        nq = _safe_normalize(queries)
        sims = nq @ self._normed.T
        k_eff = min(k, sims.shape[1])
        idxs = np.argsort(-sims, axis=1)[:, :k_eff]
        final_sims = np.take_along_axis(sims, idxs, axis=1)
        if k_eff < k:
            pad_idxs = np.full((sims.shape[0], k - k_eff), -1, dtype=np.int64)
            pad_dists = np.full((sims.shape[0], k - k_eff), np.inf, dtype=np.float32)
            idxs = np.concatenate([idxs, pad_idxs], axis=1)
            final_sims = np.concatenate([final_sims, pad_dists], axis=1)
        return 1.0 - final_sims.astype(np.float32), idxs.astype(np.int64)

# Embedding model loader (cached)
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    if not _HAS_SENTENCE_TRANSFORMERS or SentenceTransformer is None:
        return None
    try:
        model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)
        return model
    except Exception as e:
        logger.warning("Failed to load embedding model: %s", e)
        return None

# Deterministic fallback embeddings (for offline behavior)
def _deterministic_fallback_embedding(texts, dim=384):
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.md5((t or "").encode("utf-8")).digest()
        vals = []
        while len(vals) < dim:
            h = hashlib.md5(h).digest()
            vals.extend([b / 255.0 for b in h])
        arr = np.asarray(vals[:dim], dtype=np.float32)
        arr = arr / (np.linalg.norm(arr) + 1e-9)
        out[i] = arr
    return out

def embed_texts(texts, model):
    if not texts:
        dim = 384
        try:
            if model is not None and hasattr(model, "get_sentence_embedding_dimension"):
                dim = model.get_sentence_embedding_dimension()
        except Exception:
            pass
        return np.zeros((0, dim), dtype=np.float32)
    if model is None:
        return _deterministic_fallback_embedding(texts, dim=384)
    try:
        enc = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return _ensure_2d(enc.astype(np.float32))
    except Exception as e:
        logger.exception("Embedding failed; using deterministic fallback: %s", e)
        return _deterministic_fallback_embedding(texts, dim=384)

# ----------------------------
# OCR & file parsing (pptx/pdf) with fallbacks
# ----------------------------
@st.cache_resource(show_spinner="Loading OCR model...")
def get_easyocr_reader(lang_list=None):
    if lang_list is None:
        lang_list = ["en"]
    if not _HAS_EASYOCR or easyocr is None:
        return None
    try:
        reader = easyocr.Reader(lang_list, gpu=False)
        logger.info("EasyOCR reader initialized for: %s", lang_list)
        return reader
    except Exception as e:
        logger.warning("EasyOCR init failed: %s", e)
        return None

def _bytes_to_pil_image(b):
    if not b:
        return None
    if _HAS_PIL and PIL_module is not None:
        try:
            Image = PIL_module.Image
            return Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            return None
    return None

def ocr_image_bytes(image_bytes, reader):
    if not image_bytes or reader is None:
        return ""
    try:
        pil_img = _bytes_to_pil_image(image_bytes)
        if pil_img is not None:
            arr = np.array(pil_img)
            out = reader.readtext(arr, detail=0, paragraph=True)
        else:
            try:
                out = reader.readtext(image_bytes, detail=0, paragraph=True)
            except Exception:
                out = []
        if not out:
            return ""
        if isinstance(out, list):
            return "\n".join([str(x).strip() for x in out if str(x).strip()])[:10000]
        return str(out).strip()
    except Exception as e:
        logger.debug("OCR failed: %s", e)
        return ""

def _extract_from_pptx(file_bytes):
    if not _HAS_PPTX or pptx_module is None:
        raise ModuleNotFoundError("python-pptx not installed.")
    slides = []
    Presentation = pptx_module.Presentation
    prs = Presentation(io.BytesIO(file_bytes))
    for i, slide in enumerate(prs.slides):
        texts = []
        images = []
        for shape in slide.shapes:
            try:
                if getattr(shape, "has_text_frame", False) and getattr(shape, "text", None):
                    txt = (shape.text or "").strip()
                    if txt:
                        texts.append(txt)
            except Exception:
                pass
            try:
                img = getattr(shape, "image", None)
                if img is not None and getattr(img, "blob", None) is not None:
                    images.append(img.blob)
            except Exception:
                pass
        slides.append({"index": i, "text": "\n".join(texts), "images": images})
    return slides

def _extract_from_pdf(file_bytes):
    if not _HAS_PYMUPDF or fitz is None:
        raise ModuleNotFoundError("PyMuPDF not installed.")
    slides = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        text = (page.get_text("text") or "").strip()
        images = []
        try:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    d = doc.extract_image(xref)
                    if d and "image" in d:
                        images.append(d["image"])
                except Exception:
                    logger.debug("Failed extract image xref=%s page=%s", xref, i)
        except Exception:
            logger.debug("get_images failed on page %s", i)
        if not text and not images:
            try:
                pix = page.get_pixmap(dpi=150)
                images.append(pix.tobytes("png"))
            except Exception:
                pass
        slides.append({"index": i, "text": text, "images": images})
    try:
        doc.close()
    except Exception:
        pass
    return slides

def parse_and_extract_content(filename, file_bytes):
    """Parse PPTX or PDF bytes into slides with OCR."""
    try:
        ext = os.path.splitext(filename)[1].lower()
    except Exception:
        logger.exception("Bad filename: %r", filename)
        return []
    try:
        if ext == ".pptx":
            raw = _extract_from_pptx(file_bytes)
        elif ext == ".pdf":
            raw = _extract_from_pdf(file_bytes)
        else:
            logger.warning("Unsupported extension: %s", ext)
            return []
    except Exception as e:
        try:
            st.error(f"Failed to parse '{filename}': {e}")
        except Exception:
            pass
        logger.exception("Parse failed for %s: %s", filename, e)
        return []
    reader = get_easyocr_reader(["en"])
    processed = []
    total = max(1, len(raw))
    progress = None
    try:
        progress = st.progress(0, text="Extracting content...")
    except Exception:
        progress = None
    for i, s in enumerate(raw):
        try:
            imgs = s.get("images") or []
            ocr_texts = []
            for img in imgs:
                try:
                    if not img:
                        continue
                    t = ocr_image_bytes(img, reader)
                    if t:
                        ocr_texts.append(t)
                except Exception:
                    continue
            processed.append({
                "index": int(s.get("index", i)),
                "text": (s.get("text") or "").strip(),
                "images": imgs,
                "ocr_text": "\n".join(ocr_texts).strip()
            })
            if progress is not None:
                try:
                    progress.progress(min(1.0, float(i+1)/float(total)), text=f"Processing {i+1}/{total}")
                except Exception:
                    try:
                        progress.progress(min(1.0, float(i+1)/float(total)))
                    except Exception:
                        pass
        except Exception as e:
            logger.exception("Slide processing failed: %s", e)
            continue
    return processed

# ----------------------------
# JSON extraction helper (robust)
# ----------------------------
def extract_json_from_text(text):
    if not text or not isinstance(text, str):
        return None
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    candidate = None
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            candidate = None
    start_pos = None
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            start_pos = i
            break
    if start_pos is None:
        return None
    opening = {"{": "}", "[": "]"}
    stack = [opening[text[start_pos]]]
    i = start_pos
    in_string = False
    escape = False
    candidate = None
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = False
        else:
            if ch == '"' or ch == "'":
                in_string = ch
            elif ch in ("{", "["):
                stack.append(opening[ch])
            elif ch in ("}", "]"):
                if not stack:
                    break
                expected = stack.pop()
                if ch != expected:
                    break
                if not stack:
                    candidate = text[start_pos:i+1]
                    break
        i += 1
    if not candidate:
        m2 = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if not m2:
            return None
        candidate = m2.group(0)
    def clean_trailing_commas(s):
        return re.sub(r',\s*(\}|])', r'\1', s)
    attempts = 0
    s = candidate
    while attempts < 3:
        try:
            return json.loads(s)
        except Exception:
            s = clean_trailing_commas(s)
            attempts += 1
    logger.debug("Failed parsing JSON candidate: %s", candidate[:200].replace("\n", " "))
    return None

# ----------------------------
# LLM / OpenRouter wrapper (defensive)
# ----------------------------
def call_openrouter(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=1200, temperature=0.1):
    api_key = None
    try:
        api_key = st.session_state.get("OPENROUTER_API_KEY") or DEFAULT_OPENROUTER_KEY
    except Exception:
        api_key = DEFAULT_OPENROUTER_KEY
    if not api_key:
        raise RuntimeError("OpenRouter API key not configured.")
    if not _HAS_REQUESTS or requests is None:
        raise RuntimeError("requests library not available.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}], "max_tokens": max_tokens, "temperature": temperature}
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
    except Exception as e:
        logger.exception("Network exception when calling OpenRouter: %s", e)
        raise RuntimeError("Network error contacting LLM: " + str(e))
    try:
        resp.raise_for_status()
    except Exception:
        raw = "<no-body>"
        try:
            raw = resp.text
        except Exception:
            pass
        logger.error("OpenRouter returned HTTP %s: %s", resp.status_code, raw[:1000])
        raise RuntimeError(f"LLM API returned HTTP {resp.status_code}: {raw[:1000]}")
    try:
        data = resp.json()
    except Exception:
        logger.exception("Failed to parse LLM JSON response")
        raise RuntimeError("Failed to decode LLM response as JSON.")
    content = None
    try:
        choices = data.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message") or {}
                content = msg.get("content") or first.get("text") or first.get("message", {}).get("content")
        if not content:
            if "output" in data:
                content = data["output"]
            elif "result" in data:
                content = data["result"]
    except Exception:
        content = None
    if not content:
        logger.error("LLM missing expected fields; response truncated: %s", str(data)[:2000])
        raise RuntimeError("LLM response missing expected content.")
    if not isinstance(content, str):
        try:
            content = json.dumps(content)
        except Exception:
            content = str(content)
    return content.strip()

# ----------------------------
# Content generation wrappers
# ----------------------------
PROMPT_LESSON_MULTILEVEL = (
    "You are an expert teacher. Create a multi-level lesson (Beginner, Intermediate, Advanced) "
    "based on the provided text. For each level: explanation, worked example, 2-3 tips. Finally add a 3-question quiz."
)
PROMPT_MCQ_JSON = (
    "You are an AI that generates MCQs. Reply only with a valid JSON array of objects with keys: "
    "'question' (string), 'options' (array of 4 strings), 'answer_index' (0-based integer)."
)
PROMPT_FLASHCARDS_JSON = (
    "You are an AI that creates flashcards. Reply only with a valid JSON array of objects with keys: "
    "'question' and 'answer'."
)

def generate_multilevel_lesson(context: str):
    prompt = f"{PROMPT_LESSON_MULTILEVEL}\n\nTEXT:\n{context}"
    try:
        return call_openrouter("You are a helpful teacher.", prompt, max_tokens=1500, temperature=0.2)
    except Exception as e:
        logger.exception("Lesson generation failed: %s", e)
        return f"(LLM error) {e}"

def generate_mcq_set_from_text(context: str, qcount: int = 5):
    if not context:
        return []
    prompt = f"Create exactly {qcount} MCQs from this text:\n\n{context}"
    try:
        resp = call_openrouter(PROMPT_MCQ_JSON, prompt, max_tokens=800, temperature=0.0)
        parsed = extract_json_from_text(resp)
        if isinstance(parsed, list):
            return parsed
        try:
            return json.loads(resp)
        except Exception:
            logger.debug("MCQ parse failed")
            return []
    except Exception as e:
        logger.warning("MCQ generation failed: %s", e)
        return []

def generate_flashcards_from_text(context: str, n: int = 10):
    if not context:
        return []
    prompt = f"Extract up to {n} flashcards from the text below.\n\n{context}"
    try:
        resp = call_openrouter(PROMPT_FLASHCARDS_JSON, prompt, max_tokens=1000, temperature=0.0)
        parsed = extract_json_from_text(resp)
        if isinstance(parsed, list):
            return parsed
        try:
            return json.loads(resp)
        except Exception:
            logger.debug("Flashcards parse failed")
            return []
    except Exception as e:
        logger.warning("Flashcard generation failed: %s", e)
        return []

# ----------------------------
# SM-2 spaced repetition (robust)
# ----------------------------
def sm2_update_card(easiness=2.5, interval=1, repetitions=0, quality=0):
    """
    Update a flashcard record using SM-2.
    Returns: (new_easiness, new_interval, new_repetitions, next_review_timestamp)
    """
    try:
        quality = int(round(float(quality)))
    except Exception:
        quality = 0
    quality = max(0, min(5, quality))
    try:
        easiness = float(easiness)
    except Exception:
        easiness = 2.5
    try:
        interval = int(interval)
    except Exception:
        interval = 1
    try:
        repetitions = int(repetitions)
    except Exception:
        repetitions = 0

    if quality < 3:
        repetitions = 0
        interval = 1
    else:
        repetitions += 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = max(1, round(interval * easiness))

    easiness += (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    easiness = max(1.3, easiness)

    next_review = int(time.time()) + int(interval) * 86400
    return easiness, interval, repetitions, next_review

# ----------------------------
# Exports & TTS
# ----------------------------
def anki_export_tsv(upload_id, conn=None):
    try:
        rows = []
        if conn is not None:
            cur = conn.execute("SELECT question, answer FROM flashcards WHERE upload_id = ?", (upload_id,))
            rows = cur.fetchall()
        else:
            fc_map = st.session_state.get("flashcards_db", {})
            rows = [(f.get("question"), f.get("answer")) for f in fc_map.get(upload_id, [])]
        if not rows:
            try:
                st.info("No flashcards found for this upload to export.")
            except Exception:
                pass
            return None
        buf = io.StringIO()
        for q, a in rows:
            q_s = (q or "").replace("\t", " ").replace("\n", "<br>")
            a_s = (a or "").replace("\t", " ").replace("\n", "<br>")
            buf.write(f"{q_s}\t{a_s}\n")
        bts = buf.getvalue().encode("utf-8")
        filename = f"slidetutor_anki_deck_{upload_id}.txt"
        return filename, bts
    except Exception as e:
        logger.exception("anki_export_tsv failed for upload %s: %s", upload_id, e)
        try:
            st.error("An error occurred during export. See logs for details.")
        except Exception:
            pass
        return None

def text_to_speech(text, lang="en"):
    if not text or not str(text).strip():
        raise ValueError("No text provided for TTS.")
    txt = str(text).strip()
    if _HAS_GTTS and gtts_module is not None:
        try:
            mp3_fp = io.BytesIO()
            tts_obj = gtts_module.gTTS(text=txt, lang=lang, slow=False)
            tts_obj.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return "lesson_audio.mp3", mp3_fp.getvalue()
        except Exception as e:
            logger.warning("gTTS failed: %s — falling back to WAV silent audio", e)
    # silent WAV fallback
    try:
        sample_rate = 22050
        duration_seconds = max(1, min(30, int(len(txt) / 50)))
        num_samples = sample_rate * duration_seconds
        byte_rate = sample_rate * 2
        data_size = num_samples * 2
        header = b"RIFF" + (36 + data_size).to_bytes(4, "little") + b"WAVE"
        header += b"fmt " + (16).to_bytes(4, "little")
        header += (1).to_bytes(2, "little")
        header += (1).to_bytes(2, "little")
        header += (sample_rate).to_bytes(4, "little")
        header += (byte_rate).to_bytes(4, "little")
        header += (2).to_bytes(2, "little")
        header += (16).to_bytes(2, "little")
        header += b"data" + (data_size).to_bytes(4, "little")
        silence = b"\x00" * data_size
        wav_bytes = header + silence
        return "lesson_audio.wav", wav_bytes
    except Exception as e:
        logger.exception("Failed to make fallback WAV for TTS: %s", e)
        raise RuntimeError("Failed to generate audio.") from e

# ----------------------------
# Upload processing & index building helpers
# ----------------------------
def _safe_read_file_bytes(uploaded_file):
    try:
        if hasattr(uploaded_file, "getvalue"):
            return uploaded_file.getvalue()
        if hasattr(uploaded_file, "read"):
            uploaded_file.seek(0)
            return uploaded_file.read()
    except Exception:
        try:
            return bytes(uploaded_file)
        except Exception:
            logger.exception("Could not read uploaded file bytes.")
            return b""
    return b""

def process_new_upload_safe(uploaded_file):
    try:
        file_bytes = _safe_read_file_bytes(uploaded_file)
        filename = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", None) or f"upload_{int(time.time())}"
    except Exception as e:
        logger.exception("process_new_upload_safe: failed to read: %s", e)
        raise RuntimeError("Failed to read uploaded file.") from e
    upload = {
        "filename": filename,
        "file_bytes": file_bytes,
        "slide_count": 0,
        "slides_data": [],
        "full_text": "",
        "chunks": [],
        "index": None,
        "embeddings": None,
        "index_built": False,
        "status_msg": "Uploaded (not processed)",
        "db_id": None,
        "processed": False,
    }
    try:
        slides = parse_and_extract_content(filename, file_bytes) or []
        upload["slides_data"] = slides
        upload["slide_count"] = len(slides)
        parts = []
        for s in slides:
            t = (s.get("text") or "").strip()
            o = (s.get("ocr_text") or "").strip()
            combined = "\n".join([p for p in (t, o) if p]).strip()
            if combined:
                parts.append(combined)
        upload["full_text"] = "\n\n".join(parts).strip()
        upload["chunks"] = chunk_text(upload["full_text"], max_chars=1200) or ([upload["full_text"]] if upload["full_text"] else [])
        upload["status_msg"] = f"Parsed: {upload['slide_count']} slides, {len(upload['chunks'])} chunks"
        upload["processed"] = True
    except Exception as e:
        logger.exception("process_new_upload_safe: parsing failed for %s: %s", filename, e)
        upload["status_msg"] = f"Parsing error: {e}"
        upload["processed"] = False
    # persist minimal metadata
    try:
        conn = get_db_connection()
        with conn:
            cur = conn.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)",
                               (upload["filename"], int(time.time()), json.dumps({"slide_count": upload["slide_count"]})))
            upload["db_id"] = cur.lastrowid
    except Exception:
        # session fallback
        if "uploads_meta" not in st.session_state:
            st.session_state["uploads_meta"] = {}
        st.session_state["uploads_meta"][upload["filename"]] = {"slide_count": upload["slide_count"]}
    return upload

def build_vector_index_safe(upload, force=False):
    if not isinstance(upload, dict):
        raise ValueError("upload must be a dict")
    if upload.get("index_built") and not force:
        upload["status_msg"] = "Index already built"
        return upload
    chunks = upload.get("chunks") or []
    if not chunks:
        upload["status_msg"] = "No text chunks to index"
        upload["index_built"] = False
        return upload
    model = get_embedding_model()
    if model is None:
        upload["index_built"] = False
        upload["status_msg"] = "Embeddings unavailable (no sentence-transformers)"
        logger.info("Embeddings unavailable for index build")
        return upload
    try:
        embeddings = embed_texts(chunks, model)
        upload["embeddings"] = embeddings
        idx = VectorIndex(embeddings, chunks)
        upload["index"] = idx
        upload["index_built"] = True
        try:
            vec_count = embeddings.shape[0] if hasattr(embeddings, "shape") else (len(embeddings) if embeddings else 0)
            upload["status_msg"] = f"Index Ready ({vec_count} vectors)"
        except Exception:
            upload["status_msg"] = "Index Ready"
    except Exception as e:
        logger.exception("build_vector_index_safe failed: %s", e)
        upload["index_built"] = False
        upload["status_msg"] = f"Index build failed: {e}"
    return upload

# ----------------------------
# Chunking (robust)
# ----------------------------
def chunk_text(text, max_chars=1000):
    if not text or not isinstance(text, str):
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(p) <= max_chars:
            if not current:
                current = p + "\n"
            else:
                if len(current) + len(p) + 1 <= max_chars:
                    current += p + "\n"
                else:
                    chunks.append(current.strip())
                    current = p + "\n"
        else:
            try:
                sentences = re.split(r'(?<=[\.\?\!])\s+', p)
            except Exception:
                sentences = [p[i:i+max_chars] for i in range(0, len(p), max_chars)]
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(s) > max_chars:
                    for j in range(0, len(s), max_chars):
                        block = s[j:j+max_chars].strip()
                        if not block:
                            continue
                        if current and len(current) + len(block) + 1 <= max_chars:
                            current += block + " "
                        else:
                            if current:
                                chunks.append(current.strip())
                                current = ""
                            chunks.append(block)
                else:
                    if not current:
                        current = s + " "
                    elif len(current) + len(s) + 1 <= max_chars:
                        current += s + " "
                    else:
                        chunks.append(current.strip())
                        current = s + " "
            if current:
                chunks.append(current.strip())
                current = ""
    if current:
        chunks.append(current.strip())
    final = [c.strip() for c in chunks if c and c.strip()]
    return final

# ----------------------------
# Flashcards & SM-2 DB helpers
# ----------------------------
def add_flashcards_to_db_safe(upload, cards):
    if not isinstance(cards, (list, tuple)) or not cards:
        return 0
    saved = 0
    try:
        conn = get_db_connection()
        with conn:
            uid = upload.get("db_id") or upload.get("filename")
            for c in cards:
                q = (c.get("question") or "").strip()
                a = (c.get("answer") or "").strip()
                if not q or not a:
                    continue
                conn.execute("INSERT INTO flashcards (upload_id, question, answer, next_review) VALUES (?, ?, ?, ?)",
                             (str(uid), q, a, int(time.time())))
                saved += 1
        return saved
    except Exception:
        # session fallback
        db = st.session_state.setdefault("flashcards_db", {})
        uid = upload.get("db_id") or upload.get("filename") or f"upload_{int(time.time())}"
        arr = db.setdefault(str(uid), [])
        for c in cards:
            q = (c.get("question") or "").strip()
            a = (c.get("answer") or "").strip()
            if not q or not a:
                continue
            arr.append({
                "id": f"{uid}-{len(arr)+1}",
                "upload_id": str(uid),
                "question": q,
                "answer": a,
                "easiness": 2.5,
                "interval": 1,
                "repetitions": 0,
                "next_review": int(time.time())
            })
            saved += 1
        st.session_state["flashcards_db"] = db
    return saved

def get_due_flashcards_safe(upload=None, limit=100):
    now = int(time.time())
    results = []
    try:
        conn = get_db_connection()
        if upload:
            uid = upload.get("db_id") or upload.get("filename")
            rows = conn.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards WHERE upload_id = ? AND IFNULL(next_review,0) <= ? ORDER BY next_review ASC LIMIT ?", (str(uid), now, limit)).fetchall()
        else:
            rows = conn.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards WHERE IFNULL(next_review,0) <= ? ORDER BY next_review ASC LIMIT ?", (now, limit)).fetchall()
        for r in rows:
            results.append({"id": r[0], "question": r[1], "answer": r[2], "easiness": r[3] or 2.5, "interval": r[4] or 1, "repetitions": r[5] or 0, "next_review": r[6] or now, "upload_id": upload.get("db_id") if upload else None})
        return results
    except Exception:
        # session fallback
        db = st.session_state.get("flashcards_db", {})
        for uid, arr in db.items():
            if upload and str(uid) != str(upload.get("db_id") or upload.get("filename")):
                continue
            for c in arr:
                if int(c.get("next_review", 0)) <= now:
                    results.append(c)
                    if len(results) >= limit:
                        return results
    return results

def update_flashcard_review_safe(card, quality):
    try:
        q = int(quality)
    except Exception:
        q = 0
    q = max(0, min(5, q))
    eas = float(card.get("easiness", 2.5))
    ivl = int(card.get("interval", 1))
    reps = int(card.get("repetitions", 0))
    try:
        new_eas, new_ivl, new_reps, new_next = sm2_update_card(eas, ivl, reps, q)
    except Exception:
        new_eas, new_ivl, new_reps, new_next = (2.5, 1, 0, int(time.time()) + 86400)
    try:
        conn = get_db_connection()
        if isinstance(card.get("id"), int) or (isinstance(card.get("id"), str) and card.get("id").isdigit()):
            with conn:
                conn.execute("UPDATE flashcards SET easiness=?, interval=?, repetitions=?, next_review=? WHERE id=?",
                             (float(new_eas), int(new_ivl), int(new_reps), int(new_next), int(card.get("id"))))
            return True
    except Exception:
        pass
    # session fallback
    try:
        db = st.session_state.get("flashcards_db", {})
        uid = card.get("upload_id") or str(card.get("id")).split("-")[0]
        arr = db.get(str(uid), [])
        for c in arr:
            if c.get("id") == card.get("id"):
                c["easiness"] = float(new_eas)
                c["interval"] = int(new_ivl)
                c["repetitions"] = int(new_reps)
                c["next_review"] = int(new_next)
                st.session_state["flashcards_db"] = db
                return True
    except Exception:
        logger.exception("update_flashcard_review_safe failed in session fallback")
    return False

# ----------------------------
# RAG Q&A safe
# ----------------------------
def answer_question_with_rag_safe(query, indexed_uploads, top_k=None):
    if not query or not isinstance(query, str):
        return "Please enter a question."
    top_k = int(top_k or TOP_K)
    found_chunks = []
    for upload in (indexed_uploads or []):
        if not upload:
            continue
        chunks = upload.get("chunks") or []
        idx = upload.get("index")
        if idx and hasattr(idx, "search"):
            try:
                model = get_embedding_model()
                q_emb = embed_texts([query], model) if model is not None else None
                if q_emb is not None:
                    dists, ids = idx.search(q_emb, k=min(top_k, max(1, len(chunks))))
                    for j in (ids[0] if isinstance(ids[0], (list, tuple, np.ndarray)) else ids[0]):
                        if j is None or int(j) < 0:
                            continue
                        try:
                            found_chunks.append(chunks[int(j)])
                        except Exception:
                            continue
                    continue
            except Exception:
                logger.debug("Embeddings search failed for upload; falling back to keyword-scan")
        # fallback keyword overlap
        q_tokens = set(w.lower() for w in re.findall(r"\w+", query))
        scored = []
        for i, c in enumerate(chunks):
            c_tokens = set(w.lower() for w in re.findall(r"\w+", (c or "")[:1000]))
            score = len(q_tokens & c_tokens)
            if score > 0:
                scored.append((score, i, c))
        scored.sort(reverse=True)
        for sc, i, c in scored[:top_k]:
            found_chunks.append(c)
    if not found_chunks:
        return "I couldn't find relevant information in your documents."
    context = "\n---\n".join(found_chunks[: max(1, top_k * 2) ])
    system_prompt = "You are a concise assistant. Answer using ONLY the provided context. If the answer is not present, say you cannot answer."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    try:
        return call_openrouter(system_prompt, user_prompt, max_tokens=600)
    except Exception as e:
        logger.exception("RAG LLM call failed: %s", e)
        sample = "\n\n".join(found_chunks[:3])
        preview = sample[:1500] + ("..." if len(sample) > 1500 else "")
        return f"(LLM unavailable) Relevant context:\n\n{preview}"

# ----------------------------
# UI: polished, simple, professional
# ----------------------------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root{
  --bg:#071025;
  --card:#071424;
  --muted:#9AA6B2;
  --accent:#6C5CE7;
  --radius:12px;
}
body, .stApp {
  font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, Arial;
  background: linear-gradient(180deg,#051022 0%, #061428 100%);
  color: #E6F0FA;
}
.block-container { padding: 26px 28px !important; }
.card { background: rgba(255,255,255,0.02); border-radius: var(--radius); padding: 16px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 30px rgba(2,6,12,0.6); }
.small-muted { color: var(--muted); font-size: 0.95rem; }
.stButton>button { border-radius: 10px !important; padding: 8px 14px !important; background: linear-gradient(90deg,var(--accent), #4b2bd0) !important; color: #fff !important; border: none !important; box-shadow: 0 6px 18px rgba(108,92,231,0.14); }
textarea, input, .stTextInput, .stTextArea { border-radius: 10px !important; }
.preview-slide { background: rgba(255,255,255,0.02); border-radius: 10px; padding: 8px; margin-bottom: 8px; }
"""

def initialize_session_state_defaults():
    defaults = {
        "uploads": [],
        "OPENROUTER_API_KEY": DEFAULT_OPENROUTER_KEY,
        "chat_history": [],
        "due_cards": [],
        "current_card_idx": 0,
        "active_upload_idx": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render_header():
    try:
        st.markdown(f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:12px'>"
                    f"<div style='font-size:2.2rem'>🎓</div>"
                    f"<div><div style='font-weight:700;font-size:1.4rem'>{APP_TITLE}</div>"
                    f"<div class='small-muted'>{APP_SUBTITLE}</div></div></div>", unsafe_allow_html=True)
    except Exception:
        pass

def render_home():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Welcome to SlideTutor — your elegant study companion")
    st.markdown("<p class='small-muted'>Upload PDFs or PPTX, extract content, build semantic indexes, generate lessons, quizzes and flashcards, and practice with SM-2.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>Quick tips:<ul><li>Upload files in Upload tab</li><li>Process & build index to enable RAG Q&A</li><li>Generate flashcards and practice them in Flashcards tab</li></ul></div>", unsafe_allow_html=True)

def render_upload_tab():
    st.header("Upload & Process")
    st.markdown("<div class='card'>Upload PPTX or PDF files. Then process to extract text & OCR, optionally build embeddings (if available).</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PPTX / PDF — multiple", type=["pptx", "pdf"], accept_multiple_files=True, key="uploader")
    if uploaded_files:
        existing = {u["filename"] for u in st.session_state.uploads}
        for f in uploaded_files:
            if f.name not in existing:
                up = process_new_upload_safe(f)
                st.session_state.uploads.append(up)
                st.success(f"Added: {f.name}")
                safe_rerun()
    if not st.session_state.uploads:
        st.info("No uploads yet. Use the uploader above.")
        return
    for i, up in enumerate(list(st.session_state.uploads)):
        with st.expander(f"{up['filename']} — {up.get('status_msg','Ready')}", expanded=False):
            cols = st.columns([3,1,1])
            cols[0].markdown(f"**Slides:** {up.get('slide_count',0)}  —  **Chunks:** {len(up.get('chunks',[]))}")
            if cols[1].button("Build Index", key=f"build_{_sanitize_key(up['filename'])}", disabled=bool(up.get("index_built", False))):
                build_vector_index_safe(up)
                st.success(up.get("status_msg"))
                safe_rerun()
            if cols[2].button("Select", key=f"select_{_sanitize_key(up['filename'])}"):
                st.session_state.active_upload_idx = i
                st.success("Selected for operations")
                safe_rerun()
            if up.get("slides_data"):
                if st.button("Preview (first 10 slides)", key=f"preview_{_sanitize_key(up['filename'])}"):
                    for s in up.get("slides_data", [])[:10]:
                        st.markdown(f"**Slide {s['index']+1}**")
                        if s.get("text"):
                            st.write(s["text"])
                        if s.get("ocr_text"):
                            st.caption("OCR text:")
                            st.write(s["ocr_text"])
                        st.markdown("---")
            if st.button("Delete upload", key=f"del_{_sanitize_key(up['filename'])}"):
                try:
                    dbid = up.get("db_id")
                    if dbid:
                        try:
                            conn = get_db_connection()
                            with conn:
                                conn.execute("DELETE FROM uploads WHERE id = ?", (dbid,))
                        except Exception:
                            logger.exception("Failed deleting DB record")
                    st.session_state.uploads = [u for u in st.session_state.uploads if u.get("filename") != up.get("filename")]
                    st.success("Deleted upload")
                    safe_rerun()
                except Exception:
                    logger.exception("Delete failed")
                    st.error("Failed to delete upload.")

def render_lessons_tab():
    st.header("Generate Lesson")
    uploads = [u for u in st.session_state.uploads if u.get("processed")]
    if not uploads:
        st.info("Process an upload first on the Upload tab.")
        return
    opts = {u["filename"]: idx for idx, u in enumerate(uploads)}
    sel = st.selectbox("Select a processed document", list(opts.keys()))
    up = uploads[opts[sel]]
    if st.button("Generate Multi-level Lesson"):
        with st.spinner("Generating lesson..."):
            out = generate_multilevel_lesson(up.get("full_text",""))
            st.markdown(out)

def render_chat_tab():
    st.header("Ask Questions (RAG)")
    uploads = [u for u in st.session_state.uploads if u.get("index_built")]
    if not uploads:
        st.warning("No indexed uploads. Build an index on the Upload tab to enable RAG.")
        return
    prompt = st.text_input("Ask a question about your documents", key="rag_input")
    if st.button("Get answer"):
        if not prompt or not prompt.strip():
            st.warning("Provide a question.")
        else:
            with st.spinner("Searching..."):
                ans = answer_question_with_rag_safe(prompt, uploads, top_k=TOP_K)
                st.markdown("**Answer:**")
                st.write(ans)

def render_quizzes_tab():
    st.header("Generate MCQs")
    uploads = [u for u in st.session_state.uploads if u.get("processed")]
    if not uploads:
        st.info("Process an upload first.")
        return
    opts = {u["filename"]: idx for idx, u in enumerate(uploads)}
    sel = st.selectbox("Select document", list(opts.keys()))
    up = uploads[opts[sel]]
    qcount = st.slider("Number of MCQs", 1, 20, 5)
    if st.button("Generate MCQs"):
        with st.spinner("Generating..."):
            mcqs = generate_mcq_set_from_text(up.get("full_text",""), qcount=qcount)
            if mcqs:
                st.json(mcqs)
                try:
                    conn = get_db_connection()
                    now = int(time.time())
                    with conn:
                        for obj in mcqs:
                            qtext = obj.get("question","")
                            opts = obj.get("options",[])
                            ans = int(obj.get("answer_index", 0))
                            conn.execute("INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                                         (str(up.get("db_id") or up.get("filename")), qtext, json.dumps(opts), ans, now))
                    st.success("MCQs saved to DB")
                except Exception:
                    logger.exception("Save MCQs failed")
                    st.warning("Could not save MCQs to DB.")
            else:
                st.warning("No MCQs returned.")

def render_flashcards_tab():
    st.header("Flashcards")
    uploads = st.session_state.uploads
    if not uploads:
        st.info("Upload files first.")
        return
    sel = st.selectbox("Choose document", [u["filename"] for u in uploads], key="fc_doc_select")
    up = next((u for u in uploads if u["filename"] == sel), None)
    if st.button("Generate Flashcards"):
        with st.spinner("Generating flashcards..."):
            cards = generate_flashcards_from_text(up.get("full_text",""), n=20)
            saved = add_flashcards_to_db_safe(up, cards)
            st.success(f"Saved {saved} flashcards.")
    if st.button("Load due cards for practice"):
        st.session_state.due_cards = get_due_flashcards_safe()
        st.session_state.current_card_idx = 0
        safe_rerun()
    if st.session_state.get("due_cards"):
        render_flashcard_practice_ui()

def render_flashcard_practice_ui():
    cards = st.session_state.get("due_cards", [])
    if not cards:
        st.info("No due cards to practice now.")
        return
    idx = st.session_state.get("current_card_idx", 0)
    if idx >= len(cards):
        st.success("✨ You finished all due cards in this session.")
        st.session_state["due_cards"] = []
        st.session_state["current_card_idx"] = 0
        return
    card = cards[idx]
    st.markdown(f"##### Card {idx+1} of {len(cards)}")
    st.write(card.get("question"))
    show = st.button("Show Answer", key=f"show_{card.get('id')}")
    if show:
        st.write("**Answer:**")
        st.write(card.get("answer"))
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Again", key=f"again_{card.get('id')}"):
        update_flashcard_review_safe(card, 1)
        st.session_state["current_card_idx"] += 1
        safe_rerun()
    if col2.button("Hard", key=f"hard_{card.get('id')}"):
        update_flashcard_review_safe(card, 3)
        st.session_state["current_card_idx"] += 1
        safe_rerun()
    if col3.button("Good", key=f"good_{card.get('id')}"):
        update_flashcard_review_safe(card, 4)
        st.session_state["current_card_idx"] += 1
        safe_rerun()
    if col4.button("Easy", key=f"easy_{card.get('id')}"):
        update_flashcard_review_safe(card, 5)
        st.session_state["current_card_idx"] += 1
        safe_rerun()

def render_settings_tab():
    st.header("Settings & Exports")
    key = st.text_input("OpenRouter API Key (session only)", value=st.session_state.get("OPENROUTER_API_KEY",""), type="password")
    if key != st.session_state.get("OPENROUTER_API_KEY"):
        st.session_state["OPENROUTER_API_KEY"] = key
        st.success("API key saved in session (not persisted).")
    # Exports
    st.markdown("#### Export flashcards (Anki TSV)")
    upload_map = { (u.get("db_id") or u.get("filename")): u for u in st.session_state.uploads }
    if upload_map:
        sel_key = st.selectbox("Select upload", options=list(upload_map.keys()), format_func=lambda k: upload_map[k]["filename"])
        if st.button("Export Anki TSV"):
            conn = None
            try:
                conn = get_db_connection()
            except Exception:
                conn = None
            res = anki_export_tsv(sel_key, conn)
            if not res:
                st.warning("No flashcards for selected upload.")
            else:
                fname, b = res
                st.download_button("Download Anki TSV", data=b, file_name=fname, mime="text/tab-separated-values")
    else:
        st.info("No uploads available for export.")

# ----------------------------
# Main entrypoint
# ----------------------------
def main():
    try:
        st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass
    initialize_session_state_defaults()
    try:
        st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
    except Exception:
        pass
    render_header()
    tabs = st.tabs(["Home", "Upload", "Lessons", "Chat Q&A", "Quizzes", "Flashcards", "Settings"])
    with tabs[0]:
        render_home()
    with tabs[1]:
        render_upload_tab()
    with tabs[2]:
        render_lessons_tab()
    with tabs[3]:
        render_chat_tab()
    with tabs[4]:
        render_quizzes_tab()
    with tabs[5]:
        render_flashcards_tab()
    with tabs[6]:
        render_settings_tab()

if __name__ == "__main__":
    main()
