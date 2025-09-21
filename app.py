# SlideTutor — Improved version with EasyOCR (no Tesseract binary required)
# UI/UX refreshed: single-page (no sidebar), top tabs, modern CSS, subtle animations
# Save this file and run with: streamlit run slidetutor_improved_easyocr_ui.py
from __future__ import annotations
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
import io
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np

# Optional dependencies (checked at import time)
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS = False

try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    cosine_similarity = None  # type: ignore
    _HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from pptx import Presentation  # type: ignore
    _HAS_PPTX = True
except Exception:
    Presentation = None  # type: ignore
    _HAS_PPTX = False

try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    fitz = None  # type: ignore
    _HAS_PYMUPDF = False

# Streamlit cache helper (optional)
try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    _HAS_STREAMLIT = False

# Logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    # minimal default handler for standalone usage
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (n, d); convert 1D -> (1, d)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _safe_normalize(a: np.ndarray, axis: int = 1, eps: float = 1e-9) -> np.ndarray:
    """L2-normalize rows (or columns) safely."""
    a = np.asarray(a, dtype=np.float32, order="C")
    norm = np.linalg.norm(a, axis=axis, keepdims=True)
    norm[norm == 0] = eps
    return a / norm


class VectorIndexFallback:
    """
    A simple in-memory vector index using numpy. Returns (dists, idxs).
    dists are "distance-like": lower is better. We use -cosine_similarity so that
    smaller values represent closer matches (same contract as FAISS branch below).
    """

    def __init__(self, embeddings: Optional[np.ndarray], texts: Optional[List[str]] = None) -> None:
        texts = texts or []
        if embeddings is None or getattr(embeddings, "size", 0) == 0:
            # zero vectors: shape (0, dim) isn't known yet; leave dim as 0
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
            self._normed = None
        else:
            emb = np.asarray(embeddings, dtype=np.float32, order="C")
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            self.embeddings = emb
            # store normalized embeddings for cosine calculations
            self._normed = _safe_normalize(self.embeddings, axis=1)

        self.texts = list(texts)

    def _empty_result(self, n_queries: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        dists = np.full((n_queries, k), np.finfo(np.float32).max, dtype=np.float32)
        idxs = np.full((n_queries, k), -1, dtype=np.int64)
        return dists, idxs

    def search(self, q_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q_emb, dtype=np.float32, order="C")
        q = _ensure_2d(q)
        n_q, dim_q = q.shape

        if self.embeddings.size == 0:
            return self._empty_result(n_q, k)

        if self.embeddings.shape[1] != dim_q:
            logger.error("Query embedding dimension (%d) does not match index dimension (%d).", dim_q, self.embeddings.shape[1])
            return self._empty_result(n_q, k)

        normed_q = _safe_normalize(q, axis=1)

        # compute cosine similarities
        if _HAS_SKLEARN:
            try:
                sims = cosine_similarity(normed_q, self._normed)  # shape (n_q, n_db)
            except Exception as ex:
                logger.warning("sklearn cosine_similarity failed, falling back to numpy: %s", ex)
                sims = normed_q @ self._normed.T
        else:
            sims = normed_q @ self._normed.T

        # get top-k indices (higher sim is better). We return distances as -sim so smaller is better.
        n_db = sims.shape[1]
        k_eff = min(max(1, int(k)), n_db)
        idxs_part = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        # sort the partitioned top-k
        row_indices = np.arange(n_q)[:, None]
        idxs_sorted = idxs_part[np.arange(n_q)[:, None], np.argsort(-sims[row_indices, idxs_part], axis=1)]
        idxs = idxs_sorted

        # distances as negative similarity (so smaller is better)
        dists = -np.take_along_axis(sims, idxs, axis=1).astype(np.float32)

        # pad results if requested k > n_db
        if k_eff < k:
            pad_count = k - k_eff
            dists = np.pad(dists, ((0, 0), (0, pad_count)), constant_values=np.finfo(np.float32).max)
            idxs = np.pad(idxs, ((0, 0), (0, pad_count)), constant_values=-1)

        return dists, idxs.astype(np.int64)


class VectorIndex:
    """
    Hybrid wrapper: uses FAISS (fast) when available and falls back to numpy version.
    search(q_emb, k) -> (dists, idxs) where dists are smaller = better, idxs -1 means no result.
    """

    def __init__(self, embeddings: Optional[np.ndarray], texts: Optional[List[str]] = None) -> None:
        texts = texts or []
        self.texts = list(texts)
        if embeddings is None or getattr(embeddings, "size", 0) == 0:
            self.index = VectorIndexFallback(np.zeros((0, 0), dtype=np.float32), self.texts)
            self._use_faiss = False
            return

        emb = np.asarray(embeddings, dtype=np.float32, order="C")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self.embeddings = emb
        n, d = emb.shape

        if _HAS_FAISS and n > 0:
            try:
                # normalize to use inner-product as cosine similarity
                normed = _safe_normalize(self.embeddings, axis=1)
                # FAISS expects contiguous float32
                normed = np.ascontiguousarray(normed.astype(np.float32))
                self._faiss_index = faiss.IndexFlatIP(d)  # inner product
                self._faiss_index.add(normed)
                self._use_faiss = True
            except Exception as ex:
                logger.exception("Failed to build FAISS index; falling back to numpy. Error: %s", ex)
                self.index = VectorIndexFallback(self.embeddings, texts)
                self._use_faiss = False
        else:
            self.index = VectorIndexFallback(self.embeddings, texts)
            self._use_faiss = False

    def search(self, q_emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q_emb, dtype=np.float32, order="C")
        q = _ensure_2d(q)
        n_q, dim_q = q.shape

        if getattr(self, "_use_faiss", False):
            # ensure dims match
            if dim_q != self._faiss_index.d:
                logger.error("Query dim (%d) != FAISS index dim (%d)", dim_q, self._faiss_index.d)
                return np.full((n_q, k), np.finfo(np.float32).max, dtype=np.float32), np.full((n_q, k), -1, dtype=np.int64)

            # normalize queries
            q_normed = _safe_normalize(q, axis=1)
            # faiss returns (distances, indices) where distances are inner-product (higher is better)
            try:
                D, I = self._faiss_index.search(np.ascontiguousarray(q_normed), k)
                # convert D (similarities) to distance-like (smaller = better)
                D = D.astype(np.float32)
                dists = -D
                idxs = I.astype(np.int64)
                return dists, idxs
            except Exception as ex:
                logger.exception("FAISS search failed; falling back to numpy. Error: %s", ex)
                # fallback to numpy search
                return self.index.search(q, k)
        else:
            return self.index.search(q, k)


# ------------------------------
# Embedding helpers
# ------------------------------
if _HAS_STREAMLIT:
    # use streamlit's cache decorator if available
    def _cache_decorator(func):
        return st.cache_resource(func)
else:
    # fallback to lru_cache for process-local caching
    from functools import lru_cache
    def _cache_decorator(func):
        return lru_cache(maxsize=2)(func)


@_cache_decorator
def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a SentenceTransformer model. Uses streamlit.cache_resource when streamlit is present,
    otherwise uses functools.lru_cache for local caching.
    """
    if not _HAS_SENTENCE_TRANSFORMERS or SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required for embeddings. Install with: pip install sentence-transformers"
        )
    logger.info("Loading SentenceTransformer model '%s' ...", model_name)
    model = SentenceTransformer(model_name)
    # ensure model returns float32 numpy arrays
    return model


def embed_texts(model: "SentenceTransformer", texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts to float32 numpy embeddings. Returns shape (n_texts, dim).
    If texts is empty return shape (0, dim) if model is available, else (0, 0).
    """
    texts = texts or []
    if not texts:
        # try to determine dimension if model available
        try:
            dim = model.get_sentence_embedding_dimension()
            return np.zeros((0, int(dim)), dtype=np.float32)
        except Exception:
            return np.zeros((0, 0), dtype=np.float32)

    # SentenceTransformer.encode -> numpy array
    arr = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ------------------------------
# Parsers: pptx & pdf
# ------------------------------
def extract_from_pptx_bytes(file_bytes: bytes) -> List[Dict]:
    """
    Extract slide-level text and image bytes from a .pptx file.
    Returns list of dicts: {"index": int, "text": str, "images": List[bytes]}
    """
    if not _HAS_PPTX or Presentation is None:
        return [{"index": 0, "text": "[python-pptx not installed]", "images": []}]

    slides: List[Dict] = []
    try:
        prs = Presentation(io.BytesIO(file_bytes))
        for i, slide in enumerate(prs.slides):
            texts: List[str] = []
            images: List[bytes] = []
            for shape in slide.shapes:
                # text frames
                try:
                    if getattr(shape, "has_text_frame", False):
                        text_frame = getattr(shape, "text_frame", None)
                        if text_frame:
                            # gather paragraphs
                            txt = "\n".join([p.text.strip() for p in text_frame.paragraphs if p.text and p.text.strip()])
                            if txt:
                                texts.append(txt)
                except Exception:
                    # robust: skip problematic shape
                    logger.debug("Skipping shape text extraction on slide %d due to exception.", i, exc_info=True)

                # images: picture shapes have shape.image
                try:
                    # For picture shapes (pptx.Pictures), shape.image exists and has .blob
                    img = getattr(shape, "image", None)
                    if img is not None:
                        image_bytes = getattr(img, "blob", None)
                        if image_bytes:
                            images.append(image_bytes)
                except Exception:
                    logger.debug("Skipping shape image extraction on slide %d due to exception.", i, exc_info=True)

            slides.append({"index": i, "text": "\n".join(texts).strip(), "images": images})
        return slides
    except Exception as e:
        logger.exception("pptx parse error: %s", e)
        return [{"index": 0, "text": f"[pptx parse error] {e}", "images": []}]


def extract_from_pdf_bytes(file_bytes: bytes) -> List[Dict]:
    """
    Extract per-page text and images from a PDF (PyMuPDF). If a page has no selectable text
    and no embedded images, render the page as a PNG and return that image as fallback.
    """
    if not _HAS_PYMUPDF or fitz is None:
        return [{"index": 0, "text": "[pymupdf not installed, can't parse PDF]", "images": []}]

    slides: List[Dict] = []
    doc = None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i in range(doc.page_count):
            page = doc.load_page(i)
            try:
                text = page.get_text("text") or ""
                text = text.strip()
            except Exception:
                logger.debug("Page %d: text extraction failed; treating as empty.", i, exc_info=True)
                text = ""

            images: List[bytes] = []
            try:
                for img in page.get_images(full=True):
                    xref = img[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image.get("image")
                        if image_bytes:
                            images.append(image_bytes)
                    except Exception:
                        logger.debug("Failed to extract image xref %s on page %d", xref, i, exc_info=True)
            except Exception:
                logger.debug("get_images failed on page %d", i, exc_info=True)

            # If page has no selectable text and no embedded images, render to image
            if (not text) and (not images):
                try:
                    mat = fitz.Matrix(2.0, 2.0)  # raster scale
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    try:
                        img_bytes = pix.tobytes(output="png")
                    except TypeError:
                        # older pymupdf versions may not support "output" kw
                        img_bytes = pix.tobytes()
                    images.append(img_bytes)
                except Exception as e:
                    logger.debug("PDF page rendering failed for page %d: %s", i, e, exc_info=True)

            slides.append({"index": i, "text": text, "images": images})
        return slides
    except Exception as e:
        logger.exception("pdf parse error: %s", e)
        return [{"index": 0, "text": f"[pdf parse error] {e}", "images": []}]
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


# ------------------------------
# Example usage (quick smoke test)
# ------------------------------
if __name__ == "__main__":
    # simple smoke tests that don't require optional dependencies
    vecs = np.random.randn(10, 64).astype(np.float32)
    idx = VectorIndex(vecs, [f"doc_{i}" for i in range(10)])
    q = np.random.randn(2, 64).astype(np.float32)
    dists, ids = idx.search(q, k=3)
    logger.info("Search results shapes: dists=%s ids=%s", dists.shape, ids.shape)


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
# --------- Streamlit UI (replacement) ---------
import os
import io
import json
import base64
import typing
from typing import List, Dict, Optional
import numpy as np
import streamlit as st

# Ensure these names exist in your module (they were in earlier code blocks):
# - APP_TITLE, APP_SUBTITLE (strings)
# - load_sentence_transformer(model_name) -> SentenceTransformer-like
# - embed_texts(model, texts) -> np.ndarray
# - VectorIndex class
# - extract_from_pdf_bytes(file_bytes) -> List[Dict]
# - extract_from_pptx_bytes(file_bytes) -> List[Dict]
#
# If any are missing, the UI will show informative errors where those actions are required.

st.set_page_config(page_title=globals().get("APP_TITLE", "SlideTutor"), layout="wide")

# --- ensure API key in session state (non-blocking) ---
DEFAULT_OPENROUTER_KEY = globals().get("DEFAULT_OPENROUTER_KEY", "")
if "OPENROUTER_API_KEY" not in st.session_state:
    st.session_state["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", DEFAULT_OPENROUTER_KEY)

# --- insert theme CSS + decorative blobs + wrapper open tag ---
st.markdown(
    """
    <style>
    /* ---------- Paste your advanced CSS here (kept concise in this block) ---------- */
    /* For brevity I'm reusing your theme tokens & many rules unchanged, plus blob styles + small-muted */
    :root{ --bg:#071025; --bg-2:#06101A; --text:#E6F0FA; --muted:#9AA6B2; --accent:#2AB7A9; --accent-2:#4D7CFE; --radius:12px; --gap-md:20px; }
    .app-theme, .app-theme * { box-sizing: border-box; font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; color:var(--text); }
    .app-theme .container{ padding:20px; min-height:100vh; }
    .app-theme .topbar{ display:flex; gap:12px; align-items:center; justify-content:space-between; padding:12px 18px; }
    .app-theme .logo{ width:48px;height:48px;border-radius:12px; display:flex;align-items:center;justify-content:center;font-weight:700;color:white;background:linear-gradient(135deg,var(--accent-2),var(--accent)); box-shadow:0 12px 30px rgba(0,0,0,0.45); }
    .app-theme .title{ font-size:1.2rem; font-weight:700; margin:0 }
    .app-theme .subtitle{ color:var(--muted); font-size:0.88rem; margin-top:2px }
    .app-theme .tabs{ display:flex; gap:8px; align-items:center; }
    .app-theme .tab{ padding:8px 12px; border-radius:10px; font-weight:600; cursor:pointer; border:1px solid transparent; color:var(--muted); background:transparent; }
    .app-theme .tab.active{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); color:var(--text); box-shadow:0 10px 28px rgba(0,0,0,0.45); }
    .app-theme .card{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:18px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 28px rgba(2,6,12,0.5); }
    .app-theme .small-muted{ color:var(--muted); font-size:13px; }
    .app-theme .blob{ position: fixed; pointer-events:none; filter: blur(36px); z-index:0; opacity:0.55; }
    .app-theme .blob.one{ width:420px;height:420px;border-radius:50%; left:-140px; top:-120px; background: radial-gradient(circle at 30% 30%, rgba(77,124,254,0.18), transparent 30%); }
    .app-theme .blob.two{ width:320px;height:320px;border-radius:50%; right:-80px; bottom:-80px; background: radial-gradient(circle at 70% 70%, rgba(42,183,169,0.12), transparent 30%); }
    .app-theme .small-actions{ display:flex; gap:8px; }
    .app-theme .slide-img{ max-width:100%; border-radius:8px; box-shadow: 0 8px 20px rgba(0,0,0,0.45); }
    .app-theme .muted{ color:var(--muted) }
    </style>
    <div class="app-theme">
    """,
    unsafe_allow_html=True,
)

# decorative blobs
st.markdown("<div class='blob one'></div><div class='blob two'></div>", unsafe_allow_html=True)

# --- topbar / brand + tabs (keyboard-friendly) ---
with st.container():
    top_cols = st.columns([0.4, 3, 1.2])
    with top_cols[0]:
        st.markdown("<div class='logo'>ST</div>", unsafe_allow_html=True)
    with top_cols[1]:
        app_title = globals().get("APP_TITLE", "SlideTutor")
        app_sub = globals().get("APP_SUBTITLE", "AI slides → lessons • quizzes • flashcards")
        st.markdown(f"<div class='title'>{app_title}</div><div class='subtitle'>{app_sub}</div>", unsafe_allow_html=True)
    with top_cols[2]:
        st.markdown("<div style='text-align:right'><small class='small-muted'>Student edition • Improved</small></div>", unsafe_allow_html=True)

TAB_NAMES = ["Home", "Upload & Process", "Lessons", "Chat Q&A", "Quizzes", "Flashcards", "Export", "Progress", "Settings"]
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Upload & Process"

# Render top tabs as accessible buttons (styled by CSS above)
tab_cols = st.columns(len(TAB_NAMES))
for i, name in enumerate(TAB_NAMES):
    pressed = tab_cols[i].button(name, key=f"tab_btn_{name}")
    # set active on press
    if pressed:
        st.session_state["active_tab"] = name

active_tab = st.session_state["active_tab"]

# ---------- Upload state initialization ----------
if "uploads" not in st.session_state:
    # uploads : list of dicts with keys:
    # {id, filename, bytes, kind, slides, chunks, embeddings (optional), index_built(bool), status_msg}
    st.session_state["uploads"] = []

# --- helpers: chunker, add_upload, build_index_on_upload, serialize for download
def _simple_chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks of approx chunk_size characters (robust, whitespace-aware)."""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            chunks.append(p)
            continue
        start = 0
        while start < len(p):
            end = min(len(p), start + chunk_size)
            chunk = p[start:end].strip()
            chunks.append(chunk)
            if end == len(p):
                break
            start = max(0, end - overlap)
    return chunks

def _make_upload_id(filename: str) -> str:
    import hashlib, time
    h = hashlib.sha1((filename + str(time.time())).encode("utf-8")).hexdigest()[:10]
    return f"{h}_{filename}"

def add_upload_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Dict:
    """Process an uploaded file into an upload dict: extract slides and create chunks (no embeddings yet)."""
    try:
        fb = uploaded_file.read()
    except Exception as e:
        st.error(f"Failed to read uploaded file '{uploaded_file.name}': {e}")
        return {}
    filename = uploaded_file.name
    ext = filename.lower().split(".")[-1]
    kind = "pdf" if ext == "pdf" else "pptx" if ext in ("pptx", "ppt") else "unknown"
    slides = []
    try:
        if kind == "pptx":
            slides = extract_from_pptx_bytes(fb)
        elif kind == "pdf":
            slides = extract_from_pdf_bytes(fb)
        else:
            # Try PDF first, then pptx as a fallback
            try:
                slides = extract_from_pdf_bytes(fb)
            except Exception:
                try:
                    slides = extract_from_pptx_bytes(fb)
                except Exception:
                    slides = [{"index": 0, "text": "[unsupported format]", "images": []}]
    except Exception as e:
        slides = [{"index": 0, "text": f"[extraction failed] {e}", "images": []}]

    # create chunks across slides, keep mapping to slide index
    chunks: List[str] = []
    for s in slides:
        text = s.get("text", "") or ""
        per_slide_chunks = _simple_chunk_text(text)
        # prefix chunk with slide marker for traceability (optional)
        for ci, c in enumerate(per_slide_chunks):
            chunks.append(f"[slide {s.get('index',0)}] {c}")

    upload = {
        "id": _make_upload_id(filename),
        "filename": filename,
        "bytes": fb,
        "kind": kind,
        "slides": slides,
        "chunks": chunks,
        "embeddings": None,
        "index": None,
        "index_built": False,
        "status_msg": "Uploaded",
    }
    st.session_state["uploads"].append(upload)
    return upload

def build_index_for_upload(upload: Dict, model_name: Optional[str] = None) -> Dict:
    """On-demand build / rebuild of embeddings and VectorIndex. Safe, with progress UI and exceptions captured."""
    if upload is None:
        raise ValueError("upload is None")

    try:
        model_name = model_name or globals().get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        # lazy-load embedder
        try:
            model = load_sentence_transformer(model_name)
        except Exception as e:
            upload["status_msg"] = f"Embedding model load failed: {e}"
            st.error(upload["status_msg"])
            return upload

        chunks = upload.get("chunks", []) or []
        if not chunks:
            upload["embeddings"] = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
            upload["index"] = VectorIndex(upload["embeddings"], [])
            upload["index_built"] = True
            upload["status_msg"] = "No chunks to embed (empty file)."
            st.success(upload["status_msg"])
            return upload

        # show spinner and progress
        total = len(chunks)
        with st.spinner("Computing embeddings..."):
            # embed_texts should return N x D float32 array
            arr = embed_texts(model, chunks)
            if arr is None:
                raise RuntimeError("embed_texts returned None")
            arr = np.asarray(arr, dtype=np.float32)
            upload["embeddings"] = arr
            upload["index"] = VectorIndex(arr, chunks)
            upload["index_built"] = True
            upload["status_msg"] = f"Index built ({arr.shape[0]} vectors, dim={arr.shape[1] if arr.ndim>1 else 'unknown'})"
            st.success(upload["status_msg"])
            return upload
    except Exception as ex:
        upload["status_msg"] = f"Index build failed: {ex}"
        st.error(upload["status_msg"])
        return upload

def download_upload_json(upload: Dict) -> None:
    """Render a download link for the upload metadata (slides & chunks)."""
    safe = {
        "filename": upload.get("filename"),
        "kind": upload.get("kind"),
        "slides": [{"index": s.get("index"), "text": s.get("text"), "images": len(s.get("images", []))} for s in upload.get("slides", [])],
        "chunks_count": len(upload.get("chunks", [])),
    }
    payload = json.dumps(safe, indent=2)
    b64 = base64.b64encode(payload.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{upload.get("filename")}_meta.json">Download metadata (JSON)</a>'
    st.markdown(href, unsafe_allow_html=True)

# ---------- Home tab ----------
if active_tab == "Home":
    st.markdown("<div class='card' style='padding:18px'>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"<h2 style='margin:0'>Welcome to <span style='color:#a79bff'>{app_title}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Quickly upload slide decks (PPTX/PDF), build searchable semantic indices, and auto-generate lessons, quizzes and flashcards.</div>", unsafe_allow_html=True)
        st.markdown("<br/>")
        st.markdown("<div class='small-muted'>Highlights:</div>", unsafe_allow_html=True)
        st.markdown("<ul class='small-muted'><li>Scanned PDFs supported (raster fallback)</li><li>On-demand embedding / search (FAISS if available)</li><li>Export metadata & previews</li></ul>", unsafe_allow_html=True)
        st.markdown("<div class='small-actions'>", unsafe_allow_html=True)
        if st.button("Upload & Process"):
            st.session_state["active_tab"] = "Upload & Process"
        if st.button("Practice Flashcards"):
            st.session_state["active_tab"] = "Flashcards"
        if st.button("Generate Lesson"):
            st.session_state["active_tab"] = "Lessons"
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card' style='padding:12px;text-align:center'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin:6px 0'>Usage tip</h4>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Start by uploading your .pptx or .pdf in the 'Upload & Process' tab. Build the embedding index to enable Search, Lessons and Quizzes.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Upload & Process tab ----------
if active_tab == "Upload & Process":
    st.markdown("<div class='card' style='margin-bottom:12px'>", unsafe_allow_html=True)
    st.header("Upload & Process")
    st.markdown("<div class='small-muted'>Upload one or more PPTX / PDF files. Each upload is processed into slide-level text + images and chunked into semantically meaningful pieces.</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose PPTX/PDF files", accept_multiple_files=True, type=["pdf", "pptx", "ppt"])
    if uploaded:
        for f in uploaded:
            # avoid duplicate uploads by filename heuristic
            known_names = {u.get("filename") for u in st.session_state.get("uploads", [])}
            fname = getattr(f, "name", None) or "<uploaded_file>"
            if fname in known_names:
                st.warning(f"File '{fname}' already uploaded - skipping duplicate.")
                continue

            with st.spinner(f"Processing {fname} ..."):
                up = add_upload_file(f)
                # add_upload_file should return a dict (upload info). Be defensive.
                if up and isinstance(up, dict):
                    uploaded_name = up.get("filename") or fname
                    st.success(f"Uploaded: {uploaded_name}")
                elif up:  # maybe a truthy non-dict (unexpected), show safe message
                    st.success(f"Uploaded: {fname}")
                else:
                    st.error(f"Failed to process upload: {fname}")

    st.markdown("</div>", unsafe_allow_html=True)

    # List existing uploads with controls
    st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
    uploads = st.session_state.get("uploads", [])
    if not uploads:
        st.info("No uploads yet. Use the uploader above to add a PPTX or PDF.")
    else:
        for idx, up in enumerate(list(uploads)):
            with st.expander(f"{up['filename']} — {up.get('status_msg','')}"):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**File:** {up['filename']} • **Type:** {up['kind']}  ")
                    st.markdown(f"<div class='small-muted'>Chunks: {len(up.get('chunks',[]))} • Index built: {up.get('index_built')}</div>", unsafe_allow_html=True)
                    # quick preview: first 2 slides, first image if present
                    slides = up.get("slides", [])[:4]
                    for s in slides:
                        st.markdown(f"**Slide {s.get('index', '?')}**")
                        txt = s.get("text", "").strip()
                        if txt:
                            st.markdown(f"<div class='small-muted'>{txt[:800]}{'...' if len(txt)>800 else ''}</div>", unsafe_allow_html=True)
                        images = s.get("images", []) or []
                        if images:
                            try:
                                st.image(images[0], caption=f"Slide {s.get('index')} preview", use_column_width=True)
                            except Exception:
                                # safe fallback: write placeholder
                                st.markdown("<div class='small-muted'>[image preview not renderable]</div>", unsafe_allow_html=True)
                        st.markdown("---")
                with cols[1]:
                    # action buttons: build index, download metadata, delete
                    if st.button("Build / Rebuild Index", key=f"build_idx_{up['id']}"):
                        # mutate session state item in place safely
                        st.session_state["uploads"][idx] = build_index_for_upload(up)
                    if st.button("Download metadata", key=f"dlmeta_{up['id']}"):
                        download_upload_json(up)
                    if st.button("Delete", key=f"del_{up['id']}"):
                        st.session_state["uploads"].pop(idx)
                        st.experimental_rerun()  # refresh UI after deletion
                    st.markdown("<br/>")
                    st.markdown(f"<div class='small-muted'>{up.get('status_msg','')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Lessons tab (placeholder / graceful) ----------
if active_tab == "Lessons":
    st.header("Lessons (Generate from uploaded files)")
    uploads = st.session_state.get("uploads", [])
    if not uploads:
        st.info("Upload a file first (Upload & Process tab).")
    else:
        # choose upload
        options = [f"{u['filename']} ({'indexed' if u.get('index_built') else 'not indexed'})" for u in uploads]
        sel = st.selectbox("Select upload to generate lessons from", options=options)
        sel_idx = options.index(sel)
        selected = uploads[sel_idx]
        st.markdown("<div class='small-muted'>Choose a generation mode and press Generate.</div>", unsafe_allow_html=True)
        mode = st.radio("Generation Mode", ["Concise lesson", "Detailed lesson with examples", "Key-points summary"], index=0)
        if st.button("Generate Lesson"):
            # graceful: if no index/AI/keys available, show error with helpful instructions
            if not selected.get("index_built"):
                st.error("Index not built for this upload. Please go to 'Upload & Process' and press 'Build / Rebuild Index'.")
            elif not st.session_state.get("OPENROUTER_API_KEY"):
                st.error("No OPENROUTER_API_KEY set. Set it in Settings (or as environment variable).")
            else:
                # attempt to call a user-provided generate_lesson function if available
                if "generate_lesson_from_upload" in globals():
                    try:
                        with st.spinner("Generating lesson..."):
                            lesson_text = globals()["generate_lesson_from_upload"](selected, mode=mode)
                            st.success("Lesson generated")
                            st.markdown(lesson_text)
                    except Exception as e:
                        st.error(f"Lesson generation failed: {e}")
                else:
                    st.warning("No lesson-generation pipeline found in this runtime. Implement `generate_lesson_from_upload(upload, mode)` to enable this feature.")

# ---------- Chat Q&A, Quizzes, Flashcards, Export, Progress, Settings (basic placeholders) ----------
if active_tab == "Chat Q&A":
    st.header("Chat Q&A")
    st.markdown("Ask a question based on your uploaded slides (requires index & API key).")
    q = st.text_input("Enter question")
    if st.button("Ask"):
        uploads = [u for u in st.session_state.get("uploads", []) if u.get("index_built")]
        if not uploads:
            st.error("No indexed uploads available. Build an index first.")
        else:
            # search top doc chunks across all uploads (simple example)
            if "answer_question" in globals():
                try:
                    res = globals()["answer_question"](q, uploads)
                    st.markdown(res)
                except Exception as e:
                    st.error(f"Q&A failed: {e}")
            else:
                st.info("Q&A pipeline not available. Implement answer_question(query, indexed_uploads).")

if active_tab == "Quizzes":
    st.header("Auto-generated Quizzes")
    st.markdown("Generate multiple-choice quizzes from selected upload (placeholder).")
    if st.button("Generate Quiz (placeholder)"):
        st.info("Implement quiz generation or call your existing function (e.g., generate_quiz_from_upload).")

if active_tab == "Flashcards":
    st.header("Flashcards / Spaced Repetition")
    st.markdown("Practice flashcards generated from content. This is a placeholder UI; wire it to your SRS backend.")
    if st.button("Practice now"):
        st.info("Practice logic not wired. Implement or connect to your SRS (SM-2) implementation.")

if active_tab == "Export":
    st.header("Export")
    st.markdown("Export all metadata and optionally embeddings (if index built).")
    if st.button("Export all metadata JSON"):
        # build combined metadata and provide download link
        exports = []
        for u in st.session_state.get("uploads", []):
            exports.append({
                "filename": u.get("filename"),
                "kind": u.get("kind"),
                "chunks_count": len(u.get("chunks", [])),
                "indexed": bool(u.get("index_built")),
            })
        payload = json.dumps(exports, indent=2)
        b64 = base64.b64encode(payload.encode()).decode()
        st.markdown(f'<a href="data:application/json;base64,{b64}" download="slide_tutor_export.json">Download export</a>', unsafe_allow_html=True)

if active_tab == "Progress":
    st.header("Progress")
    st.markdown("Track your learning progress here (placeholder).")

if active_tab == "Settings":
    st.header("Settings")
    st.text_input("OpenRouter API Key (session only)", value=st.session_state.get("OPENROUTER_API_KEY", ""), key="OPENROUTER_API_KEY_INPUT", type="password")
    if st.button("Save API key to session"):
        st.session_state["OPENROUTER_API_KEY"] = st.session_state.get("OPENROUTER_API_KEY_INPUT")
        st.success("API key saved to session.")
    st.markdown("<div class='small-muted'>Tip: Prefer storing API keys as environment variables in production.</div>", unsafe_allow_html=True)

# --- close wrapper ---
st.markdown("</div>", unsafe_allow_html=True)
# --------- end replacement ---------

# ------------------------------
# ------------------------------
# Upload & Process + App Tabs (robust replacement)
# ------------------------------
import io
import json
import time
import traceback
import base64
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st

# Use a fallback logger function if `log()` isn't defined in the user's module
try:
    log  # type: ignore
except NameError:
    import logging
    _logger = logging.getLogger(__name__)
    if not _logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        _logger.addHandler(h)
    _logger.setLevel(logging.INFO)

    def log(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        _logger.info(msg)

# Utility: safe access to DB cursor
def _get_db_cursor():
    """Return a cursor if _db_conn is available, else None."""
    try:
        cur = _db_conn.cursor()
        return cur
    except Exception as e:
        log("DB connection not available:", e)
        return None

# Safe wrapper for optional helpers
def _safe_call(func_name: str, *args, default=None, **kwargs):
    """Call a global function by name if present; otherwise return default."""
    f = globals().get(func_name)
    if callable(f):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            log(f"Optional helper '{func_name}' failed:", e)
            return default
    else:
        log(f"Optional helper '{func_name}' not found; skipping.")
        return default

# Safe chunker wrapper: prefer user's chunk_text if present; fallback to simple splitter
def _simple_chunk_text(text: str, chunk_size: int = 800, overlap: int = 80) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            chunks.append(p)
            continue
        start = 0
        while start < len(p):
            end = min(len(p), start + chunk_size)
            chunk = p[start:end].strip()
            chunks.append(chunk)
            if end == len(p):
                break
            start = max(0, end - overlap)
    return chunks

def _chunk_text_safe(text: str) -> List[str]:
    if "chunk_text" in globals() and callable(globals()["chunk_text"]):
        try:
            return globals()["chunk_text"](text)
        except Exception as e:
            log("User chunk_text failed; falling back to simple chunker:", e)
            return _simple_chunk_text(text)
    else:
        return _simple_chunk_text(text)

# Safe OCR wrapper
def _ocr_images_safe(img_bytes_list: List[bytes]) -> List[str]:
    if "ocr_image_bytes_list" in globals() and callable(globals()["ocr_image_bytes_list"]):
        try:
            return globals()["ocr_image_bytes_list"](img_bytes_list)
        except Exception as e:
            log("ocr_image_bytes_list failed:", e)
            return []
    else:
        # easyocr not available or function not defined
        return []

# Safe index builder wrapper (calls your build_index_for_upload if present)
def _build_index_safe(upload_obj: Dict) -> Dict:
    if "build_index_for_upload" in globals() and callable(globals()["build_index_for_upload"]):
        try:
            return globals()["build_index_for_upload"](upload_obj)
        except Exception as e:
            log("build_index_for_upload failed:", e)
            return upload_obj
    else:
        # fallback: create empty embeddings/index fields to maintain contract
        try:
            dim = 1
            upload_obj["embeddings"] = np.zeros((0, dim), dtype=np.float32)
            upload_obj["index"] = VectorIndex(upload_obj["embeddings"], upload_obj.get("chunks", []))
            upload_obj["index_built"] = False
        except Exception as ex:
            log("fallback index creation failed:", ex)
        return upload_obj

# Helper: get db upload id (if you store uploads in DB)
def upload_db_id(upload: Dict) -> Optional[int]:
    if upload is None:
        return None
    if upload.get("db_id") is not None:
        return upload.get("db_id")
    # try to find by filename/metadata in DB if _db_conn present
    cur = _get_db_cursor()
    if cur is None:
        return None
    try:
        cur.execute("SELECT id FROM uploads WHERE filename = ? ORDER BY uploaded_at DESC LIMIT 1", (upload.get("filename"),))
        row = cur.fetchone()
        if row:
            return int(row[0])
    except Exception as e:
        log("upload_db_id lookup failed:", e)
    return None

# --------------------------------------------------------------------------------
# The actual UI tabs (keeps exactly the features you had, but hardened)
# --------------------------------------------------------------------------------

# Upload & Process
if active_tab == "Upload & Process":

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
            file_size = int(getattr(uploaded_file, "size", len(raw_bytes)))
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
            slides: List[Dict[str, Any]] = []
            lower = fname.lower()
            if lower.endswith(".pptx") or lower.endswith(".ppt"):
                if not globals().get("_HAS_PPTX", False):
                    st.error("python-pptx not installed; cannot parse PPTX. Install python-pptx to enable this feature.")
                    slides = [{"index": 0, "text": "[python-pptx not installed]", "images": []}]
                else:
                    try:
                        slides = extract_from_pptx_bytes(raw_bytes)
                    except Exception as e:
                        log("pptx extraction error:", e)
                        slides = [{"index": 0, "text": f"[pptx parse error] {e}", "images": []}]
            elif lower.endswith(".pdf"):
                if not globals().get("_HAS_PYMUPDF", False):
                    st.error("pymupdf not installed; cannot parse PDF. Install pymupdf to enable full PDF parsing.")
                    slides = [{"index": 0, "text": "[pymupdf not installed, can't parse PDF]", "images": []}]
                else:
                    try:
                        slides = extract_from_pdf_bytes(raw_bytes)
                    except Exception as e:
                        log("pdf extraction error:", e)
                        slides = [{"index": 0, "text": f"[pdf parse error] {e}", "images": []}]
            else:
                # best-effort try both extractors
                try:
                    slides = extract_from_pdf_bytes(raw_bytes)
                except Exception:
                    try:
                        slides = extract_from_pptx_bytes(raw_bytes)
                    except Exception:
                        slides = [{"index": 0, "text": "[Unsupported file type or extraction failed]", "images": []}]

            # OCR any slide images (only if easyocr available; keep resilient)
            with st.spinner("Running OCR on slide images (if any)..."):
                for si, s in enumerate(slides):
                    imgs = s.get("images") or []
                    if imgs:
                        try:
                            ocr_texts = _ocr_images_safe(imgs)
                            appended = "\n".join([t for t in ocr_texts if t])
                            if appended:
                                s["text"] = (s.get("text", "") + "\n\n" + appended).strip()
                        except Exception as e:
                            log(f"OCR failure on slide {si}:", e)

            # chunk text into manageable pieces
            chunks: List[str] = []
            mapping: List[Dict[str, Any]] = []
            for s in slides:
                try:
                    parts = _chunk_text_safe(s.get("text", "") or "")
                except Exception as e:
                    log("chunk_text failed for slide", s.get("index"), e)
                    parts = [s.get("text", "") or ""]
                for p in parts:
                    chunks.append(p)
                    mapping.append({"slide": int(s.get("index", 0)), "text": p})

            # build upload object; use millisecond timestamp as id (keeps compatibility)
            upload_obj: Dict[str, Any] = {
                "id": int(time.time() * 1000),
                "filename": fname,
                "uploaded_at": int(time.time()),
                "slides": slides,
                "chunks": chunks,
                "mapping": mapping,
                "embeddings": None,
                "index": None,
                "index_built": False,
                "status_msg": "Uploaded",
            }

            # attempt to build embeddings/index but fail gracefully if models missing
            with st.spinner("Creating embeddings and index (this may take a while)..."):
                try:
                    upload_obj = _build_index_safe(upload_obj)
                    # If build_index_for_upload sets status messages, keep them; otherwise set default
                    if upload_obj.get("index_built") is None:
                        upload_obj["index_built"] = bool(upload_obj.get("embeddings") is not None and getattr(upload_obj.get("embeddings"), "size", 0) > 0)
                    upload_obj["status_msg"] = upload_obj.get("status_msg", "Index build attempted")
                except Exception as e:
                    log("Index build failed; continuing without embeddings:", e)
                    upload_obj["embeddings"] = np.zeros((0, 1), dtype=np.float32)
                    upload_obj["index"] = VectorIndex(upload_obj["embeddings"], upload_obj.get("chunks", []))
                    upload_obj["index_built"] = False
                    upload_obj["status_msg"] = "Index build failed (fallback index created)"

            # persist upload metadata in DB and capture DB id (defensive)
            try:
                cur = _get_db_cursor()
                if cur:
                    meta = {"n_slides": len(slides), "n_chunks": len(chunks), "file_size": file_size}
                    cur.execute(
                        "INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)",
                        (fname, upload_obj["uploaded_at"], json.dumps(meta))
                    )
                    _db_conn.commit()
                    db_id = getattr(cur, "lastrowid", None)
                    upload_obj["db_id"] = int(db_id) if db_id is not None else None
                else:
                    upload_obj["db_id"] = None
            except Exception as e:
                log("Could not save upload metadata to DB:", e)
                upload_obj["db_id"] = None

            # store in session (in-memory) but avoid duplicates (same filename & slide count)
            try:
                if "uploads" not in st.session_state:
                    st.session_state["uploads"] = []
                # remove previous identical upload (filename + slide count) to avoid clutter
                existing = None
                for u in st.session_state["uploads"]:
                    if u.get("filename") == upload_obj.get("filename") and len(u.get("slides", [])) == len(upload_obj.get("slides", [])):
                        existing = u
                        break
                if existing:
                    st.session_state["uploads"].remove(existing)
                st.session_state["uploads"].append(upload_obj)
            except Exception as e:
                log("Could not append upload to session; attempting append anyway:", e)
                try:
                    st.session_state["uploads"].append(upload_obj)
                except Exception as ex2:
                    log("Final append failed:", ex2)

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
                        except Exception:
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
                                t_idx = int(sthumb.get("index", i + ci))
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

        except Exception as e:
            # top-level error handling so UI doesn't crash
            st.error(f"Unexpected error during upload processing: {e}")
            log("Unhandled upload processing error:", e, traceback.format_exc())

# ------------------------------
# Lessons
# ------------------------------
elif active_tab == "Lessons":

    st.header("Generate Multi-level Lessons")
    if not st.session_state.get("uploads"):
        st.info("No uploads yet. Go to Upload & Process.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state.get("uploads", [])}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_lessons")
        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if upload is None:
            st.warning("Selected upload not found.")
        else:
            slides = upload.get("slides", [])
            max_idx = max([int(s["index"]) for s in slides]) if slides else 0
            slide_idx = st.number_input("Slide/Page index", min_value=0, max_value=max_idx, value=0)
            slide_text = next((s.get("text", "") for s in slides if int(s.get("index", 0)) == int(slide_idx)), "")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader(f"Slide/Page {slide_idx} — preview")
            st.write(slide_text if len(slide_text) < 4000 else slide_text[:4000] + "...")
            st.markdown("</div>", unsafe_allow_html=True)

            deep = st.checkbox("Produce a deeply detailed lesson (longer, step-by-step)", value=False)
            auto_create = st.checkbox("Also auto-create quiz + flashcards and save to DB", value=True)

            if st.button("Generate Lesson (Beginner→Advanced)"):
                idx = upload.get("index")
                related = ""
                snippets: List[str] = []
                if idx and upload.get("chunks"):
                    try:
                        # try to search using the upload's index
                        if "load_sentence_transformer" in globals() and callable(globals()["load_sentence_transformer"]):
                            model = load_sentence_transformer()
                            q_emb = embed_texts(model, [slide_text])
                            D, I = idx.search(q_emb, globals().get("TOP_K", 5))
                            for j in I[0]:
                                if isinstance(j, int) and 0 <= j < len(upload["chunks"]):
                                    slide_num = upload["mapping"][j]["slide"] if j < len(upload["mapping"]) else None
                                    prefix = f"[Slide {slide_num}] " if slide_num is not None else ""
                                    snippets.append(prefix + upload["chunks"][j])
                    except Exception as e:
                        log("Index search failed:", e)
                related = "\n\n".join(snippets)
                with st.spinner("Generating lesson via OpenRouter..."):
                    try:
                        if deep:
                            lesson = _safe_call("generate_deep_lesson", slide_text, related, default=None)
                        else:
                            lesson = _safe_call("generate_multilevel_lesson", slide_text, related, default=None)
                        if lesson is None:
                            st.warning("Lesson-generation function not available in this environment.")
                            lesson = "[Lesson generation not available]"
                    except Exception as e:
                        lesson = f"[Lesson generation failed: {e}]"
                        log("Lesson generation error:", e)
                st.subheader("Generated Lesson")
                st.markdown(lesson)

                if auto_create:
                    st.info("Attempting to auto-generate MCQs and flashcards from lesson and saving to DB.")
                    mcqs = _safe_call("generate_mcq_set_from_text", lesson, qcount=8, default=[])
                    fcards = _safe_call("generate_flashcards_from_text", lesson, n=12, default=[])
                    cur = _get_db_cursor()
                    if cur is None:
                        st.warning("Database not available; generated artifacts will not be persisted.")
                    else:
                        try:
                            db_uid = upload_db_id(upload)
                            for q in mcqs or []:
                                cur.execute("INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                                            (db_uid, q.get("question", ""), json.dumps(q.get("options", [])), int(q.get("answer_index", 0)), int(time.time())))
                            inserted = 0
                            for card in fcards or []:
                                qtext = card.get("q") or card.get("question") or ""
                                atext = card.get("a") or card.get("answer") or ""
                                if qtext and atext:
                                    cur.execute("INSERT INTO flashcards (upload_id, question, answer, easiness, interval, repetitions, next_review) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                                (db_uid, qtext, atext, 2.5, 1, 0, int(time.time())))
                                    inserted += 1
                            _db_conn.commit()
                            st.success(f"Saved {len(mcqs or [])} MCQs and {inserted} flashcards to DB (if any).")
                        except Exception as e:
                            st.warning(f"Could not save generated artifacts: {e}")

                # Optional TTS export with gTTS if available
                if globals().get("_HAS_GTTS") and st.button("Export lesson as MP3 (TTS)"):
                    try:
                        fname, mp3bytes = _safe_call("text_to_speech_download", lesson, default=(None, None))
                        if fname and mp3bytes:
                            st.download_button("Download lesson audio", mp3bytes, file_name=fname, mime="audio/mpeg")
                        else:
                            st.warning("TTS helper not available or returned no data.")
                    except Exception as e:
                        st.error(f"TTS failed: {e}")

# ------------------------------
# Chat Q&A
# ------------------------------
elif active_tab == "Chat Q&A":

    st.header("Ask questions about your upload (Retrieval + LLM)")
    if not st.session_state.get("uploads"):
        st.info("No uploads yet. Upload files first.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_chat")

        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if upload:
            question = st.text_area("Ask a question about the slides/pages")
            if st.button("Get Answer") and question.strip():
                # embedding + search
                top_ctx = []
                try:
                    model = None
                    if "load_sentence_transformer" in globals() and callable(globals()["load_sentence_transformer"]):
                        model = load_sentence_transformer()
                    if model is not None:
                        q_emb = embed_texts(model, [question])
                    else:
                        q_emb = None
                    idx = upload.get("index")
                    if idx and q_emb is not None:
                        D, I = idx.search(q_emb, globals().get("TOP_K", 5))
                        for j in I[0]:
                            if isinstance(j, int) and 0 <= j < len(upload["chunks"]):
                                slide_num = upload["mapping"][j]["slide"] if j < len(upload["mapping"]) else None
                                prefix = f"[Slide {slide_num}] " if slide_num is not None else ""
                                top_ctx.append(prefix + upload["chunks"][j])
                except Exception as e:
                    log("Q&A indexing step failed:", e)

                top_ctx_text = "\n\n".join(top_ctx)
                system = "You are a helpful tutor. Use the provided context to answer concisely; cite slide/page indices if possible."
                prompt = f"CONTEXT:\n{top_ctx_text}\n\nQUESTION:\n{question}\n\nAnswer concisely and provide one short example or analogy."
                with st.spinner("Querying OpenRouter..."):
                    try:
                        ans = _safe_call("call_openrouter_chat", system, prompt, max_tokens=450, default="[OpenRouter call not available]")
                        st.subheader("Answer")
                        st.write(ans)
                    except Exception as e:
                        st.error(f"OpenRouter call failed: {e}")

                if top_ctx:
                    st.markdown("---")
                    st.markdown("**Context used (excerpt):**")
                    for j, s in enumerate(top_ctx[:globals().get("TOP_K", 5)]):
                        st.code(s[:800] + ("..." if len(s) > 800 else ""))

# ------------------------------
# Quizzes
# ------------------------------
elif active_tab == "Quizzes":

    st.header("Auto-generated Quizzes")
    if not st.session_state.get("uploads"):
        st.info("No uploads yet.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_quizzes")

        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if upload:
            slides = upload.get("slides", [])
            max_idx = max([int(s["index"]) for s in slides]) if slides else 0
            slide_idx = st.number_input("Slide/Page index to generate quiz from", min_value=0, max_value=max_idx, value=0)
            text = next((s.get("text", "") for s in slides if int(s.get("index", 0)) == int(slide_idx)), "")
            if st.button("Generate Quiz (MCQs)"):
                with st.spinner("Creating MCQs..."):
                    qset = _safe_call("generate_mcq_set_from_text", text, qcount=5, default=[])
                st.success("Quiz ready — try it below")
                for qi, q in enumerate(qset):
                    st.markdown(f"**Q{qi + 1}.** {q.get('question', '')}")
                    opts = q.get("options", [])
                    if not isinstance(opts, list) or len(opts) == 0:
                        opts = ["A", "B", "C", "D"]
                    choice = st.radio(f"Select Q{qi + 1}", opts, key=f"quiz_{sel_id}_{slide_idx}_{qi}")
                    if st.button(f"Submit Q{qi + 1}", key=f"submit_{sel_id}_{slide_idx}_{qi}"):
                        chosen_idx = opts.index(choice) if choice in opts else 0
                        correct_idx = int(q.get("answer_index", 0))
                        if chosen_idx == correct_idx:
                            st.success("Correct")
                        else:
                            correct_text = opts[correct_idx] if 0 <= correct_idx < len(opts) else "Unknown"
                            st.error(f"Incorrect — correct answer: {correct_text}")
                # try to persist
                cur = _get_db_cursor()
                if cur:
                    try:
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
    if not st.session_state.get("uploads"):
        st.info("No uploads yet. Go to Upload & Process.")
    else:
        options = {u["id"]: u["filename"] for u in st.session_state["uploads"]}
        sel_id = st.selectbox("Select upload to work with", options=list(options.keys()), format_func=lambda k: options[k], key="select_upload_flashcards")

        upload = next((u for u in st.session_state["uploads"] if u["id"] == sel_id), None)
        if not upload:
            st.warning("Selected upload not found.")
        else:
            slides = upload.get("slides", [])
            max_idx = max([int(s["index"]) for s in slides]) if slides else 0
            slide_idx = st.number_input("Slide/Page index to extract flashcards from", min_value=0, max_value=max_idx, value=0, key=f"flash_idx_{sel_id}")
            text = next((s.get("text", "") for s in slides if int(s.get("index", 0)) == int(slide_idx)), "")

            # generate flashcards
            if st.button("Generate Flashcards from this slide/page", key=f"gen_flash_{sel_id}_{slide_idx}"):
                with st.spinner("Generating flashcards..."):
                    cards = _safe_call("generate_flashcards_from_text", text, n=12, default=[])
                if not cards:
                    st.warning("No flashcards generated (helper not available or returned no cards).")
                else:
                    cur = _get_db_cursor()
                    if cur is None:
                        st.warning("Database not available; generated flashcards will not be persisted.")
                    else:
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
            cur = _get_db_cursor()
            if cur is not None:
                try:
                    db_uid = upload_db_id(upload)
                    cur.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards WHERE upload_id = ? AND (next_review IS NULL OR next_review <= ?) ORDER BY next_review ASC",
                                (db_uid, now))
                    due_cards = cur.fetchall()
                except Exception as e:
                    log("Failed to fetch due cards:", e)
                    due_cards = []
            else:
                due_cards = []

            if not due_cards:
                st.info("No cards due for this upload. Generate some or wait for scheduled review.")
            else:
                for row in due_cards:
                    # row could be tuple depending on DB
                    try:
                        fid, qtext, atext, eas, inter, reps, nxt = row
                    except Exception:
                        # row format unexpected - convert via indices carefully
                        fid = row[0]
                        qtext = row[1] if len(row) > 1 else ""
                        atext = row[2] if len(row) > 2 else ""
                        eas = row[3] if len(row) > 3 else 2.5
                        inter = row[4] if len(row) > 4 else 1
                        reps = row[5] if len(row) > 5 else 0
                        nxt = row[6] if len(row) > 6 else 0

                    st.markdown(f"**Q:** {qtext}")
                    if st.button(f"Show Answer", key=f"show_{fid}"):
                        st.markdown(f"**A:** {atext}")
                        rating = st.slider("How well did you recall? (0-5)", 0, 5, 3, key=f"rating_{fid}")
                        if st.button("Submit Rating", key=f"submit_rating_{fid}"):
                            try:
                                eas_new, interval_new, reps_new, next_review = _safe_call("sm2_update_card", eas, inter, reps, rating, default=(eas, inter, reps, int(time.time())))
                                cur.execute("UPDATE flashcards SET easiness = ?, interval = ?, repetitions = ?, next_review = ? WHERE id = ?",
                                            (eas_new, interval_new, reps_new, next_review, fid))
                                _db_conn.commit()
                                st.success("Card updated; next review scheduled.")
                            except Exception as e:
                                st.error(f"Failed to update card: {e}")

# ------------------------------
# Export
# ------------------------------
elif active_tab == "Export":

    st.header("Export — Anki / Audio / Raw")
    if not st.session_state.get("uploads"):
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
                    res = _safe_call("anki_export_csv_for_upload", upload_db_id(upload), _db_conn, default=None)
                    if res:
                        fname, data = res
                        st.download_button("Download Anki TSV", data, file_name=fname, mime="text/tab-separated-values", key=f"dl_anki_{sel_id}")
                    else:
                        st.warning("anki_export_csv_for_upload helper not available.")
                except Exception as e:
                    st.error(f"Export failed: {e}")

            st.markdown("---")
            if globals().get("_HAS_GTTS"):
                if st.button("Export all generated lessons as MP3 (single)", key=f"export_mp3_{sel_id}"):
                    lesson_text = ""
                    for s in upload.get("slides", [])[:50]:
                        lesson_text += f"Slide {s['index']}. {s.get('text','')}\n\n"
                    try:
                        res = _safe_call("text_to_speech_download", lesson_text, default=None)
                        if res:
                            fname, data = res
                            st.download_button("Download lessons MP3", data, file_name=fname, mime="audio/mpeg", key=f"dl_mp3_{sel_id}")
                        else:
                            st.warning("TTS helper not available.")
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

    cur = _get_db_cursor()
    rows = []
    if cur is not None:
        try:
            cur.execute("SELECT id, filename, uploaded_at, meta FROM uploads ORDER BY uploaded_at DESC")
            rows = cur.fetchall()
        except Exception as e:
            log("Failed to fetch uploads from DB:", e)
            rows = []

    if not rows:
        st.info("No uploads logged in DB yet.")
    else:
        for r in rows:
            try:
                uid, fname, uploaded_at, meta = r
            except Exception:
                # best-effort unpack
                uid = r[0]; fname = r[1]; uploaded_at = r[2]; meta = r[3] if len(r) > 3 else "{}"
            st.markdown(f"**{fname}** — uploaded at {time.ctime(int(uploaded_at))}")
            meta_obj = None
            try:
                meta_obj = json.loads(meta) if isinstance(meta, str) else meta
            except Exception:
                meta_obj = {}
            st.write(meta_obj)
            c2 = _get_db_cursor()
            if c2:
                try:
                    c2.execute("SELECT COUNT(*) FROM flashcards WHERE upload_id = ?", (uid,))
                    fc_count = int(c2.fetchone()[0] or 0)
                    c2.execute("SELECT COUNT(*) FROM quizzes WHERE upload_id = ?", (uid,))
                    q_count = int(c2.fetchone()[0] or 0)
                except Exception as e:
                    log("Failed counting artifacts:", e)
                    fc_count = 0; q_count = 0
            else:
                fc_count = 0; q_count = 0
            st.write(f"Flashcards: {fc_count} • Quizzes: {q_count}")
            st.markdown("---")

    c = _get_db_cursor()
    total_fc = total_q = 0
    if c:
        try:
            c.execute("SELECT COUNT(*) FROM flashcards")
            total_fc = int(c.fetchone()[0] or 0)
            c.execute("SELECT COUNT(*) FROM quizzes")
            total_q = int(c.fetchone()[0] or 0)
        except Exception as e:
            log("Failed to fetch totals:", e)
    st.write(f"Total flashcards in DB: {total_fc}")
    st.write(f"Total quizzes in DB: {total_q}")

# ------------------------------
# Settings
# ------------------------------
elif active_tab == "Settings":

    st.header("Settings & Diagnostics")
    if "OPENROUTER_API_KEY" not in st.session_state:
        st.session_state["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", globals().get("DEFAULT_OPENROUTER_KEY", ""))

    key = st.text_input("OpenRouter API key (leave blank to use env/default)", value=st.session_state.get("OPENROUTER_API_KEY", ""), type="password", key="api_key_input")
    if st.button("Save API Key (session only)", key="save_api_key"):
        st.session_state["OPENROUTER_API_KEY"] = (key.strip() or st.session_state.get("OPENROUTER_API_KEY", ""))
        st.success("API key set for this session.")

    st.markdown("Diagnostics:")
    st.write({
        "faiss_available": bool(globals().get("_HAS_FAISS", False)),
        "pymupdf_available": bool(globals().get("_HAS_PYMUPDF", False)),
        "easyocr_available": bool(globals().get("_HAS_EASYOCR", False)),
        "sentence_transformers_available": bool(globals().get("_HAS_SENTENCE_TRANSFORMERS", False)),
        "gtts_available": bool(globals().get("_HAS_GTTS", False))
    })

    if st.button("Test OpenRouter (small ping)", key="test_openrouter"):
        try:
            ping = _safe_call("call_openrouter_chat", "You are a test bot.", "Say 'pong' in a plain short reply.", max_tokens=20, default="[OpenRouter not available]")
            st.code(ping)
        except Exception as e:
            st.error(f"OpenRouter test failed: {e}")

    if st.button("Clear all session uploads (session state only)", key="clear_uploads_btn"):
        st.session_state["uploads"] = []
        st.success("Cleared session uploads.")
