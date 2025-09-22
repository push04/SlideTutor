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

# ----------------------------
# Helper: graceful optional imports
# ----------------------------
def _try_import(module_name: str, package_name: Optional[str] = None) -> Tuple[Optional[ModuleType], bool]:
    """
    Try to import a module by name. Returns (module_or_None, bool_installed).
    Logs a warning if missing.
    """
    if package_name is None:
        package_name = module_name
    try:
        module = __import__(module_name)
        return module, True
    except Exception:
        logging.getLogger(__name__).warning(
            "Optional module '%s' not found. To enable features install: pip install %s",
            module_name, package_name
        )
        return None, False

# Attempt optional imports
st, _HAS_STREAMLIT = _try_import("streamlit")
np, _HAS_NUMPY = _try_import("numpy")
requests, _HAS_REQUESTS = _try_import("requests")
pptx_module, _HAS_PPTX = _try_import("pptx", "python-pptx")
fitz, _HAS_PYMUPDF = _try_import("fitz", "PyMuPDF")  # PyMuPDF exposes `fitz`
easyocr_module, _HAS_EASYOCR = _try_import("easyocr")
sentence_transformers_module, _HAS_SENTENCE_TRANSFORMERS = _try_import("sentence_transformers")
faiss, _HAS_FAISS = _try_import("faiss", "faiss-cpu")
gtts_module, _HAS_GTTS = _try_import("gtts")
PIL_module, _HAS_PIL = _try_import("PIL", "Pillow")

# Map some names
easyocr = easyocr_module if _HAS_EASYOCR else None
SentenceTransformer = None
if _HAS_SENTENCE_TRANSFORMERS:
    try:
        # import the class directly if package present
        SentenceTransformer = sentence_transformers_module.SentenceTransformer
    except Exception:
        SentenceTransformer = None

# If Streamlit missing, we cannot continue — but we should raise a clear error.
if not _HAS_STREAMLIT:
    raise RuntimeError("Streamlit is required. Install: pip install streamlit")

# If numpy missing, raise as numeric ops rely on it.
if not _HAS_NUMPY:
    raise RuntimeError("NumPy is required. Install: pip install numpy")

# ----------------------------
# Logging + configuration
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("slidetutor")

# Config from environment or Streamlit secrets (if available)
DEFAULT_OPENROUTER_KEY: str = (
    (st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") and st.secrets is not None else None)
    or os.getenv("OPENROUTER_API_KEY", "")
)
OPENROUTER_API_URL: str = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K: int = int(os.getenv("TOP_K", "5"))
DB_PATH: str = os.getenv("SLIDETUTOR_DB", "slidetutor.sqlite3")
APP_TITLE = "SlideTutor AI"
APP_SUBTITLE = "Your personal AI tutor for lecture slides and documents."

# ----------------------------
# Types
# ----------------------------
class SlideData(TypedDict):
    index: int
    text: str
    images: List[bytes]
    ocr_text: str

class APIError(Exception):
    pass

# ----------------------------
# Database
# ----------------------------
# Use st.cache_resource to reuse connection across Streamlit reruns
@st.cache_resource
def get_db_connection(path: str = DB_PATH) -> sqlite3.Connection:
    logger.info("Opening DB: %s", path)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    """Initialize DB schema."""
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

# Initialize DB at import time (safe-guarded)
try:
    db_conn = get_db_connection()
    init_db(db_conn)
except Exception as e:
    logger.exception("Failed initializing DB: %s", e)
    # Re-raise to avoid continuing in a broken state
    raise

# DB helpers
def save_upload_record(conn: sqlite3.Connection, filename: str, meta: Optional[Dict] = None) -> int:
    """Insert an uploads record and return its id."""
    ts = int(time.time())
    meta_json = json.dumps(meta or {})
    cur = conn.cursor()
    cur.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)", (filename, ts, meta_json))
    conn.commit()
    upload_id = cur.lastrowid
    logger.info("Saved upload record id=%s filename=%s", upload_id, filename)
    return upload_id

def fetch_uploads(conn: sqlite3.Connection, limit: int = 50) -> List[Dict]:
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

# ----------------------------
# Vector index (FAISS optional, numpy fallback)
# ----------------------------
_NORM_EPS = 1e-9

def _ensure_2d(arr: "np.ndarray") -> "np.ndarray":
    arr = np.asarray(arr)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr

def _safe_normalize(arr: "np.ndarray", axis: int = 1) -> "np.ndarray":
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm = np.where(norm < _NORM_EPS, 1.0, norm)
    return arr / norm

class VectorIndex:
    """
    Simple wrapper around FAISS (if available) or NumPy for cosine-similarity search.
    Stores original texts and embeddings in-memory.
    """
    def __init__(self, embeddings: Optional["np.ndarray"], texts: List[str]):
        self.texts = list(texts or [])
        if embeddings is None:
            self.embeddings = np.zeros((0, 0), dtype=np.float32)
        else:
            arr = np.asarray(embeddings, dtype=np.float32)
            arr = arr.reshape((arr.shape[0], -1)) if arr.ndim == 2 else _ensure_2d(arr)
            self.embeddings = arr
        # dimension
        self.dimension = self.embeddings.shape[1] if self.embeddings.size else 0
        # normalized embeddings for cosine
        self._normed_embeddings = _safe_normalize(self.embeddings) if self.embeddings.size else np.zeros_like(self.embeddings)
        # faiss index
        self._use_faiss = False
        self.faiss_index = None
        if _HAS_FAISS and self.embeddings.size:
            try:
                # Use inner-product on normalized vectors => cosine similarity
                idx = faiss.IndexFlatIP(self.dimension)
                idx.add(np.ascontiguousarray(self._normed_embeddings))
                self.faiss_index = idx
                self._use_faiss = True
                logger.info("FAISS index built with %d vectors (dim=%d)", self.embeddings.shape[0], self.dimension)
            except Exception as e:
                logger.warning("FAISS index build failed, falling back to NumPy: %s", e)
                self._use_faiss = False

    def search(self, query_embeddings: "np.ndarray", k: int) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Returns (dists, idxs) where dists are 1 - cosine_sim (so lower is better),
        and idxs are integer indices (or -1 for padded).
        """
        queries = _ensure_2d(np.asarray(query_embeddings, dtype=np.float32))
        if self._use_faiss and self.faiss_index is not None and self.dimension:
            try:
                qn = np.ascontiguousarray(_safe_normalize(queries))
                sims, idxs = self.faiss_index.search(qn, k)
                # sims are inner-product similarities; convert to distances
                dists = 1.0 - sims.astype(np.float32)
                return dists, idxs.astype(np.int64)
            except Exception as e:
                logger.warning("FAISS search failed: %s -- falling back to numpy", e)
        return self._numpy_search(queries, k)

    def _numpy_search(self, queries: "np.ndarray", k: int) -> Tuple["np.ndarray", "np.ndarray"]:
        if self.embeddings.size == 0:
            return np.full((queries.shape[0], k), np.inf, dtype=np.float32), np.full((queries.shape[0], k), -1, dtype=np.int64)
        nq = _safe_normalize(queries)
        sims = nq @ self._normed_embeddings.T  # shape (nq, n)
        # For each query, pick top-k highest sims
        k_eff = min(k, sims.shape[1])
        idxs = np.argsort(-sims, axis=1)[:, :k_eff]
        final_sims = np.take_along_axis(sims, idxs, axis=1)
        # pad if k_eff < k
        if k_eff < k:
            pad_idxs = np.full((sims.shape[0], k - k_eff), -1, dtype=np.int64)
            pad_dists = np.full((sims.shape[0], k - k_eff), np.inf, dtype=np.float32)
            idxs = np.concatenate([idxs, pad_idxs], axis=1)
            final_sims = np.concatenate([final_sims, pad_dists], axis=1)
        return 1.0 - final_sims.astype(np.float32), idxs.astype(np.int64)

# ----------------------------
# Embeddings and chunking
# ----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Load and cache the sentence-transformers model.
    Throws RuntimeError if the library is not installed.
    """
    if not _HAS_SENTENCE_TRANSFORMERS or SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. `pip install sentence-transformers` to enable embeddings.")
    try:
        model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)
        return model
    except Exception as e:
        logger.exception("Failed to load embedding model: %s", e)
        raise

def embed_texts(texts: List[str], model) -> "np.ndarray":
    """Return a 2D numpy array of embeddings. Safe for empty list."""
    if not texts:
        # infer dimension if possible
        try:
            dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
            if not dim:
                dim = 384
        except Exception:
            dim = 384
        return np.zeros((0, dim), dtype=np.float32)
    try:
        enc = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return _ensure_2d(enc.astype(np.float32))
    except Exception as e:
        logger.exception("Embedding failed: %s", e)
        raise

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Chunk text by paragraphs into ~max_chars pieces.
    Keeps paragraph boundaries where possible.
    """
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current += (p + "\n")
        else:
            if current:
                chunks.append(current.strip())
            # if single paragraph too long, break it naïvely
            if len(p) > max_chars:
                # break into pieces
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars].strip())
                current = ""
            else:
                current = p + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def extract_json_from_text(text: str) -> Optional[Any]:
    """Try to extract first JSON (object/array) found in text. Be lenient to trailing commas."""
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if not match:
        return None
    s = match.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # attempt to clean common issues: trailing commas
        cleaned = re.sub(r',\s*([\]}])', r'\1', s)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

# ----------------------------
# OCR / File parsing
# ----------------------------
@st.cache_resource(show_spinner="Loading OCR model...")
def get_easyocr_reader(lang_list: List[str] = ["en"]) -> Optional["easyocr.Reader"]:
    """Return an EasyOCR reader if available, else None."""
    if not _HAS_EASYOCR or easyocr is None:
        logger.info("EasyOCR not available; skipping OCR.")
        return None
    try:
        reader = easyocr.Reader(lang_list, gpu=False)
        logger.info("EasyOCR reader initialized for langs: %s", lang_list)
        return reader
    except Exception as e:
        logger.warning("EasyOCR init failed: %s", e)
        return None

def _bytes_to_pil_image(b: bytes):
    """Convert raw image bytes to a PIL Image if Pillow available."""
    if not b:
        return None
    if _HAS_PIL and PIL_module is not None:
        try:
            from PIL import Image
            return Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            return None
    return None

def ocr_image_bytes(image_bytes: bytes, reader) -> str:
    """
    Run OCR on image bytes. Prefer converting to PIL (or numpy) if necessary.
    Returns concatenated text or empty string.
    """
    if not image_bytes or reader is None:
        return ""
    try:
        # EasyOCR can accept a numpy array or PIL.Image
        pil_img = _bytes_to_pil_image(image_bytes)
        if pil_img is not None:
            # convert to numpy array for easyocr
            np_img = np.array(pil_img)
            out = reader.readtext(np_img, detail=0, paragraph=True)
        else:
            # fallback: try passing raw bytes to readtext (some versions accept file path or array)
            out = reader.readtext(image_bytes, detail=0, paragraph=True)
        if not out:
            return ""
        if isinstance(out, list):
            return "\n".join([str(x).strip() for x in out if str(x).strip()])[:10000]  # cap length
        return str(out).strip()
    except Exception as e:
        logger.debug("OCR failed for image: %s", e)
        return ""

def parse_and_extract_content(filename: str, file_bytes: bytes) -> List[SlideData]:
    """
    Parse a .pptx or .pdf file from bytes and extract per-slide/page text + images.
    Returns a list of SlideData entries.
    """
    file_ext = os.path.splitext(filename)[1].lower()
    raw_slides: List[Dict[str, Any]] = []
    if file_ext == ".pptx":
        if not _HAS_PPTX or pptx_module is None:
            raise RuntimeError("python-pptx not installed. Install: pip install python-pptx")
        try:
            Presentation = pptx_module.Presentation
            prs = Presentation(io.BytesIO(file_bytes))
            for i, slide in enumerate(prs.slides):
                texts = []
                images: List[bytes] = []
                for shape in slide.shapes:
                    try:
                        if getattr(shape, "has_text_frame", False):
                            txt = shape.text or ""
                            if txt.strip():
                                texts.append(txt.strip())
                    except Exception:
                        pass
                    try:
                        # Access shape.image.blob in a safe way
                        img = getattr(shape, "image", None)
                        if img is not None and getattr(img, "blob", None) is not None:
                            images.append(img.blob)
                    except Exception:
                        # Some shapes don't have image attribute
                        pass
                raw_slides.append({"index": i, "text": "\n".join(texts).strip(), "images": images})
        except Exception as e:
            logger.exception("Failed to parse PPTX: %s", e)
            raise
    elif file_ext == ".pdf":
        if not _HAS_PYMUPDF or fitz is None:
            raise RuntimeError("PyMuPDF not installed. Install: pip install PyMuPDF")
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for i, page in enumerate(doc):
                text = page.get_text("text") or ""
                text = text.strip()
                images: List[bytes] = []
                try:
                    for img in page.get_images(full=True):
                        xref = img[0]
                        try:
                            imgdict = doc.extract_image(xref)
                            if imgdict and "image" in imgdict:
                                images.append(imgdict["image"])
                        except Exception:
                            logger.debug("Could not extract image xref=%s on page=%s", xref, i)
                    # If no images extracted and no text, snapshot the page
                    if not text and not images:
                        try:
                            pix = page.get_pixmap(dpi=150)
                            images.append(pix.tobytes("png"))
                        except Exception:
                            pass
                except Exception:
                    logger.debug("Failed while extracting images from page %s", i)
                raw_slides.append({"index": i, "text": text, "images": images})
        except Exception as e:
            logger.exception("Failed to parse PDF: %s", e)
            raise
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    # Run OCR on images if available
    ocr_reader = get_easyocr_reader()
    processed: List[SlideData] = []
    for s in raw_slides:
        ocr_texts = []
        for img in s.get("images", []):
            try:
                t = ocr_image_bytes(img, ocr_reader)
                if t:
                    ocr_texts.append(t)
            except Exception:
                continue
        ocr_text = "\n".join(ocr_texts).strip()
        processed.append({"index": int(s["index"]), "text": s.get("text", "") or "", "images": s.get("images", []), "ocr_text": ocr_text})
    return processed

# ----------------------------
# LLM / OpenRouter call wrapper
# ----------------------------
def call_openrouter(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 1500, temperature: float = 0.1) -> str:
    """
    Call the OpenRouter-compatible API. Uses st.session_state override if provided.
    Raises APIError on problems.
    """
    api_key = (st.session_state.get("OPENROUTER_API_KEY") if hasattr(st, "session_state") else None) or DEFAULT_OPENROUTER_KEY
    if not api_key:
        raise APIError("OpenRouter API key not configured. Set OPENROUTER_API_KEY env or in Streamlit secrets.")
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
        body = resp.json()
        # Try common response layouts
        # OpenRouter typically returns choices -> [{message: {content: "..."} }]
        choices = body.get("choices") or []
        if choices:
            first = choices[0]
            # message.content or text
            content = None
            if isinstance(first, dict):
                content = (first.get("message") or {}).get("content") or first.get("text") or first.get("message", {}).get("content")
            if not content:
                # fallback: maybe body has 'text'
                content = body.get("text")
            if not content:
                raise APIError("Empty content from OpenRouter response")
            return str(content).strip()
        # If no choices but top-level 'output' or 'results'
        if "output" in body:
            return str(body["output"]).strip()
        raise APIError("No choices returned from OpenRouter")
    except requests.exceptions.RequestException as e:
        logger.exception("OpenRouter request failed: %s", e)
        raise APIError(f"OpenRouter request failed: {e}")
    except ValueError as e:
        logger.exception("Failed to decode OpenRouter response: %s", e)
        raise APIError("Invalid response from OpenRouter")

# Prompts
PROMPT_LESSON_MULTILEVEL = ("You are an expert teacher. Create a multi-level lesson (Beginner, Intermediate, Advanced) "
                           "based on the provided text. For each level, give a clear explanation, a worked example, "
                           "and key tips. Finally, add a short 3-question quiz.")
PROMPT_MCQ_JSON = ("You are an AI that generates multiple-choice questions from text. Reply ONLY with a valid JSON "
                   "array of objects, where each object has keys 'question', 'options' (4 strings), and 'answer_index' (0-based).")
PROMPT_FLASHCARDS_JSON = ("You are an AI that creates flashcards from text. Reply ONLY with a valid JSON array of objects, "
                          "where each object has keys 'question' and 'answer'.")

def generate_multilevel_lesson(context: str) -> str:
    return call_openrouter(PROMPT_LESSON_MULTILEVEL, f"Generate a lesson from this text:\n\n{context}")

def generate_mcq_set_from_text(context: str, qcount: int = 5) -> List[Dict]:
    resp = call_openrouter(PROMPT_MCQ_JSON, f"Create exactly {qcount} MCQs from this text:\n\n{context}")
    parsed = extract_json_from_text(resp)
    return parsed or []

def generate_flashcards_from_text(context: str, n: int = 10) -> List[Dict]:
    resp = call_openrouter(PROMPT_FLASHCARDS_JSON, f"Create {n} flashcards from this text:\n\n{context}")
    parsed = extract_json_from_text(resp)
    return parsed or []

# ----------------------------
# RAG Q&A & upload index builder
# ----------------------------
def build_upload_index(slides: List[SlideData], model) -> Dict[str, Any]:
    """
    Build an in-memory index for a list of slides (SlideData).
    Returns a dict with keys:
        - 'chunks': list[str]
        - 'index': VectorIndex
        - 'slides': original slides
        - 'meta': statistics
    """
    chunks: List[str] = []
    origin_map: List[Tuple[int, int]] = []  # (slide_index, chunk_local_index)
    # Create chunks combining slide text + OCR text
    for s in slides:
        combined = "\n".join([s.get("text", ""), s.get("ocr_text", "")]).strip()
        if not combined:
            continue
        pieces = chunk_text(combined, max_chars=1000)
        for pi, p in enumerate(pieces):
            chunks.append(p)
            origin_map.append((s["index"], pi))
    if not chunks:
        # create an empty index
        emb = np.zeros((0, 384), dtype=np.float32)
        idx = VectorIndex(emb, [])
        return {"chunks": [], "index": idx, "slides": slides, "meta": {"chunks": 0}}
    # embed chunks
    emb = embed_texts(chunks, model)
    idx = VectorIndex(emb, chunks)
    logger.info("Built upload index: %d chunks", len(chunks))
    return {"chunks": chunks, "index": idx, "slides": slides, "meta": {"chunks": len(chunks), "origin_map": origin_map}}

def answer_question_with_rag(query: str, uploads: List[Dict]) -> str:
    """
    Given a query and a list of in-memory uploads (each with 'index' VectorIndex and 'chunks' list),
    perform similarity search and call LLM with top-context.
    """
    if not query:
        return "Please provide a question."
    try:
        model = get_embedding_model()
    except Exception as e:
        return "Embeddings not available: " + str(e)
    q_emb = embed_texts([query], model)
    found_chunks: List[str] = []
    # each upload expected to be dict with 'index' (VectorIndex) and 'chunks'
    for upload in uploads:
        idx: VectorIndex = upload.get("index")
        chunks = upload.get("chunks") or []
        if not idx or idx.dimension == 0:
            continue
        dists, ids = idx.search(q_emb, k=min(TOP_K, max(1, len(chunks))))
        # ids for first query (ids[0])
        for j in ids[0]:
            if j is None or int(j) < 0:
                continue
            try:
                found_chunks.append(chunks[int(j)])
            except Exception:
                continue
    if not found_chunks:
        return "Couldn't find relevant information in your documents."
    # Use a reasonable context limit
    context = "\n---\n".join(found_chunks[: min(len(found_chunks), TOP_K * 3)])
    sys_prompt = "You are a precise assistant. Answer using ONLY the provided context. If not available, say you cannot answer."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    try:
        return call_openrouter(sys_prompt, user_prompt, max_tokens=600)
    except APIError as e:
        return f"LLM Error: {e}"

# ----------------------------
# SM-2 (spaced repetition)
# ----------------------------
def sm2_update_card(easiness: float, interval: int, reps: int, quality: int) -> Tuple[float, int, int, int]:
    """
    Return updated (easiness, interval, repetitions, next_review_timestamp)
    """
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
def anki_export_tsv(upload_id: int, conn: sqlite3.Connection) -> Optional[Tuple[str, bytes]]:
    cursor = conn.execute("SELECT question, answer FROM flashcards WHERE upload_id = ?", (upload_id,))
    rows = cursor.fetchall()
    if not rows:
        return None
    buf = io.StringIO()
    for q, a in rows:
        qs = q.replace("\t", " ").replace("\n", "<br>")
        as_ = a.replace("\t", " ").replace("\n", "<br>")
        buf.write(f"{qs}\t{as_}\n")
    filename = f"anki_deck_upload_{upload_id}.txt"
    return filename, buf.getvalue().encode("utf-8")

# ----------------------------
# Streamlit UI (tiny)
# ----------------------------
def main_ui():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    # Settings: allow user to provide OpenRouter key at runtime
    with st.expander("Settings / API Key (optional)"):
        key = st.text_input("OpenRouter API key (optional, overrides env/secrets)", value="", type="password")
        if key:
            st.session_state["OPENROUTER_API_KEY"] = key

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Upload file")
        uploaded = st.file_uploader("Upload a .pptx or .pdf", type=["pptx", "pdf"], accept_multiple_files=False)
        if uploaded is not None:
            filename = uploaded.name
            file_bytes = uploaded.read()
            st.write(f"Got file: **{filename}** ({len(file_bytes)} bytes)")
            if st.button("Parse and index file"):
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
                    # Build index
                    try:
                        model = get_embedding_model()
                        upload_obj = build_upload_index(slides, model)
                        # store in session for quick use during this session
                        if "uploads_in_memory" not in st.session_state:
                            st.session_state["uploads_in_memory"] = []
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

        st.markdown("### Existing uploads")
        uploads_list = fetch_uploads(db_conn, limit=20)
        for u in uploads_list:
            st.write(f"- ID {u['id']} — {u['filename']} — uploaded at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(u['uploaded_at']))}")

    with col2:
        st.header("Ask a question (RAG)")
        query = st.text_area("Question about your uploaded documents", height=120)
        if st.button("Answer question"):
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
        sample_text = st.text_area("Text to generate flashcards or MCQs from", height=120)
        col3, col4 = st.columns([1, 1])
        with col3:
            if st.button("Generate 5 MCQs"):
                if not sample_text.strip():
                    st.warning("Provide some text above to generate MCQs.")
                else:
                    try:
                        mcqs = generate_mcq_set_from_text(sample_text, qcount=5)
                        st.write(mcqs)
                    except Exception as e:
                        st.error("Failed to generate MCQs: " + str(e))
        with col4:
            if st.button("Generate 10 flashcards"):
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
# ----------------------------
# ----------------------------
# Optional imports (graceful)
# ----------------------------
def _try_import(module_name: str, package_name: Optional[str] = None) -> Tuple[Optional[ModuleType], bool]:
    if package_name is None:
        package_name = module_name
    try:
        module = __import__(module_name)
        return module, True
    except Exception:
        logging.getLogger(__name__).warning("Optional module '%s' not found. pip install %s to enable related features.", module_name, package_name)
        return None, False

st, _HAS_STREAMLIT = _try_import("streamlit")
np, _HAS_NUMPY = _try_import("numpy")
requests, _HAS_REQUESTS = _try_import("requests")
pptx_module, _HAS_PPTX = _try_import("pptx", "python-pptx")
fitz, _HAS_PYMUPDF = _try_import("fitz", "PyMuPDF")
easyocr_module, _HAS_EASYOCR = _try_import("easyocr")
sentence_transformers_module, _HAS_SENTENCE_TRANSFORMERS = _try_import("sentence_transformers")
faiss, _HAS_FAISS = _try_import("faiss", "faiss-cpu")
gtts_module, _HAS_GTTS = _try_import("gtts")
PIL_module, _HAS_PIL = _try_import("PIL", "Pillow")

# Map classes / names
easyocr = easyocr_module if _HAS_EASYOCR else None
SentenceTransformer = None
if _HAS_SENTENCE_TRANSFORMERS:
    try:
        SentenceTransformer = sentence_transformers_module.SentenceTransformer
    except Exception:
        SentenceTransformer = None

# Require Streamlit and NumPy
if not _HAS_STREAMLIT:
    raise RuntimeError("Streamlit is required. Install with: pip install streamlit")
if not _HAS_NUMPY:
    raise RuntimeError("NumPy is required. Install with: pip install numpy")

# ----------------------------
# Logging + config
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("slidetutor")

DEFAULT_OPENROUTER_KEY: str = (
    (st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") and st.secrets is not None else None)
    or os.getenv("OPENROUTER_API_KEY", "")
)
OPENROUTER_API_URL: str = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K: int = int(os.getenv("TOP_K", "5"))
DB_PATH: str = os.getenv("SLIDETUTOR_DB", "slidetutor.sqlite3")
APP_TITLE = "SlideTutor AI"
APP_SUBTITLE = "Your personal AI tutor for lecture slides and documents."

# ----------------------------
# Types
# ----------------------------
class SlideData(TypedDict):
    index: int
    text: str
    images: List[bytes]
    ocr_text: str

class APIError(Exception):
    pass

# ----------------------------
# Database helpers
# ----------------------------
@st.cache_resource
def get_db_connection(path: str = DB_PATH) -> sqlite3.Connection:
    logger.info("Opening DB at %s", path)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
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

# Initialize DB
try:
    _conn = get_db_connection()
    init_db(_conn)
except Exception as e:
    logger.exception("DB init failed: %s", e)
    raise

def save_upload_record(conn: sqlite3.Connection, filename: str, meta: Optional[Dict] = None) -> int:
    ts = int(time.time())
    meta_json = json.dumps(meta or {})
    cur = conn.cursor()
    cur.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)", (filename, ts, meta_json))
    conn.commit()
    return cur.lastrowid

def fetch_uploads(conn: sqlite3.Connection, limit: int = 50) -> List[Dict]:
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

# ----------------------------
# Vector index (FAISS optional) & helpers
# ----------------------------
_NORM_EPS = 1e-9

def _ensure_2d(arr: "np.ndarray") -> "np.ndarray":
    arr = np.asarray(arr)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr

def _safe_normalize(arr: "np.ndarray", axis: int = 1) -> "np.ndarray":
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm = np.where(norm < _NORM_EPS, 1.0, norm)
    return arr / norm

class VectorIndex:
    def __init__(self, embeddings: Optional["np.ndarray"], texts: List[str]):
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
                logger.info("Built FAISS index with %d vectors", self.embeddings.shape[0])
            except Exception as e:
                logger.warning("FAISS build failed, falling back to NumPy: %s", e)
                self._use_faiss = False

    def search(self, query_embeddings: "np.ndarray", k: int) -> Tuple["np.ndarray", "np.ndarray"]:
        queries = _ensure_2d(np.asarray(query_embeddings, dtype=np.float32))
        if self._use_faiss and self.faiss_index is not None and self.dimension:
            try:
                qn = np.ascontiguousarray(_safe_normalize(queries))
                sims, idxs = self.faiss_index.search(qn, k)
                dists = 1.0 - sims.astype(np.float32)
                return dists, idxs.astype(np.int64)
            except Exception as e:
                logger.warning("FAISS search failed: %s -- falling back to NumPy", e)
        return self._numpy_search(queries, k)

    def _numpy_search(self, queries: "np.ndarray", k: int) -> Tuple["np.ndarray", "np.ndarray"]:
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
# Embeddings & chunking
# ----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    if not _HAS_SENTENCE_TRANSFORMERS or SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed. Install: pip install sentence-transformers")
    try:
        model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)
        return model
    except Exception as e:
        logger.exception("Failed to load embedding model: %s", e)
        raise

def embed_texts(texts: List[str], model) -> "np.ndarray":
    if not texts:
        try:
            dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)() or 384
        except Exception:
            dim = 384
        return np.zeros((0, dim), dtype=np.float32)
    try:
        enc = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return _ensure_2d(enc.astype(np.float32))
    except Exception as e:
        logger.exception("Embedding texts failed: %s", e)
        raise

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
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

def extract_json_from_text(text: str) -> Optional[Any]:
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
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
# OCR / Image utils
# ----------------------------
def _bytes_to_pil_image(b: bytes):
    if not b:
        return None
    if _HAS_PIL and PIL_module is not None:
        try:
            from PIL import Image
            return Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            return None
    return None

@st.cache_resource(show_spinner="Loading OCR model...")
def get_easyocr_reader(lang_list: List[str] = ["en"]) -> Optional["easyocr.Reader"]:
    if not _HAS_EASYOCR or easyocr is None:
        logger.info("EasyOCR not available.")
        return None
    try:
        reader = easyocr.Reader(lang_list, gpu=False)
        logger.info("EasyOCR initialized for %s", lang_list)
        return reader
    except Exception as e:
        logger.warning("EasyOCR init failed: %s", e)
        return None

def ocr_image_bytes(image_bytes: bytes, reader) -> str:
    if not image_bytes or reader is None:
        return ""
    try:
        pil = _bytes_to_pil_image(image_bytes)
        if pil is not None:
            arr = np.array(pil)
            out = reader.readtext(arr, detail=0, paragraph=True)
        else:
            out = reader.readtext(image_bytes, detail=0, paragraph=True)
        if not out:
            return ""
        if isinstance(out, list):
            return "\n".join([str(x).strip() for x in out if str(x).strip()])[:10000]
        return str(out).strip()
    except Exception as e:
        logger.debug("OCR failed: %s", e)
        return ""

# ----------------------------
# File parsing: PPTX / PDF
# ----------------------------
def _extract_from_pptx(file_bytes: bytes) -> List[Dict[str, Any]]:
    if not _HAS_PPTX or pptx_module is None:
        raise ModuleNotFoundError("python-pptx not installed. Install: pip install python-pptx")
    slides_content = []
    try:
        Presentation = pptx_module.Presentation
        prs = Presentation(io.BytesIO(file_bytes))
        for i, slide in enumerate(prs.slides):
            texts: List[str] = []
            images: List[bytes] = []
            for shape in slide.shapes:
                try:
                    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
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
    except Exception as e:
        logger.exception("Failed to parse PPTX: %s", e)
        raise
    return slides_content

def _extract_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    if not _HAS_PYMUPDF or fitz is None:
        raise ModuleNotFoundError("PyMuPDF not installed. Install: pip install PyMuPDF")
    slides_content: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = (page.get_text("text") or "").strip()
            images: List[bytes] = []
            try:
                for img in page.get_images(full=True):
                    xref = img[0]
                    try:
                        imgdict = doc.extract_image(xref)
                        if imgdict and "image" in imgdict:
                            images.append(imgdict["image"])
                    except Exception:
                        logger.debug("Couldn't extract image xref=%s page=%s", xref, i)
            except Exception:
                logger.debug("Error enumerating images on page %s", i)
            if not text and not images:
                try:
                    pix = page.get_pixmap(dpi=150)
                    images.append(pix.tobytes("png"))
                except Exception:
                    pass
            slides_content.append({"index": i, "text": text, "images": images})
        doc.close()
    except Exception as e:
        logger.exception("Failed to parse PDF: %s", e)
        raise
    return slides_content

def parse_and_extract_content(filename: str, file_bytes: bytes) -> List[SlideData]:
    ext = os.path.splitext(filename)[1].lower()
    raw_slides: List[Dict[str, Any]] = []
    if ext == ".pptx":
        raw_slides = _extract_from_pptx(file_bytes)
    elif ext == ".pdf":
        raw_slides = _extract_from_pdf(file_bytes)
    else:
        raise ValueError("Unsupported file type: " + ext)

    reader = get_easyocr_reader()
    processed: List[SlideData] = []
    for s in raw_slides:
        ocr_texts: List[str] = []
        for img in s.get("images", []) or []:
            try:
                t = ocr_image_bytes(img, reader)
                if t:
                    ocr_texts.append(t)
            except Exception:
                continue
        processed.append({"index": int(s["index"]), "text": s.get("text", "") or "", "images": s.get("images", []), "ocr_text": "\n".join(ocr_texts).strip()})
    return processed

# ----------------------------
# OpenRouter LLM wrapper + prompts
# ----------------------------
def call_openrouter(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 1500, temperature: float = 0.1) -> str:
    api_key = (st.session_state.get("OPENROUTER_API_KEY") if hasattr(st, "session_state") else None) or DEFAULT_OPENROUTER_KEY
    if not api_key:
        raise APIError("OpenRouter API key not configured. Provide in Settings or set OPENROUTER_API_KEY env.")
    if not _HAS_REQUESTS or requests is None:
        raise APIError("requests library not available.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role":"system","content":system_prompt}, {"role":"user","content":user_prompt}], "max_tokens": max_tokens, "temperature": temperature}
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            content = (first.get("message") or {}).get("content") or first.get("text") or data.get("text")
            if not content:
                raise APIError("Empty content from OpenRouter response")
            return str(content).strip()
        # fallback
        if "output" in data:
            return str(data["output"]).strip()
        raise APIError("No choices returned from OpenRouter")
    except requests.exceptions.RequestException as e:
        logger.exception("OpenRouter request failed: %s", e)
        raise APIError(f"OpenRouter request failed: {e}")
    except ValueError as e:
        logger.exception("OpenRouter response decode failed: %s", e)
        raise APIError("Invalid response from OpenRouter")

PROMPT_LESSON_MULTILEVEL = ("You are an expert teacher. Create a multi-level lesson (Beginner, Intermediate, Advanced) "
                           "based on the provided text. For each level, give a clear explanation, a worked example, "
                           "and key tips. Finally, add a short 3-question quiz.")
PROMPT_MCQ_JSON = ("You are an AI that generates multiple-choice questions from text. Reply ONLY with a valid JSON "
                   "array of objects, where each object has keys 'question', 'options' (4 strings), and 'answer_index' (0-based).")
PROMPT_FLASHCARDS_JSON = ("You are an AI that creates flashcards from text. Reply ONLY with a valid JSON array of objects, "
                          "where each object has keys 'question' and 'answer'.")

def generate_multilevel_lesson(context: str) -> str:
    return call_openrouter(PROMPT_LESSON_MULTILEVEL, f"Generate a lesson from this text:\n\n{context}")

def generate_mcq_set_from_text(context: str, qcount: int = 5) -> List[Dict]:
    resp = call_openrouter(PROMPT_MCQ_JSON, f"Create exactly {qcount} MCQs from this text:\n\n{context}")
    parsed = extract_json_from_text(resp)
    return parsed or []

def generate_flashcards_from_text(context: str, n: int = 10) -> List[Dict]:
    resp = call_openrouter(PROMPT_FLASHCARDS_JSON, f"Create {n} flashcards from this text:\n\n{context}")
    parsed = extract_json_from_text(resp)
    return parsed or []

# ----------------------------
# RAG pipeline
# ----------------------------
def build_upload_index(slides: List[SlideData], model) -> Dict[str, Any]:
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
    return {"chunks": chunks, "index": idx, "slides": slides, "meta": {"chunks": len(chunks), "origin_map": origin_map}}

def answer_question_with_rag(query: str, uploads: List[Dict]) -> str:
    if not query:
        return "Please provide a question."
    try:
        model = get_embedding_model()
    except Exception as e:
        return "Embeddings not available: " + str(e)
    q_emb = embed_texts([query], model)
    found_chunks = []
    for upload in uploads:
        idx: VectorIndex = upload.get("index")
        chunks = upload.get("chunks") or []
        if not idx or idx.dimension == 0:
            continue
        dists, ids = idx.search(q_emb, k=min(TOP_K, max(1, len(chunks))))
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
def sm2_update_card(easiness: float, interval: int, reps: int, quality: int) -> Tuple[float, int, int, int]:
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
def anki_export_tsv(upload_id: int, conn: sqlite3.Connection) -> Optional[Tuple[str, bytes]]:
    cursor = conn.execute("SELECT question, answer FROM flashcards WHERE upload_id = ?", (upload_id,))
    rows = cursor.fetchall()
    if not rows:
        return None
    buf = io.StringIO()
    for q, a in rows:
        qs = q.replace("\t", " ").replace("\n", "<br>")
        as_ = a.replace("\t", " ").replace("\n", "<br>")
        buf.write(f"{qs}\t{as_}\n")
    filename = f"anki_deck_upload_{upload_id}.txt"
    return filename, buf.getvalue().encode("utf-8")

# ----------------------------
# Beautiful minimal CSS (professional & simple)
# ----------------------------
APP_CSS = """
/* Import Inter if available */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

:root{
  --bg:#0f1724;
  --card:#0b1220;
  --muted:#94a3b8;
  --accent:#7c3aed;
  --glass: rgba(255,255,255,0.03);
  --radius:14px;
}

body, .stApp, .main {
  font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  background: linear-gradient(180deg, #071029 0%, #071229 100%);
  color: #e6eef8;
}

/* Card-like containers for main areas */
.stApp .block-container {
  padding: 28px 32px !important;
}

.card {
  background: var(--glass);
  border-radius: var(--radius);
  padding: 18px;
  box-shadow: 0 6px 18px rgba(5,10,20,0.6);
  border: 1px solid rgba(255,255,255,0.03);
  margin-bottom: 16px;
}

h1, h2, h3 {
  color: #f1f5f9;
  font-weight: 700;
}

/* Buttons */
.element-container .stButton>button {
  border-radius: 10px !important;
  padding: 8px 14px !important;
  background: linear-gradient(90deg, var(--accent), #5b21b6) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 4px 12px rgba(124,58,237,0.18);
}

/* Sidebar style */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(10,14,22,0.7), rgba(10,14,22,0.4));
  border-right: 1px solid rgba(255,255,255,0.03);
}

/* Inputs / text areas */
textarea, input, .stTextInput, .stTextArea {
  border-radius: 10px !important;
}

/* Small muted text */
.small-muted {
  color: var(--muted);
  font-size: 0.92rem;
}

/* Table-like preview */
.preview-slide {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  padding: 12px;
  border-radius: 10px;
  margin-bottom: 8px;
  border: 1px solid rgba(255,255,255,0.02);
}
"""

# ----------------------------
# Session state & UI helpers
# ----------------------------
def initialize_session_state():
    defaults = {
        "uploads": [],  # list of dicts: filename, file_bytes, status_msg, slides_data, chunks, embeddings, index, index_built, db_id, full_text
        "OPENROUTER_API_KEY": st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") and st.secrets is not None else os.getenv("OPENROUTER_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def handle_file_processing(upload_idx: int):
    try:
        upload = st.session_state["uploads"][upload_idx]
    except Exception:
        st.error("Upload not found in session.")
        return
    filename = upload.get("filename", "file")
    try:
        with st.spinner(f"Processing {filename} ..."):
            upload["status_msg"] = "Extracting content..."
            slides = parse_and_extract_content(filename, upload["file_bytes"])
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
            # Embeddings & index
            if _HAS_SENTENCE_TRANSFORMERS and SentenceTransformer is not None:
                try:
                    upload["status_msg"] = "Embedding chunks..."
                    model = get_embedding_model()
                    upload["embeddings"] = embed_texts(upload["chunks"], model)
                    upload["index"] = VectorIndex(upload["embeddings"], upload["chunks"])
                    upload["index_built"] = True
                except Exception as e:
                    logger.exception("Embedding/index build failed: %s", e)
                    upload["index_built"] = False
                    upload["status_msg"] = "Embedding failed"
            else:
                upload["index_built"] = False
                upload["status_msg"] = "Embeddings unavailable"
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

# UI render parts
def render_home():
    st.markdown(f"<div class='card'><h1 style='margin:0'>{APP_TITLE}</h1><p class='small-muted'>{APP_SUBTITLE}</p></div>", unsafe_allow_html=True)
    st.write("Use the left tab bar to upload files, build indexes, generate lessons, MCQs, flashcards, and ask questions using RAG.")

def render_upload_tab():
    st.header("Upload & Process")
    st.markdown("<div class='card'>Use this uploader to add PPTX or PDF files. After upload, click 'Process' to extract text and build an index (if embeddings available).</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("PPTX / PDF (multiple)", type=["pptx", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        existing = {u["filename"] for u in st.session_state.uploads}
        for f in uploaded_files:
            if f.name not in existing:
                st.session_state.uploads.append({
                    "filename": f.name,
                    "file_bytes": f.getvalue(),
                    "status_msg": "Ready (not processed)",
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
    for i, up in enumerate(list(st.session_state.uploads)):
        with st.expander(f"{up['filename']} — {up['status_msg']}", expanded=False):
            cols = st.columns([3,1,1,1])
            cols[0].markdown(f"**{up['filename']}**\n\nStatus: {up['status_msg']}")
            if cols[1].button("Process", key=f"proc_{i}", disabled=up.get("index_built", False)):
                handle_file_processing(i)
                st.experimental_rerun()
            if cols[2].button("Preview slides", key=f"preview_{i}"):
                slides = up.get("slides_data", [])
                if not slides:
                    st.warning("Process file first to preview slides.")
                else:
                    for s in slides[:50]:
                        st.markdown(f"<div class='preview-slide'><strong>Slide {s['index']+1}</strong><div style='margin-top:8px'>{(s['text'] or '')[:400]}</div>{('<em>OCR:</em><div>'+ (s['ocr_text'] or '')[:300] + '</div>') if s.get('ocr_text') else ''}</div>", unsafe_allow_html=True)
            if cols[3].button("Delete", key=f"del_{i}"):
                try:
                    if up.get("db_id"):
                        conn = get_db_connection()
                        with conn:
                            conn.execute("DELETE FROM uploads WHERE id = ?", (up["db_id"],))
                except Exception:
                    pass
                st.session_state.uploads.pop(i)
                st.experimental_rerun()

def render_lessons_tab():
    st.header("Lessons")
    processed_names = ["<none>"] + [u["filename"] for u in st.session_state.uploads if u.get("full_text")]
    selected = st.selectbox("Choose upload", processed_names)
    if selected == "<none>":
        st.info("Select a processed upload to generate a lesson.")
        return
    upload = next((u for u in st.session_state.uploads if u["filename"] == selected), None)
    text = upload.get("full_text") or "\n".join(upload.get("chunks") or [])
    if not text.strip():
        st.warning("No extracted text in this document.")
        return
    with st.form("lesson_form"):
        level = st.selectbox("Level", ["Beginner->Advanced", "Beginner only", "Beginner+Intermediate"])
        q = st.number_input("Number of quiz questions", min_value=1, max_value=10, value=3)
        submit = st.form_submit_button("Generate lesson")
    if submit:
        try:
            ctx = text
            lesson = generate_multilevel_lesson(ctx)
            st.success("Lesson generated")
            st.text_area("Generated lesson", lesson, height=400)
        except APIError as e:
            st.error(f"LLM error: {e}")
        except Exception as e:
            st.exception(e)

def render_mcq_tab():
    st.header("Generate MCQs")
    processed_names = ["<none>"] + [u["filename"] for u in st.session_state.uploads if u.get("full_text")]
    selected = st.selectbox("Choose processed upload", processed_names, key="mcq_select")
    if selected == "<none>":
        st.info("Select a processed upload")
        return
    upload = next((u for u in st.session_state.uploads if u["filename"] == selected), None)
    n = st.slider("Number of MCQs", 1, 20, 5)
    if st.button("Generate MCQs"):
        try:
            context = upload.get("full_text") or "\n".join(upload.get("chunks") or [])
            mcqs = generate_mcq_set_from_text(context, qcount=n)
            if not mcqs:
                st.warning("No MCQs returned.")
            else:
                conn = get_db_connection()
                with conn:
                    now = int(time.time())
                    for obj in mcqs:
                        qtext = obj.get("question", "")
                        opts = obj.get("options") or []
                        ans = int(obj.get("answer_index", 0))
                        conn.execute("INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                                     (upload.get("db_id"), qtext, json.dumps(opts), ans, now))
                st.success(f"Generated and saved {len(mcqs)} MCQs")
                st.json(mcqs)
        except APIError as e:
            st.error(f"LLM error: {e}")
        except Exception as e:
            st.exception(e)

def render_flashcards_tab():
    st.header("Flashcards")
    processed_names = ["<none>"] + [u["filename"] for u in st.session_state.uploads if u.get("full_text")]
    selected = st.selectbox("Choose processed upload", processed_names, key="fc_select")
    if selected == "<none>":
        st.info("Select a processed upload")
        return
    upload = next((u for u in st.session_state.uploads if u["filename"] == selected), None)
    n = st.number_input("How many flashcards?", min_value=1, max_value=100, value=10)
    if st.button("Generate flashcards"):
        try:
            context = upload.get("full_text") or "\n".join(upload.get("chunks") or [])
            fcs = generate_flashcards_from_text(context, n=n)
            if not fcs:
                st.warning("No flashcards returned.")
            else:
                conn = get_db_connection()
                with conn:
                    for obj in fcs:
                        qtext = (obj.get("question") or "")[:2000]
                        ans = (obj.get("answer") or "")[:2000]
                        conn.execute("INSERT INTO flashcards (upload_id, question, answer, next_review) VALUES (?, ?, ?, ?)",
                                     (upload.get("db_id"), qtext, ans, int(time.time())))
                st.success(f"Saved {len(fcs)} flashcards")
                st.json(fcs)
        except APIError as e:
            st.error(f"LLM error: {e}")
        except Exception as e:
            st.exception(e)
    st.markdown("### Review flashcards (SM-2)")
    conn = get_db_connection()
    rows = conn.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards ORDER BY next_review ASC LIMIT 20").fetchall()
    if not rows:
        st.info("No flashcards stored.")
    else:
        for (cid, q, a, eas, ivl, reps, nxt) in rows:
            with st.expander(q):
                st.write(a)
                cols = st.columns([1,1,1,1,1])
                quality = cols[0].radio("Quality (0-5)", [0,1,2,3,4,5], key=f"q_{cid}", index=5)
                if cols[4].button("Mark", key=f"m_{cid}"):
                    new_eas, new_ivl, new_reps, new_next = sm2_update_card(eas, ivl, reps, int(quality))
                    with conn:
                        conn.execute("UPDATE flashcards SET easiness=?, interval=?, repetitions=?, next_review=? WHERE id=?",
                                     (float(new_eas), int(new_ivl), int(new_reps), int(new_next), cid))
                    st.success("Updated review schedule")

def render_chat_tab():
    st.header("Chat Q&A (RAG over your uploads)")
    query = st.text_input("Ask a question about your documents", key="rag_query")
    selected_uploads = [u for u in st.session_state.uploads if u.get("index_built")]
    if st.button("Answer"):
        if not selected_uploads:
            st.warning("No processed uploads with embeddings. Process uploads first.")
        else:
            with st.spinner("Searching..."):
                answer = answer_question_with_rag(query, selected_uploads)
                st.markdown("**Answer:**")
                st.write(answer)

def render_quizzes_tab():
    st.header("Quizzes")
    conn = get_db_connection()
    upload_ids = {u["db_id"]: u["filename"] for u in st.session_state.uploads if u.get("db_id")}
    if not upload_ids:
        st.info("No quizzes stored yet.")
        return
    sel = st.selectbox("Select upload quizzes", ["<all>"] + list(upload_ids.values()))
    if st.button("Take quiz"):
        if sel == "<all>":
            rows = conn.execute("SELECT question, options, correct_index FROM quizzes ORDER BY RANDOM() LIMIT 10").fetchall()
        else:
            uid = next((k for k,v in upload_ids.items() if v==sel), None)
            rows = conn.execute("SELECT question, options, correct_index FROM quizzes WHERE upload_id=? ORDER BY RANDOM() LIMIT 10", (uid,)).fetchall()
        if not rows:
            st.info("No quiz questions available.")
            return
        score = 0
        for idx, (q, opts_json, correct_index) in enumerate(rows):
            opts = json.loads(opts_json)
            answer = st.radio(f"Q{idx+1}: {q}", opts, key=f"quiz_{idx}")
            if st.button("Submit answer", key=f"sub_{idx}"):
                if opts.index(answer) == correct_index:
                    st.success("Correct!")
                    score += 1
                else:
                    st.error(f"Wrong. Correct: {opts[correct_index]}")
        st.write("Finish — your score (this session):", score)

def render_settings_tab():
    st.header("Settings")
    st.text_input("OpenRouter API Key (stored in session only)", key="OPENROUTER_API_KEY", type="password")
    st.write("Environment variables:", {"OPENROUTER_API_URL": OPENROUTER_API_URL, "EMBEDDING_MODEL": EMBEDDING_MODEL_NAME})
    if st.button("Clear all uploads (session)"):
        st.session_state.uploads = []
        st.success("Cleared session uploads")

def render_exports_tab():
    st.header("Exports")
    conn = get_db_connection()
    upload_map = {u["db_id"]: u["filename"] for u in st.session_state.uploads if u.get("db_id")}
    if not upload_map:
        st.info("No uploads saved to DB yet.")
        return
    sel_dbid = st.selectbox("Select upload to export flashcards", list(upload_map.keys()), format_func=lambda x: upload_map.get(x))
    if st.button("Export Anki TSV"):
        res = anki_export_tsv(sel_dbid, conn)
        if not res:
            st.warning("No flashcards for selected upload.")
        else:
            fname, b = res
            st.download_button("Download Anki TSV", data=b, file_name=fname, mime="text/tab-separated-values")

# ----------------------------
# Main app
# ----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
    initialize_session_state()

    st.sidebar.title(APP_TITLE)
    st.sidebar.markdown(APP_SUBTITLE)
    st.sidebar.markdown("### Session")
    st.sidebar.write(f"Indexed uploads (session): {len(st.session_state.uploads)}")
    tabs = st.tabs(["Home", "Upload & Process", "Lessons", "MCQs", "Flashcards", "Chat Q&A", "Quizzes", "Exports", "Settings"])
    with tabs[0]:
        render_home()
    with tabs[1]:
        render_upload_tab()
    with tabs[2]:
        render_lessons_tab()
    with tabs[3]:
        render_mcq_tab()
    with tabs[4]:
        render_flashcards_tab()
    with tabs[5]:
        render_chat_tab()
    with tabs[6]:
        render_quizzes_tab()
    with tabs[7]:
        render_exports_tab()
    with tabs[8]:
        render_settings_tab()

if __name__ == "__main__":
    main()
# ------------------------------
# Main Content Extraction Pipeline
# ------------------------------
def parse_and_extract_content(filename, file_bytes):
    """
    Parse a file (pptx/pdf) and extract per-slide text, images and OCR.
    Returns a list of slide dicts: {"index": int, "text": str, "images": [bytes], "ocr_text": str}.
    This function is defensive: it logs and returns [] on any parse/ocr failures.
    """
    try:
        file_ext = os.path.splitext(filename)[1].lower()
    except Exception:
        logger.exception("parse_and_extract_content: unable to determine file extension for filename=%r", filename)
        return []

    logger.info("Starting content extraction for %r (ext=%s)", filename, file_ext)

    # Parse into raw slide dicts using existing parser helpers (should be defined elsewhere)
    try:
        if file_ext == ".pptx":
            raw_slides = _extract_from_pptx(file_bytes)
        elif file_ext == ".pdf":
            raw_slides = _extract_from_pdf(file_bytes)
        else:
            logger.warning("Unsupported file extension: %s", file_ext)
            return []
    except Exception as e:
        # show friendly streamlit message if available, but continue gracefully
        try:
            st.error(f"Failed to parse '{filename}': {str(e)}")
        except Exception:
            pass
        logger.exception("Error while parsing file %s", filename)
        return []

    # Acquire OCR reader (may return None if not available)
    try:
        ocr_reader = get_easyocr_reader(["en"])
    except Exception:
        ocr_reader = None
        logger.warning("OCR reader initialization failed or not available.")

    processed_slides = []
    total = max(1, len(raw_slides))
    # If Streamlit progress exists, create progress indicator; otherwise use a no-op fallback.
    progress = None
    try:
        progress = st.progress(0, text="Starting processing...")
    except Exception:
        progress = None

    for i, slide in enumerate(raw_slides):
        try:
            imgs = slide.get("images") or []
            # Perform OCR on images one-by-one; guard against very large images and broken bytes.
            ocr_texts = []
            for img_idx, img_bytes in enumerate(imgs):
                try:
                    if not img_bytes:
                        continue
                    # A single image OCR is isolated with try/except to avoid blocking entire pipeline.
                    t = ocr_image_bytes(img_bytes, ocr_reader)
                    if t:
                        ocr_texts.append(t)
                except Exception as e_img:
                    logger.debug("OCR failed for %s slide %s image %s: %s", filename, slide.get("index"), img_idx, e_img)
                    continue

            combined_ocr = "\n".join([t for t in ocr_texts if t]).strip()
            processed_slides.append({
                "index": int(slide.get("index", i)),
                "text": (slide.get("text") or "").strip(),
                "images": imgs,
                "ocr_text": combined_ocr,
            })

            # Update progress if available (use clamped value)
            if progress is not None:
                try:
                    progress.progress(min(1.0, float(i + 1) / float(total)), text=f"Processing slide {i+1}/{total}...")
                except Exception:
                    # older streamlit may not support text param
                    try:
                        progress.progress(min(1.0, float(i + 1) / float(total)))
                    except Exception:
                        pass

        except Exception as e_slide:
            # If one slide fails, continue processing remaining slides
            logger.exception("Failed processing slide index %s in %s: %s", slide.get("index"), filename, e_slide)
            continue

    logger.info("Extraction finished: file=%s slides=%d", filename, len(processed_slides))
    return processed_slides


# ------------------------------
# Text Processing Utilities
# ------------------------------
def chunk_text(text, max_chars=1000):
    """
    Robust chunking: preserve paragraph boundaries; if a paragraph is too long,
    split by sentences. Guarantees that each returned chunk <= max_chars (except if a single sentence > max_chars,
    which will be split naively).
    """
    if not text or not isinstance(text, str):
        return []

    # Normalize whitespace and unify newlines
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    # Split into paragraphs
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(p) <= max_chars:
            # try to append to current chunk
            if not current:
                current = p + "\n"
            else:
                if len(current) + len(p) + 1 <= max_chars:
                    current += p + "\n"
                else:
                    chunks.append(current.strip())
                    current = p + "\n"
        else:
            # Paragraph too long: split into sentences using a lightweight regex split (relies on `re` present)
            try:
                sentences = re.split(r'(?<=[\.\?\!])\s+', p)
            except Exception:
                # fallback: naive fixed-size splitting
                sentences = [p[i:i+max_chars] for i in range(0, len(p), max_chars)]

            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if len(s) > max_chars:
                    # as a last resort, hard split the long sentence
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

            # finish paragraph handling
            if current:
                chunks.append(current.strip())
                current = ""

    if current:
        chunks.append(current.strip())

    # final sanity: remove empty chunks and trim
    final = [c.strip() for c in chunks if c and c.strip()]
    return final


def extract_json_from_text(text):
    """
    Robust JSON extraction:
    - Finds the first '{' or '[' and attempts to parse the balanced JSON block by scanning for matching brackets.
    - Ignores markdown code fences and attempts to clean trailing commas.
    Returns a Python object (dict/list) or None.
    """
    if not text or not isinstance(text, str):
        return None

    # quick check for obvious inline JSON (fast path)
    simple = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    candidate = None
    if simple:
        # try to parse content inside code fence first
        candidate = simple.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # continue to general extraction
            candidate = None

    # find first brace or bracket
    start_pos = None
    start_char = None
    for i, ch in enumerate(text):
        if ch in ("{", "["):
            start_pos = i
            start_char = ch
            break
    if start_pos is None:
        # no JSON-like start
        return None

    # scan forward to find matching bracket, accounting for strings and escaped quotes
    stack = []
    opening = {"{": "}", "[": "]"}
    expected_close = opening.get(start_char)
    stack.append(expected_close)
    i = start_pos
    in_string = False
    escape = False
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
                    # mismatch
                    break
                expected = stack.pop()
                if ch != expected:
                    # mismatch
                    break
                if not stack:
                    # found full balanced JSON from start_pos to i
                    candidate = text[start_pos:i+1]
                    break
        i += 1

    if not candidate:
        # fallback: simple regex as last resort (may capture partial)
        m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if not m:
            return None
        candidate = m.group(0)

    # Try tolerant JSON parse: remove trailing commas before } or ]
    def clean_trailing_commas(s):
        s1 = re.sub(r',\s*(\}|])', r'\1', s)
        return s1

    candidate_clean = candidate
    attempts = 0
    while attempts < 3:
        try:
            return json.loads(candidate_clean)
        except Exception as e:
            candidate_clean = clean_trailing_commas(candidate_clean)
            attempts += 1
            continue

    # give up
    logger.debug("extract_json_from_text: failed to parse JSON candidate (first200): %s", candidate[:200].replace("\n", " "))
    return None


# ------------------------------
# LLM API Communication (OpenRouter)
# ------------------------------
# note: uses `requests`, `OPENROUTER_API_URL`, `DEFAULT_OPENROUTER_KEY`, `st`, and `logger` which must be in surrounding file.

def call_openrouter(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=1024, temperature=0.1, is_json=False):
    """
    Call OpenRouter-style chat completions API.
    Returns the assistant content (string).
    Raises APIError on missing API key or request/parsing error.
    This wrapper is defensive and logs raw responses for easier debugging.
    """
    api_key = None
    try:
        api_key = st.session_state.get("OPENROUTER_API_KEY") or DEFAULT_OPENROUTER_KEY
    except Exception:
        api_key = DEFAULT_OPENROUTER_KEY

    if not api_key:
        raise APIError("OpenRouter API key is not configured. Set OPENROUTER_API_KEY in session/env/secrets.")

    if "requests" not in globals() or requests is None:
        raise APIError("HTTP client 'requests' is not available in the environment.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    if is_json:
        # some routers accept a structured response_format; only add if supported by your API
        try:
            body["response_format"] = {"type": "json_object"}
        except Exception:
            pass

    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
    except Exception as e:
        logger.exception("Network exception when calling OpenRouter: %s", e)
        raise APIError("Network error when contacting LLM API: " + str(e))

    # status handling
    try:
        resp.raise_for_status()
    except Exception as e:
        # attempt to surface useful debugging info
        raw = None
        try:
            raw = resp.text
        except Exception:
            raw = "<no-body>"
        logger.error("OpenRouter returned status %s. Body:\n%s", resp.status_code, raw)
        raise APIError(f"LLM API returned HTTP {resp.status_code}: {raw[:1000]}")

    # parse JSON body
    try:
        data = resp.json()
    except Exception as e:
        logger.exception("Failed to parse JSON from OpenRouter response: %s", e)
        raise APIError("Failed to decode LLM response as JSON.")

    # try common response shapes (OpenRouter / OpenAI-style)
    content = None
    try:
        # choices -> [ { "message": {"content": "..." }} ]
        choices = data.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message") or {}
                content = msg.get("content") or first.get("text") or first.get("message", {}).get("content")
        # fallback: top-level 'output' or 'result' fields
        if not content:
            if "output" in data:
                content = data["output"]
            elif "result" in data:
                content = data["result"]
    except Exception:
        content = None

    if not content:
        # last resort: stringify the entire response for debugging
        logger.error("OpenRouter response did not contain expected fields. Full response: %s", json.dumps(data)[:2000])
        raise APIError("LLM response missing expected content.")

    # ensure string
    if not isinstance(content, str):
        try:
            content = json.dumps(content)
        except Exception:
            content = str(content)

    return content.strip()


# ------------------------------
# Content Generation Wrappers
# ------------------------------
PROMPT_LESSON_MULTILEVEL = """
You are a patient expert teacher. Your task is to explain a topic from a slide at three levels of difficulty.
Produce three sections clearly titled '## Beginner', '## Intermediate', and '## Advanced'.
For each section, provide:
1. A clear, concise explanation of the core concept.
2. One small, practical worked example.
3. 2-3 essential tips or key takeaways.
Finally, create a '## Quick Quiz' section with 3 multiple-choice questions (and provide the correct answers).
The tone should be encouraging and accessible.
"""

PROMPT_LESSON_DEEP = """
You are an expert university instructor creating a detailed lesson for a B.Tech 2nd-year Mechanical Engineering student from India.
The lesson must be structured into three clearly labeled sections: '## Foundations', '## Developing Concepts', and '## Advanced Application'.
For each section, you must include ALL of the following:
1.  Concept Explanation: Start from first principles and progressively increase complexity.
2.  Worked Examples: Provide two distinct examples: one numerical (with explicit steps and SI units) and one conceptual.
3.  Common Mistakes: List 2-3 common pitfalls or misunderstandings students have.
4.  Practice MCQs: Write 5 short multiple-choice questions with the correct answer clearly indicated.
5.  Flashcard Q&A: Generate 8 concise question/answer pairs suitable for flashcards.
"""

PROMPT_MCQ_JSON = """
You are an AI assistant that generates high-quality multiple-choice questions (MCQs) from a given text.
Your response MUST be a single, valid JSON array. Each object in the array represents one MCQ and must have these exact keys:
- "question": A string containing the question text.
- "options": An array of 4 strings representing the possible answers.
- "answer_index": A 0-based integer indicating the correct option.
Only return the JSON array — no explanation or extra text.
"""

PROMPT_FLASHCARDS_JSON = """
You are an AI assistant that extracts concise question-and-answer pairs from text, suitable for flashcards.
Your response MUST be a single, valid JSON array of objects. Each object must have these exact keys:
- "question": A string for the front of the card (max 80 chars).
- "answer": A string for the back of the card (max 200 chars).
Only return the JSON array — no explanation or extra text.
"""

def generate_multilevel_lesson(slide_text, related_texts=""):
    user_prompt = f"Based on this slide text and additional context, produce a multi-level lesson.\n\nSLIDE TEXT:\n{slide_text}\n\nRELATED TEXT:\n{related_texts}"
    try:
        return call_openrouter(PROMPT_LESSON_MULTILEVEL, user_prompt, max_tokens=1500, temperature=0.2)
    except APIError as e:
        logger.error("generate_multilevel_lesson: LLM error: %s", e)
        return f"LLM error: {e}"

def generate_deep_lesson(slide_text, related_texts=""):
    user_prompt = f"Deep-dive lesson required for B.Tech student.\n\nSLIDE:\n{slide_text}\n\nRELATED:\n{related_texts}"
    try:
        return call_openrouter(PROMPT_LESSON_DEEP, user_prompt, model="gpt-4o-mini", max_tokens=3000, temperature=0.1)
    except APIError as e:
        logger.error("generate_deep_lesson: LLM error: %s", e)
        return f"LLM error: {e}"

def generate_mcq_set_from_text(text, qcount=5):
    if not text or not isinstance(text, str):
        return []
    user_prompt = f"Create exactly {qcount} MCQs from the text below.\n\n{text}"
    try:
        resp = call_openrouter(PROMPT_MCQ_JSON, user_prompt, max_tokens=800, temperature=0.0, is_json=True)
        parsed = extract_json_from_text(resp)
        if isinstance(parsed, list):
            return parsed
        # If extract_json fails, try to parse resp directly
        try:
            return json.loads(resp)
        except Exception:
            logger.debug("generate_mcq_set_from_text: could not parse LLM response as JSON; returning []")
            return []
    except APIError as e:
        logger.warning("generate_mcq_set_from_text: %s", e)
        return []

def generate_flashcards_from_text(text, n=10):
    if not text or not isinstance(text, str):
        return []
    user_prompt = f"Extract up to {n} flashcards from the text below.\n\n{text}"
    try:
        resp = call_openrouter(PROMPT_FLASHCARDS_JSON, user_prompt, max_tokens=1000, temperature=0.0, is_json=True)
        parsed = extract_json_from_text(resp)
        if isinstance(parsed, list):
            return parsed
        try:
            return json.loads(resp)
        except Exception:
            logger.debug("generate_flashcards_from_text: could not parse LLM response as JSON; returning []")
            return []
    except APIError as e:
        logger.warning("generate_flashcards_from_text: %s", e)
        return []


# ------------------------------
# Spaced Repetition (SM-2 Algorithm)
# ------------------------------
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

# -----------------------------------------
# Export, TTS and Main Streamlit UI (robust)
# NOTE: No import statements added here.
# Assumes `st`, `logger`, `parse_and_extract_content`, `chunk_text`,
# `get_easyocr_reader`, `ocr_image_bytes`, `call_openrouter`,
# `generate_multilevel_lesson`, `generate_mcq_set_from_text`,
# `generate_flashcards_from_text`, `sm2_update_card`, `answer_question_with_rag`
# are defined elsewhere in your file. The code provides fallbacks if some modules are missing.
# -----------------------------------------

# ---------- Helpful runtime checks & small fallbacks ----------
# Check for sqlite3
_SQLITE_AVAILABLE = ("sqlite3" in globals() and globals().get("sqlite3") is not None)
_sqlite_mod = globals().get("sqlite3") if _SQLITE_AVAILABLE else None

# Check for gTTS
_GTTS_AVAILABLE = ("gTTS" in globals() and globals().get("gTTS") is not None)
_gtts_cls = globals().get("gTTS") if _GTTS_AVAILABLE else None

# Check for numpy for safe shape checks in UI
_NP_AVAILABLE = ("np" in globals() and globals().get("np") is not None)
_np = globals().get("np") if _NP_AVAILABLE else None

# Initialize an in-session fallback DB if sqlite3 not available
if not _SQLITE_AVAILABLE:
    if "flashcards_db" not in st.session_state:
        # flashcards_db is a dict: upload_id -> list of flashcard dicts
        st.session_state["flashcards_db"] = {}
    if "quizzes_db" not in st.session_state:
        st.session_state["quizzes_db"] = {}

# ---------- Export: Anki TSV ----------
def anki_export_tsv(upload_id, conn=None):
    """
    Export flashcards for upload_id as TSV bytes.
    If sqlite3 is available and conn is provided it'll query DB, otherwise uses in-session fallback.
    Returns (filename, bytes) or None on failure / no cards.
    """
    try:
        rows = []
        if _SQLITE_AVAILABLE and conn is not None:
            cur = conn.execute("SELECT question, answer FROM flashcards WHERE upload_id = ?", (upload_id,))
            rows = cur.fetchall()
        else:
            # in-session fallback
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

# ---------- Text-to-Speech (TTS) ----------
def text_to_speech(text, lang="en"):
    """
    Generate audio bytes for the given text.
    Uses gTTS if available. If not, returns a short silent WAV fallback.
    Returns (filename, bytes).
    """
    if not text or not str(text).strip():
        raise ValueError("No text provided for TTS.")
    txt = str(text).strip()
    # Try gTTS first
    if _GTTS_AVAILABLE and _gtts_cls is not None:
        try:
            mp3_fp = io.BytesIO()
            tts_obj = _gtts_cls(text=txt, lang=lang, slow=False)
            tts_obj.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return "lesson_audio.mp3", mp3_fp.getvalue()
        except Exception as e:
            logger.warning("gTTS failed: %s — falling back to WAV silent audio", e)

    # Fallback: produce a short silent WAV (1 second) so download still works
    # Simple WAV header with 1-second silence, 16-bit PCM, mono, 22050 Hz
    try:
        sample_rate = 22050
        duration_seconds = max(1, min(30, int(len(txt) / 50)))  # estimate duration by text length (approx.)
        num_samples = sample_rate * duration_seconds
        byte_rate = sample_rate * 2  # 16-bit mono => 2 bytes/sample
        data_size = num_samples * 2
        # RIFF header
        header = b"RIFF" + (36 + data_size).to_bytes(4, "little") + b"WAVE"
        # fmt subchunk
        header += b"fmt " + (16).to_bytes(4, "little")  # Subchunk1Size
        header += (1).to_bytes(2, "little")  # PCM format
        header += (1).to_bytes(2, "little")  # channels
        header += (sample_rate).to_bytes(4, "little")  # sample rate
        header += (byte_rate).to_bytes(4, "little")  # byte rate
        header += (2).to_bytes(2, "little")  # block align
        header += (16).to_bytes(2, "little")  # bits per sample
        # data subchunk
        header += b"data" + (data_size).to_bytes(4, "little")
        silence = b"\x00" * data_size
        wav_bytes = header + silence
        return "lesson_audio.wav", wav_bytes
    except Exception as e:
        logger.exception("Failed to make fallback WAV for TTS: %s", e)
        raise RuntimeError("Failed to generate audio.") from e

# ---------- Helper: Process new upload ----------
def process_new_upload(uploaded_file):
    """
    Create a new upload dictionary from a Streamlit uploaded file-like object.
    This runs parsing and chunking but does not build embeddings unless explicitly requested.
    """
    filename = getattr(uploaded_file, "name", "uploaded_file")
    try:
        file_bytes = uploaded_file.getvalue()
    except Exception:
        try:
            # Some file objects expose read()
            file_bytes = uploaded_file.read()
        except Exception:
            raise RuntimeError("Could not read uploaded file bytes.")

    upload = {
        "filename": filename,
        "file_bytes": file_bytes,
        "status_msg": "Parsed (not indexed)",
        "slides_data": [],
        "chunks": [],
        "full_text": "",
        "slide_count": 0,
        "embeddings": None,
        "index": None,
        "index_built": False,
        "db_id": None,
    }

    try:
        slides = parse_and_extract_content(filename, file_bytes)
        upload["slides_data"] = slides
        upload["slide_count"] = len(slides)
        # Aggregate text
        parts = []
        for s in slides:
            txt = (s.get("text") or "").strip()
            ocr = (s.get("ocr_text") or "").strip()
            combined = "\n".join([p for p in (txt, ocr) if p]).strip()
            if combined:
                parts.append(combined)
        full = "\n\n".join(parts).strip()
        upload["full_text"] = full
        upload["chunks"] = chunk_text(full, max_chars=1200) or ([full] if full else [])
        upload["status_msg"] = f"Parsed — {upload['slide_count']} slides, {len(upload['chunks'])} chunks"
    except Exception as e:
        logger.exception("process_new_upload failed for %s: %s", filename, e)
        upload["status_msg"] = f"Error parsing file: {e}"

    # Optionally persist minimal metadata to sqlite if available
    if _SQLITE_AVAILABLE:
        try:
            conn = get_db_connection()
            with conn:
                cur = conn.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)",
                                   (upload["filename"], int(time.time()), json.dumps({"slide_count": upload["slide_count"]})))
                upload["db_id"] = cur.lastrowid
        except Exception as e:
            logger.warning("Failed to save upload record to DB: %s", e)
            upload["db_id"] = None
    else:
        # store a minimal record in session for fallback
        if "uploads_meta" not in st.session_state:
            st.session_state["uploads_meta"] = {}
        st.session_state["uploads_meta"][upload["filename"]] = {"slide_count": upload["slide_count"]}

    return upload

# ---------- Helper: Build vector index (attempt) ----------
def build_vector_index(upload):
    """
    Build embeddings & vector index if embedding model is available.
    This function will try to call get_embedding_model() and embed_texts(); if unavailable, it gracefully marks index_built=False.
    """
    if not upload or not isinstance(upload, dict):
        return
    if upload.get("index_built"):
        return

    # check if embedding model function exists
    try:
        model = get_embedding_model()
    except Exception as e:
        upload["index_built"] = False
        upload["status_msg"] = f"Embeddings unavailable: {e}"
        logger.info("Embedding model not available: %s", e)
        return

    try:
        chunks = upload.get("chunks") or []
        if not chunks:
            upload["status_msg"] = "No text to embed"
            upload["index_built"] = False
            return
        emb = embed_texts(chunks, model)
        upload["embeddings"] = emb
        idx = VectorIndex(emb, chunks)
        upload["index"] = idx
        upload["index_built"] = True
        upload["status_msg"] = f"Index built ({len(chunks)} chunks)"
    except Exception as e:
        logger.exception("Failed to build vector index: %s", e)
        upload["index_built"] = False
        upload["status_msg"] = f"Index build failed: {e}"

# ---------- Flashcard DB helpers ----------
def add_flashcards_to_db(upload, cards):
    """
    Save generated flashcards either into sqlite3 (if available) or into session fallback.
    `cards` is expected to be a list of dicts with keys 'question' and 'answer'.
    """
    if not cards:
        return 0
    saved = 0
    if _SQLITE_AVAILABLE:
        try:
            conn = get_db_connection()
            with conn:
                for obj in cards:
                    qtext = (obj.get("question") or "").strip()[:2000]
                    ans = (obj.get("answer") or "").strip()[:2000]
                    conn.execute("INSERT INTO flashcards (upload_id, question, answer, next_review) VALUES (?, ?, ?, ?)",
                                 (upload.get("db_id"), qtext, ans, int(time.time())))
                    saved += 1
            return saved
        except Exception as e:
            logger.exception("Saving flashcards to sqlite failed: %s", e)
            # fall through to session fallback

    # session fallback
    try:
        db = st.session_state.setdefault("flashcards_db", {})
        uid = upload.get("db_id") or upload.get("filename")
        arr = db.setdefault(uid, [])
        for obj in cards:
            arr.append({
                "id": f"{uid}-{len(arr)+1}",
                "upload_id": uid,
                "question": (obj.get("question") or "").strip(),
                "answer": (obj.get("answer") or "").strip(),
                "easiness": 2.5,
                "interval": 1,
                "repetitions": 0,
                "next_review": int(time.time())
            })
            saved += 1
        st.session_state["flashcards_db"] = db
    except Exception as e:
        logger.exception("Failed to save flashcards to session fallback: %s", e)
    return saved

def get_due_flashcards(limit=50):
    """
    Return a list of due flashcards (from sqlite if available, otherwise session fallback).
    Each card is a dict with keys: id, question, answer, easiness, interval, repetitions, next_review.
    """
    now = int(time.time())
    rows = []
    if _SQLITE_AVAILABLE:
        try:
            conn = get_db_connection()
            cur = conn.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards WHERE IFNULL(next_review,0) <= ? ORDER BY next_review ASC LIMIT ?", (now, limit))
            for r in cur.fetchall():
                rows.append({
                    "id": r[0],
                    "question": r[1],
                    "answer": r[2],
                    "easiness": r[3] or 2.5,
                    "interval": r[4] or 1,
                    "repetitions": r[5] or 0,
                    "next_review": r[6] or now
                })
            return rows
        except Exception as e:
            logger.exception("get_due_flashcards (sqlite) failed: %s", e)
            # fallback to session

    # session fallback
    try:
        db = st.session_state.get("flashcards_db", {})
        for uid, lst in db.items():
            for c in lst:
                if int(c.get("next_review", 0)) <= now:
                    rows.append(c)
                    if len(rows) >= limit:
                        return rows
    except Exception as e:
        logger.exception("get_due_flashcards (session) failed: %s", e)
    return rows

def update_flashcard_review(card, quality):
    """
    Update flashcard metadata using SM-2 algorithm. Writes to DB if available, otherwise to session fallback.
    `card` can be a dict with 'id' and 'upload_id' fields.
    `quality` is an integer 0-5.
    """
    try:
        q = int(quality)
    except Exception:
        q = 0
    q = max(0, min(5, q))
    # extract current values (favor explicit fields)
    eas = float(card.get("easiness", 2.5))
    ivl = int(card.get("interval", 1))
    reps = int(card.get("repetitions", 0))
    new_eas, new_ivl, new_reps, new_next = sm2_update_card(eas, ivl, reps, q)

    if _SQLITE_AVAILABLE:
        try:
            conn = get_db_connection()
            with conn:
                conn.execute("UPDATE flashcards SET easiness=?, interval=?, repetitions=?, next_review=? WHERE id=?",
                             (float(new_eas), int(new_ivl), int(new_reps), int(new_next), card.get("id")))
            return True
        except Exception as e:
            logger.exception("update_flashcard_review (sqlite) failed: %s", e)
            # fallback to session

    # session fallback: find the card and update
    try:
        db = st.session_state.setdefault("flashcards_db", {})
        uid = card.get("upload_id")
        if uid is None:
            # try to derive
            uid = card.get("id", "").split("-")[0] if card.get("id") else None
        lst = db.get(uid, [])
        for c in lst:
            if c.get("id") == card.get("id"):
                c["easiness"] = float(new_eas)
                c["interval"] = int(new_ivl)
                c["repetitions"] = int(new_reps)
                c["next_review"] = int(new_next)
                st.session_state["flashcards_db"] = db
                return True
    except Exception as e:
        logger.exception("update_flashcard_review (session) failed: %s", e)
    return False

# ---------- UI helper: active upload ----------
def get_active_upload():
    idx = st.session_state.get("active_upload_idx")
    uploads = st.session_state.get("uploads", [])
    if idx is None:
        return None
    if isinstance(idx, int) and 0 <= idx < len(uploads):
        return uploads[idx]
    # allow active selection by filename
    for u in uploads:
        if u.get("filename") == idx:
            return u
    return None

# ---------- Flashcard practice UI renderer ----------
def render_flashcard_practice_ui():
    """User-facing practice UI that works with both sqlite and session fallback."""
    cards = st.session_state.get("due_cards", [])
    if not cards:
        st.info("No due cards to practice right now.")
        return
    idx = st.session_state.get("current_card_idx", 0)
    if idx >= len(cards):
        st.success("✨ You finished all due cards in this session.")
        st.session_state["due_cards"] = []
        st.session_state["current_card_idx"] = 0
        return
    card = cards[idx]
    st.markdown(f"##### Card {idx + 1} of {len(cards)}")
    st.write(card.get("question"))
    if st.button("Show Answer", key=f"show_{card.get('id')}"):
        st.write("**Answer:**")
        st.write(card.get("answer"))
        st.markdown("How well did you recall this?")

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Again", key=f"again_{card.get('id')}"):
        update_flashcard_review(card, 1)
        st.session_state["current_card_idx"] = idx + 1
        st.experimental_rerun()
    if col2.button("Hard", key=f"hard_{card.get('id')}"):
        update_flashcard_review(card, 3)
        st.session_state["current_card_idx"] = idx + 1
        st.experimental_rerun()
    if col3.button("Good", key=f"good_{card.get('id')}"):
        update_flashcard_review(card, 4)
        st.session_state["current_card_idx"] = idx + 1
        st.experimental_rerun()
    if col4.button("Easy", key=f"easy_{card.get('id')}"):
        update_flashcard_review(card, 5)
        st.session_state["current_card_idx"] = idx + 1
        st.experimental_rerun()

# ---------- Render header ----------
def render_header():
    try:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
              <div style="font-size:2.4rem">🎓</div>
              <div>
                <div style="font-weight:700;font-size:1.6rem">{APP_TITLE}</div>
                <div style="color:#9AA6B2;font-size:0.95rem">{APP_SUBTITLE}</div>
              </div>
            </div>
            """, unsafe_allow_html=True
        )
    except Exception:
        # best-effort; if st.markdown not available, ignore
        pass

# ---------- CSS (beautiful, minimal, professional) ----------
APP_CSS = """
:root{
  --bg:#071025;
  --card:#071424;
  --muted:#9AA6B2;
  --accent:#4D7CFE;
  --accent-2:#2AB7A9;
  --glass: rgba(255,255,255,0.03);
  --radius:14px;
}

body, .stApp, .main {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  background: linear-gradient(180deg,#051022 0%, #061428 100%);
  color: #E6F0FA;
}

.block-container {
  padding: 28px 32px !important;
}

/* Cards */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: var(--radius);
  padding: 18px;
  margin-bottom: 16px;
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: 0 8px 24px rgba(2,6,12,0.5);
}

/* Headings */
h1, h2, h3, h4 {
  color: #F1F5F9;
  margin: 6px 0;
}

/* Buttons */
.stButton>button {
  border-radius: 10px !important;
  padding: 8px 14px !important;
  background: linear-gradient(90deg,var(--accent), #5b21b6) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 6px 18px rgba(77,124,254,0.14);
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(10,14,22,0.9), rgba(8,12,18,0.6));
  border-right: 1px solid rgba(255,255,255,0.03);
}

/* Muted text */
.small-muted { color: var(--muted); font-size: 0.95rem; }

/* File preview */
.preview-slide {
  background: rgba(255,255,255,0.02);
  border-radius: 10px;
  padding: 10px;
  margin-bottom: 8px;
}

/* Inputs */
textarea, input, .stTextInput, .stTextArea {
  border-radius: 10px !important;
}
"""

# ---------- Main application ----------
def main():
    # Page config and state init
    try:
        st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="auto")
    except Exception:
        pass

    if "OPENROUTER_API_KEY" not in st.session_state:
        st.session_state.OPENROUTER_API_KEY = DEFAULT_OPENROUTER_KEY
    if "uploads" not in st.session_state:
        st.session_state.uploads = []
    if "active_upload_idx" not in st.session_state:
        st.session_state.active_upload_idx = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "due_cards" not in st.session_state:
        st.session_state.due_cards = []
    if "current_card_idx" not in st.session_state:
        st.session_state.current_card_idx = 0

    # Styling
    try:
        st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # Header
    render_header()

    # Tabs
    tabs = st.tabs(["🏠 Home", "📤 Upload", "📚 Lessons", "💬 Chat Q&A", "📝 Quizzes", "🃏 Flashcards", "⚙️ Settings"])
    with tabs[0]:
        # Home
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Welcome to SlideTutor — an intelligent study companion.")
        st.markdown("<p class='small-muted'>Upload slides, build a search index, generate lessons & quizzes, practice flashcards, and optionally export to Anki.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        # Upload
        render_upload_tab()

    with tabs[2]:
        # Lessons
        render_lessons_tab()

    with tabs[3]:
        # Chat Q&A
        render_chat_tab()

    with tabs[4]:
        # Quizzes
        render_quizzes_tab()

    with tabs[5]:
        # Flashcards
        render_flashcards_tab()

    with tabs[6]:
        # Settings
        render_settings_tab()

# ---------- UI parts referenced by main() ----------
def render_upload_tab():
    st.markdown("### Upload & prepare documents")
    st.markdown("<p class='small-muted'>Drop PPTX or PDF files. Each will be parsed into slides/pages and chunked for indexing.</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PPTX or PDF files", type=["pptx", "pdf"], accept_multiple_files=True)
    if uploaded:
        exist = {u["filename"] for u in st.session_state.uploads}
        for f in uploaded:
            if f.name in exist:
                st.warning(f"File '{f.name}' already uploaded; skipped.")
                continue
            try:
                with st.spinner(f"Parsing {f.name}..."):
                    new = process_new_upload(f)
                    st.session_state.uploads.append(new)
                    st.success(f"Uploaded: {f.name}")
            except Exception as e:
                st.error(f"Failed to process {f.name}: {e}")

    if not st.session_state.uploads:
        st.info("No uploads yet.")
        return

    st.markdown("---")
    st.markdown("#### Uploaded documents")
    for i, up in enumerate(st.session_state.uploads):
        with st.expander(f"{up['filename']} — {up.get('status_msg','Ready')}", expanded=False):
            c1, c2, c3 = st.columns([2,1,1])
            c1.write(f"Slides/Pages: {up.get('slide_count', 0)}")
            c1.write(f"Chunks: {len(up.get('chunks', []))}")
            if up.get("index_built"):
                try:
                    emb = up.get("embeddings")
                    shape_desc = ""
                    if _NP_AVAILABLE and emb is not None:
                        try:
                            shape_desc = f"{_np.asarray(emb).shape[0]} vectors × {_np.asarray(emb).shape[1]} dims"
                        except Exception:
                            shape_desc = "built (shape unknown)"
                    else:
                        shape_desc = "built"
                    c1.write(f"Index: {shape_desc}")
                except Exception:
                    c1.write("Index: built")
            if c2.button("Build Index", key=f"build_{i}", disabled=up.get("index_built", False)):
                with st.spinner("Building index..."):
                    build_vector_index(up)
                st.experimental_rerun()
            if c3.button("Delete", key=f"del_{i}"):
                st.session_state.uploads.pop(i)
                st.experimental_rerun()

def render_lessons_tab():
    uploads = [u for u in st.session_state.uploads if u.get("index_built")]
    if not uploads:
        st.warning("No processed uploads with indexes. Upload & build index first.")
        return
    opts = {u["filename"]: i for i, u in enumerate(uploads)}
    sel = st.selectbox("Select a document", options=list(opts.keys()))
    if not sel:
        return
    up = uploads[opts[sel]]
    if st.button("Generate Multi-Level Lesson"):
        with st.spinner("Generating lesson..."):
            res = generate_multilevel_lesson(up.get("full_text", ""))
            st.markdown(res)

def render_chat_tab():
    uploads = [u for u in st.session_state.uploads if u.get("index_built")]
    if not uploads:
        st.warning("Build an index for at least one document to use Chat Q&A.")
        return
    st.markdown("Ask questions about your documents. Answers are produced by the LLM using retrieved context.")
    for msg in st.session_state.get("chat_history", []):
        role = msg.get("role")
        content = msg.get("content")
        try:
            with st.chat_message(role):
                st.markdown(content)
        except Exception:
            st.write(f"{role.upper()}: {content}")
    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Searching documents and calling LLM..."):
            try:
                answer = answer_question_with_rag(prompt, uploads)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                try:
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                except Exception:
                    st.write("Assistant:", answer)
            except APIError as e:
                st.error(f"LLM error: {e}")

def render_quizzes_tab():
    upload = get_active_upload()
    if not upload or not upload.get("index_built"):
        st.warning("Select a processed document (index built) from Lessons tab.")
        return
    st.markdown(f"Generate MCQs for: **{upload['filename']}**")
    qcount = st.slider("Questions", 3, 15, 5)
    if st.button("Generate MCQs"):
        with st.spinner("Generating MCQs..."):
            mcqs = generate_mcq_set_from_text(upload.get("full_text",""), qcount)
            if not mcqs:
                st.error("No MCQs generated.")
            else:
                for i, m in enumerate(mcqs):
                    st.markdown(f"**Q{i+1}.** {m.get('question')}")
                    for j,opt in enumerate(m.get("options",[])):
                        st.write(f"- {opt}")

def render_flashcards_tab():
    uploads = st.session_state.get("uploads", [])
    if not uploads:
        st.warning("Upload files first.")
        return
    st.markdown("Generate & practice flashcards")
    options = {u["filename"]: i for i,u in enumerate(uploads)}
    sel = st.selectbox("Pick document for cards", list(options.keys()))
    if sel:
        up = uploads[options[sel]]
        if st.button("Generate Flashcards"):
            with st.spinner("Generating..."):
                cards = generate_flashcards_from_text(up.get("full_text",""), n=20)
                saved = add_flashcards_to_db(up, cards)
                st.success(f"Saved {saved} flashcards.")
    if st.button("Load due cards for practice"):
        st.session_state.due_cards = get_due_flashcards()
        st.session_state.current_card_idx = 0
        st.experimental_rerun()
    if st.session_state.get("due_cards"):
        render_flashcard_practice_ui()

def render_settings_tab():
    st.markdown("### Settings & Exports")
    api_key = st.text_input("OpenRouter API Key (session-only)", value=st.session_state.OPENROUTER_API_KEY or "", type="password")
    if api_key != st.session_state.OPENROUTER_API_KEY:
        st.session_state.OPENROUTER_API_KEY = api_key
        st.success("API key updated in session.")
    # Export utilities
    st.markdown("#### Exports")
    # show uploads that have db_id or fallback key
    upload_map = {}
    for u in st.session_state.uploads:
        key = u.get("db_id") or u.get("filename")
        upload_map[key] = u
    if upload_map:
        sel_key = st.selectbox("Select upload to export flashcards", options=list(upload_map.keys()), format_func=lambda k: upload_map[k]["filename"])
        if st.button("Export Anki TSV"):
            conn = get_db_connection() if _SQLITE_AVAILABLE else None
            res = anki_export_tsv(sel_key, conn)
            if not res:
                st.warning("No flashcards to export.")
            else:
                fname, b = res
                st.download_button("Download Anki TSV", data=b, file_name=fname, mime="text/tab-separated-values")
    else:
        st.info("No uploads available for export.")

# ---------- Run ----------
# call main() only if this file is run as a script; if integrating keep as callable function
try:
    if __name__ == "__main__":
        main()
except Exception:
    # In case Streamlit environment doesn't like __name__ check, call main anyway
    try:
        main()
    except Exception as e:
        logger.exception("Failed to start main UI: %s", e)

# ------------------------------
# Robust Core Logic & Backend Functions (single self-contained block)
# NOTE: This block intentionally does NOT add any `import` statements.
# It defensively checks for optional globals (sqlite3, np, requests, etc.)
# and falls back to safe, dependency-free implementations when needed.
# It also defines concise, robust helper functions so the rest of your app
# can rely on consistent behavior.
# ------------------------------

# ---------- Configuration & CSS ----------
# (You can paste this APP_CSS into your UI rendering code)
APP_CSS = """
:root{
  --bg:#071025; --card:#071424; --muted:#9AA6B2; --accent:#4D7CFE; --accent-2:#2AB7A9; --radius:14px;
}
body, .stApp { background: linear-gradient(180deg,#051022 0%, #061428 100%); color: #E6F0FA; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial; }
.block-container { padding: 28px 32px !important; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius: var(--radius); padding: 18px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 24px rgba(2,6,12,0.5); }
.small-muted { color: var(--muted); font-size: 0.95rem; }
.stButton>button { border-radius: 10px !important; padding: 8px 14px !important; background: linear-gradient(90deg,var(--accent), #5b21b6) !important; color: #fff !important; border: none !important; box-shadow: 0 6px 18px rgba(77,124,254,0.14); }
.preview-slide { background: rgba(255,255,255,0.02); border-radius: 10px; padding: 10px; margin-bottom: 8px; }
"""

# ---------- Environment feature detection ----------
_HAS_SQLITE = ("sqlite3" in globals() and globals().get("sqlite3") is not None)
_sqlite = globals().get("sqlite3") if _HAS_SQLITE else None

_HAS_NUMPY = ("np" in globals() and globals().get("np") is not None)
_np = globals().get("np") if _HAS_NUMPY else None

# Optional high-level functions & classes that may be defined elsewhere in your app.
_get_embedding_model = globals().get("get_embedding_model")
_embed_texts = globals().get("embed_texts")
_VectorIndex = globals().get("VectorIndex")
_parse_and_extract_content = globals().get("parse_and_extract_content")
_chunk_text = globals().get("chunk_text")
_call_openrouter = globals().get("call_openrouter")
_generate_multilevel_lesson = globals().get("generate_multilevel_lesson")
_generate_mcq_set_from_text = globals().get("generate_mcq_set_from_text")
_generate_flashcards_from_text = globals().get("generate_flashcards_from_text")
_sm2_update_card = globals().get("sm2_update_card")

# Basic defensive fallbacks (if the above are not defined, provide simple alternatives)
def _ensure_func(fallback_name, fallback):
    if globals().get(fallback_name) is None:
        globals()[fallback_name] = fallback
        return fallback
    return globals().get(fallback_name)

# Minimal fallback for chunk_text (if not present)
if _chunk_text is None:
    def _simple_chunk_text(text, max_chars=1000):
        if not text:
            return []
        text = text.replace("\r\n", "\n").strip()
        if len(text) <= max_chars:
            return [text]
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i+max_chars])
            i += max_chars
        return chunks
    _chunk_text = _ensure_func("chunk_text", _simple_chunk_text)
else:
    _chunk_text = globals()["chunk_text"]

# Minimal fallback for parse_and_extract_content
if _parse_and_extract_content is None:
    def _simple_parse_and_extract_content(filename, file_bytes):
        # Best-effort parser: returns a single "slide" with plain text extracted if bytes look like text.
        text = ""
        try:
            # try decode as utf-8 (works for plain .txt disguised as pdf/pptx in tests)
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        return [{"index": 0, "text": (text[:3000] or ""), "images": [], "ocr_text": ""}]
    _parse_and_extract_content = _ensure_func("parse_and_extract_content", _simple_parse_and_extract_content)
else:
    _parse_and_extract_content = globals()["parse_and_extract_content"]

# Minimal fallback for embedding model functions and VectorIndex:
if _embed_texts is None or _get_embedding_model is None or _VectorIndex is None:
    # Provide safe no-op embeddings and a simple vector index using normalized bag-of-words counts.
    def _fallback_get_embedding_model(*args, **kwargs):
        raise RuntimeError("Embedding model not available. Install sentence-transformers and define get_embedding_model to enable semantic search.")
    def _fallback_embed_texts(texts, model=None):
        # Create deterministic simple numeric "embeddings": length and character counts of lowercased word tokens
        if not texts:
            # default dimension 32
            return _np.zeros((0, 32), dtype=_np.float32) if _HAS_NUMPY else []
        dims = 32
        rows = []
        for t in texts:
            s = (t or "").lower()
            tokens = [w for w in re.findall(r"\w+", s)]
            vec = [len(tokens), sum(len(w) for w in tokens)]
            # add some hashed token counts to fill dims
            for i in range(dims - 2):
                h = 0
                if tokens:
                    h = sum(ord(ch) for ch in tokens[i % len(tokens)]) % 100
                vec.append(h)
            if _HAS_NUMPY:
                rows.append(_np.array(vec, dtype=_np.float32))
            else:
                rows.append(vec)
        if _HAS_NUMPY:
            return _np.vstack(rows)
        return rows
    class _FallbackVectorIndex:
        def __init__(self, embeddings, texts):
            self.texts = list(texts or [])
            if _HAS_NUMPY and hasattr(embeddings, "shape"):
                self.emb = _np.asarray(embeddings, dtype=_np.float32)
            else:
                # convert to numpy if available
                try:
                    import math  # only used if numpy absent; this import MAY fail if not allowed - avoid import
                except Exception:
                    pass
                if _HAS_NUMPY:
                    self.emb = _np.asarray(embeddings, dtype=_np.float32)
                else:
                    self.emb = embeddings
            self.dimension = self.emb.shape[1] if _HAS_NUMPY and getattr(self.emb, "size", None) else 0
        def search(self, query_embeddings, k):
            # simple dot-product using numpy if available, otherwise simple length-based ranking
            if not query_embeddings:
                return [], []
            q = query_embeddings
            if _HAS_NUMPY:
                qn = q / (_np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
                embn = self.emb / (_np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9)
                sims = qn @ embn.T
                idxs = _np.argsort(-sims, axis=1)[:, :k]
                # convert to distances: 1 - sim
                dists = 1.0 - _np.take_along_axis(sims, idxs, axis=1)
                return dists.astype(_np.float32), idxs.astype(_np.int64)
            else:
                # fallback ranking by string match count
                qstr = str(q)[-100:]
                scores = []
                for i, t in enumerate(self.texts):
                    common = len(set(re.findall(r"\w+", qstr)) & set(re.findall(r"\w+", (t or "")[:500].lower())))
                    scores.append((common, i))
                scores.sort(reverse=True)
                idxs = [i for _, i in scores[:k]]
                dists = [[0.0 for _ in idxs]]
                return dists, [idxs]
    # Expose fallbacks
    _get_embedding_model = _ensure_func("get_embedding_model", _fallback_get_embedding_model)
    _embed_texts = _ensure_func("embed_texts", _fallback_embed_texts)
    _VectorIndex = _ensure_func("VectorIndex", _FallbackVectorIndex)
else:
    _get_embedding_model = globals()["get_embedding_model"]
    _embed_texts = globals()["embed_texts"]
    _VectorIndex = globals()["VectorIndex"]

# Minimal fallback for LLM call
if _call_openrouter is None:
    def _fallback_call_openrouter(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=1024, temperature=0.1, is_json=False):
        # Do not attempt network calls. Provide a friendly placeholder.
        logger.warning("call_openrouter is not defined. Returning a synthetic placeholder response.")
        return "LLM not configured. Install/configure OpenRouter and define call_openrouter() to enable generation."
    _call_openrouter = _ensure_func("call_openrouter", _fallback_call_openrouter)
else:
    _call_openrouter = globals()["call_openrouter"]

# Ensure sm2_update_card fallback
if _sm2_update_card is None:
    def _fallback_sm2_update_card(easiness=2.5, interval=1, repetitions=0, quality=0):
        q = max(0, min(5, int(quality)))
        if q < 3:
            repetitions = 0
            interval = 1
        else:
            repetitions += 1
            interval = 1 if repetitions == 1 else (6 if repetitions == 2 else max(1, round(interval * easiness)))
        easiness = max(1.3, float(easiness) + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
        next_review = int(time.time()) + interval * 86400
        return easiness, interval, repetitions, next_review
    _sm2_update_card = _ensure_func("sm2_update_card", _fallback_sm2_update_card)
else:
    _sm2_update_card = globals()["sm2_update_card"]

# ---------- Database helpers ----------
def get_db_connection_safe(path=None):
    """
    Returns a sqlite3 connection if sqlite3 is available and DB path is set,
    otherwise returns None and the application should use session-state fallbacks.
    """
    try:
        if not _HAS_SQLITE:
            return None
        db_path = path or globals().get("DB_PATH") or "slidetutor.sqlite3"
        conn = _sqlite.connect(db_path, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON;")
        # create tables if missing
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                uploaded_at INTEGER NOT NULL,
                meta TEXT
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flashcards (
                id INTEGER PRIMARY KEY,
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
                id INTEGER PRIMARY KEY,
                upload_id INTEGER,
                question TEXT NOT NULL,
                options TEXT NOT NULL,
                correct_index INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
            );
        """)
        conn.commit()
        return conn
    except Exception as e:
        logger.exception("get_db_connection_safe failed: %s", e)
        return None

# ---------- Utility helpers ----------
def _safe_read_file_bytes(uploaded_file):
    """Robustly read bytes from a Streamlit UploadedFile or file-like object."""
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

def get_upload_db_id(upload):
    """Resolve or create a db_id for an upload; use sqlite3 if possible, otherwise return filename key."""
    if not isinstance(upload, dict):
        return None
    if "db_id" in upload and upload["db_id"]:
        return upload["db_id"]
    # attempt to persist minimal upload record
    conn = get_db_connection_safe()
    if conn is not None:
        try:
            with conn:
                cur = conn.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)",
                                   (upload.get("filename"), int(time.time()), json.dumps({"slide_count": upload.get("slide_count", 0)})))
                dbid = cur.lastrowid
                upload["db_id"] = dbid
                return dbid
        except Exception as e:
            logger.warning("get_upload_db_id: failed to insert upload record: %s", e)
            # fallback to filename
    # fallback key: filename
    key = upload.get("filename") or f"upload_{int(time.time())}"
    upload["db_id"] = key
    return key

# ---------- Core backend functions (robust) ----------

def process_new_upload_safe(uploaded_file):
    """
    Read, parse, chunk and persist metadata for a new uploaded file.
    Returns a robust upload dict. This function does not build embeddings by default.
    """
    try:
        file_bytes = _safe_read_file_bytes(uploaded_file)
        filename = getattr(uploaded_file, "name", None) or getattr(uploaded_file, "filename", None) or f"upload_{int(time.time())}"
    except Exception as e:
        logger.exception("process_new_upload_safe: failed to read uploaded file: %s", e)
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
        slides = _parse_and_extract_content(filename, file_bytes) or []
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
        upload["chunks"] = _chunk_text(upload["full_text"], max_chars=1200) or ([upload["full_text"]] if upload["full_text"] else [])
        upload["status_msg"] = f"Parsed: {upload['slide_count']} slides, {len(upload['chunks'])} chunks"
        upload["processed"] = True
    except Exception as e:
        logger.exception("process_new_upload_safe: parsing failed for %s: %s", filename, e)
        upload["status_msg"] = f"Parsing error: {e}"
        upload["processed"] = False

    # persist metadata to sqlite if available (but continue if not)
    try:
        dbid = get_upload_db_id(upload)
        upload["db_id"] = dbid
    except Exception as e:
        logger.debug("process_new_upload_safe: persistence skipped: %s", e)

    return upload

def build_vector_index_safe(upload, force=False):
    """
    Builds an embedding matrix and a vector index for the upload.
    If embeddings or index creation fails, index_built will remain False and status_msg updated.
    """
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

    # Attempt the real embedding pipeline
    try:
        model = _get_embedding_model()
    except Exception as e:
        upload["index_built"] = False
        upload["status_msg"] = f"Embeddings unavailable: {e}"
        logger.info("build_vector_index_safe: embedding model unavailable: %s", e)
        return upload

    try:
        embeddings = _embed_texts(chunks, model)
        upload["embeddings"] = embeddings
        idx = _VectorIndex(embeddings, chunks)
        upload["index"] = idx
        upload["index_built"] = True
        # determine vector count for message
        try:
            vec_count = embeddings.shape[0] if _HAS_NUMPY and hasattr(embeddings, "shape") else (len(embeddings) if embeddings else 0)
            upload["status_msg"] = f"Index Ready ({vec_count} vectors)"
        except Exception:
            upload["status_msg"] = "Index Ready"
    except Exception as e:
        logger.exception("build_vector_index_safe: failed to embed or build index: %s", e)
        upload["index_built"] = False
        upload["status_msg"] = f"Index build failed: {e}"

    return upload

def answer_question_with_rag_safe(query, indexed_uploads, top_k=None):
    """
    Retrieval-Augmented answer:
    - If embeddings + index are available: use them.
    - Otherwise fallback to keyword-overlap ranking across chunks.
    Returns a string answer (from LLM if available, otherwise a concise fallback).
    """
    if not query or not isinstance(query, str):
        return "Please enter a question."

    top_k = int(top_k or globals().get("TOP_K", 5))

    # Collect candidate chunks
    found_chunks = []
    for upload in (indexed_uploads or []):
        if not upload:
            continue
        chunks = upload.get("chunks") or []
        idx = upload.get("index")
        if idx and hasattr(idx, "search"):
            # embeddings-based search
            try:
                # embed the query
                try:
                    model = _get_embedding_model()
                    q_emb = _embed_texts([query], model)
                except Exception:
                    q_emb = None
                if q_emb is not None:
                    dists, ids = idx.search(q_emb, k=min(top_k, max(1, len(chunks))))
                    for j in (ids[0] if isinstance(ids[0], (list, tuple)) else ids[0]):
                        if j is None or int(j) < 0:
                            continue
                        try:
                            found_chunks.append(chunks[int(j)])
                        except Exception:
                            continue
                    continue  # next upload
            except Exception:
                logger.debug("Embeddings search failed, will fallback to keyword scan for this upload")

        # Fallback: keyword overlap ranking (fast, dependency-free)
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
        return "I couldn't find relevant information in your documents to answer that."

    # Build context (trim to safe length)
    context = "\n---\n".join(found_chunks[: max(1, top_k * 2) ])
    system_prompt = "You are a concise assistant. Answer using ONLY the provided context. If the answer is not present, say you cannot answer."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    # Call LLM if available
    try:
        resp = _call_openrouter(system_prompt, user_prompt, model=globals().get("DEFAULT_LLM_MODEL", "gpt-4o-mini"), max_tokens=600)
        return resp
    except Exception as e:
        logger.exception("answer_question_with_rag_safe: LLM call failed: %s", e)
        # Friendly fallback: return the top matching chunks as a short summary
        try:
            sample = "\n\n".join(found_chunks[:3])
            preview = sample[:1500] + ("..." if len(sample) > 1500 else "")
            return f"(LLM unavailable) Relevant context:\n\n{preview}"
        except Exception:
            return "No LLM available and failed to construct fallback answer."

def add_flashcards_to_db_safe(upload, cards):
    """
    Save flashcards into sqlite if available, otherwise into st.session_state fallback.
    Cards is a list of {"question":..., "answer":...}.
    Returns number saved.
    """
    if not isinstance(cards, (list, tuple)) or not cards:
        return 0
    saved = 0
    # prefer sqlite
    conn = get_db_connection_safe()
    if conn is not None:
        try:
            with conn:
                uid = get_upload_db_id(upload)
                for c in cards:
                    q = (c.get("question") or "").strip()
                    a = (c.get("answer") or "").strip()
                    if not q or not a:
                        continue
                    conn.execute("INSERT INTO flashcards (upload_id, question, answer, next_review) VALUES (?, ?, ?, ?)",
                                 (uid, q, a, int(time.time())))
                    saved += 1
            return saved
        except Exception as e:
            logger.exception("add_flashcards_to_db_safe (sqlite) failed: %s", e)
            # fall back to session below

    # session fallback
    try:
        db = st.session_state.setdefault("flashcards_db", {})
        uid = upload.get("db_id") or upload.get("filename") or f"upload_{int(time.time())}"
        arr = db.setdefault(uid, [])
        for c in cards:
            q = (c.get("question") or "").strip()
            a = (c.get("answer") or "").strip()
            if not q or not a:
                continue
            arr.append({
                "id": f"{uid}-{len(arr)+1}",
                "upload_id": uid,
                "question": q,
                "answer": a,
                "easiness": 2.5,
                "interval": 1,
                "repetitions": 0,
                "next_review": int(time.time())
            })
            saved += 1
        st.session_state["flashcards_db"] = db
    except Exception as e:
        logger.exception("add_flashcards_to_db_safe (session) failed: %s", e)
    return saved

def get_due_flashcards_safe(upload=None, limit=100):
    """
    Retrieve due flashcards. If upload provided, scope to that upload. Uses sqlite when available, otherwise session fallback.
    """
    now = int(time.time())
    results = []
    conn = get_db_connection_safe()
    if conn is not None:
        try:
            if upload:
                uid = get_upload_db_id(upload)
                rows = conn.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards WHERE upload_id = ? AND IFNULL(next_review,0) <= ? ORDER BY next_review ASC LIMIT ?", (uid, now, limit)).fetchall()
            else:
                rows = conn.execute("SELECT id, question, answer, easiness, interval, repetitions, next_review FROM flashcards WHERE IFNULL(next_review,0) <= ? ORDER BY next_review ASC LIMIT ?", (now, limit)).fetchall()
            for r in rows:
                results.append({"id": r[0], "question": r[1], "answer": r[2], "easiness": r[3] or 2.5, "interval": r[4] or 1, "repetitions": r[5] or 0, "next_review": r[6] or now})
            return results
        except Exception as e:
            logger.exception("get_due_flashcards_safe (sqlite) failed: %s", e)

    # session fallback
    try:
        db = st.session_state.get("flashcards_db", {})
        for uid, arr in db.items():
            if upload and uid != (upload.get("db_id") or upload.get("filename")):
                continue
            for c in arr:
                if int(c.get("next_review", 0)) <= now:
                    results.append(c)
                    if len(results) >= limit:
                        return results
    except Exception as e:
        logger.exception("get_due_flashcards_safe (session) failed: %s", e)
    return results

def update_flashcard_review_safe(card, quality):
    """
    Update a flashcard's SM-2 fields in sqlite if possible, otherwise update session fallback.
    Expects 'card' to include an 'id' and either 'upload_id' (session) or numeric id (sqlite).
    """
    try:
        eas = float(card.get("easiness", 2.5))
        ivl = int(card.get("interval", 1))
        reps = int(card.get("repetitions", 0))
    except Exception:
        eas, ivl, reps = 2.5, 1, 0
    try:
        new_eas, new_ivl, new_reps, new_next = _sm2_update_card(eas, ivl, reps, int(quality))
    except Exception:
        new_eas, new_ivl, new_reps, new_next = (2.5, 1, 0, int(time.time()) + 86400)

    conn = get_db_connection_safe()
    if conn is not None and isinstance(card.get("id"), int):
        try:
            with conn:
                conn.execute("UPDATE flashcards SET easiness=?, interval=?, repetitions=?, next_review=? WHERE id=?",
                             (float(new_eas), int(new_ivl), int(new_reps), int(new_next), int(card.get("id"))))
            return True
        except Exception as e:
            logger.exception("update_flashcard_review_safe (sqlite) failed: %s", e)

    # session fallback: find and update
    try:
        db = st.session_state.get("flashcards_db", {})
        uid = card.get("upload_id") or card.get("id", "").split("-")[0]
        arr = db.get(uid, [])
        for c in arr:
            if c.get("id") == card.get("id"):
                c["easiness"] = float(new_eas)
                c["interval"] = int(new_ivl)
                c["repetitions"] = int(new_reps)
                c["next_review"] = int(new_next)
                st.session_state["flashcards_db"] = db
                return True
    except Exception as e:
        logger.exception("update_flashcard_review_safe (session) failed: %s", e)
    return False

# ---------- End of core functions ----------
# Export names to global space for other UI code to call (if desired)
globals().update({
    "process_new_upload_safe": process_new_upload_safe,
    "build_vector_index_safe": build_vector_index_safe,
    "answer_question_with_rag_safe": answer_question_with_rag_safe,
    "add_flashcards_to_db_safe": add_flashcards_to_db_safe,
    "get_due_flashcards_safe": get_due_flashcards_safe,
    "update_flashcard_review_safe": update_flashcard_review_safe,
    "get_db_connection_safe": get_db_connection_safe,
    "get_upload_db_id": get_upload_db_id,
    "APP_CSS_CORE": APP_CSS
})

# ------------------------------
# Quick usage notes (inline help)
# ------------------------------
# - Use process_new_upload_safe(uploaded_file) to create a standardized upload dict.
# - Use build_vector_index_safe(upload) to attempt embedding + index build.
# - Use answer_question_with_rag_safe(query, [upload, ...]) to query indexed uploads.
# - Use add_flashcards_to_db_safe(upload, cards) to persist generated flashcards.
# - Use get_due_flashcards_safe(upload) to retrieve due cards and update_flashcard_review_safe(card, quality) to schedule next review.
#
# These functions will prefer real dependencies (sqlite3, sentence-transformers, VectorIndex, call_openrouter)
# if they are available in the surrounding script; otherwise they fall back to safe, dependency-free behavior.
# ------------------------------
