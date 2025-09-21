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
    if package_name is None:
        package_name = module_name
    try:
        module = __import__(module_name)
        return module, True
    except ImportError:
        logging.getLogger(__name__).warning(f"Module '{module_name}' not found. Optional: pip install {package_name}")
        return None, False

# Core required libs
st, _HAS_STREAMLIT = _try_import("streamlit")
if not _HAS_STREAMLIT:
    raise RuntimeError("streamlit is required. Install: pip install streamlit")

np, _HAS_NUMPY = _try_import("numpy")
if not _HAS_NUMPY:
    raise RuntimeError("numpy is required. Install: pip install numpy")

requests, _HAS_REQUESTS = _try_import("requests")
pptx_module, _HAS_PPTX = _try_import("pptx", "python-pptx")
fitz, _HAS_PYMUPDF = _try_import("fitz", "PyMuPDF")  # PyMuPDF imported as fitz
easyocr_module, _HAS_EASYOCR = _try_import("easyocr")
easyocr = easyocr_module if _HAS_EASYOCR else None
sentence_transformers_module, _HAS_SENTENCE_TRANSFORMERS = _try_import("sentence_transformers")
SentenceTransformer = sentence_transformers_module.SentenceTransformer if _HAS_SENTENCE_TRANSFORMERS else None
faiss, _HAS_FAISS = _try_import("faiss", "faiss-cpu")
gtts_module, _HAS_GTTS = _try_import("gtts")

# ----------------------------
# Logging + configuration
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger("slidetutor")

DEFAULT_OPENROUTER_KEY: str = (st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else None) or os.getenv("OPENROUTER_API_KEY", "")
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
@st.cache_resource
def get_db_connection(path: str = DB_PATH) -> sqlite3.Connection:
    logger.info("Opening DB: %s", path)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    with conn:
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
    logger.info("DB initialized.")

# Initialize DB
try:
    init_db(get_db_connection())
except Exception as e:
    logger.exception("Failed initializing DB: %s", e)

# ----------------------------
# Vector index (FAISS optional, numpy fallback)
# ----------------------------
_NORM_EPS = 1e-9

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr

def _safe_normalize(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    norm[norm < _NORM_EPS] = _NORM_EPS
    return arr / norm

class VectorIndex:
    def __init__(self, embeddings: np.ndarray, texts: List[str]):
        self.texts = texts
        self.embeddings = _ensure_2d(np.asarray(embeddings, dtype=np.float32)) if embeddings is not None else np.zeros((0, 0), dtype=np.float32)
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
                logger.info("FAISS index built with %d vectors", self.embeddings.shape[0])
            except Exception as e:
                logger.warning("FAISS index build failed, falling back to NumPy: %s", e)
                self._use_faiss = False

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = _ensure_2d(np.asarray(query_embeddings, dtype=np.float32))
        if self._use_faiss and self.faiss_index is not None and self.dimension:
            try:
                qn = np.ascontiguousarray(_safe_normalize(queries))
                sims, idxs = self.faiss_index.search(qn, k)
                # faiss returns similarity (cosine via inner product), convert to distance-like
                dists = 1.0 - sims.astype(np.float32)
                return dists, idxs.astype(np.int64)
            except Exception as e:
                logger.warning("FAISS search failed: %s -- falling back to numpy", e)
        return self._numpy_search(queries, k)

    def _numpy_search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings.size == 0:
            return np.full((queries.shape[0], k), np.inf, dtype=np.float32), np.full((queries.shape[0], k), -1, dtype=np.int64)
        nq = _safe_normalize(queries)
        sims = nq @ self._normed_embeddings.T
        k_eff = min(k, sims.shape[1])
        idxs = np.argsort(-sims, axis=1)[:, :k_eff]
        final_sims = np.take_along_axis(sims, idxs, axis=1)
        # pad if necessary
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
    if not _HAS_SENTENCE_TRANSFORMERS:
        raise RuntimeError("sentence-transformers not installed. `pip install sentence-transformers` to enable embeddings.")
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.exception("Failed to load embedding model: %s", e)
        raise

def embed_texts(texts: List[str], model) -> np.ndarray:
    if not texts:
        # attempt to infer dim from model
        dim = getattr(model, "get_sentence_embedding_dimension", lambda: 384)()
        return np.zeros((0, dim), dtype=np.float32)
    enc = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return _ensure_2d(enc.astype(np.float32))

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current += (p + "\n")
        else:
            if current:
                chunks.append(current.strip())
            current = p + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def extract_json_from_text(text: str) -> Optional[Any]:
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # try to be lenient by fixing common trailing commas, etc.
        cleaned = re.sub(r',\s*([\]}])', r'\1', match.group(0))
        try:
            return json.loads(cleaned)
        except Exception:
            return None

# ----------------------------
# OCR / File parsing
# ----------------------------
@st.cache_resource(show_spinner="Loading OCR model...")
def get_easyocr_reader(lang_list: List[str] = ["en"]) -> Optional["easyocr.Reader"]:
    if not _HAS_EASYOCR:
        return None
    try:
        return easyocr.Reader(lang_list, gpu=False)
    except Exception as e:
        logger.warning("EasyOCR init failed: %s", e)
        return None

def ocr_image_bytes(image_bytes: bytes, reader) -> str:
    if not image_bytes or reader is None:
        return ""
    try:
        # EasyOCR accepts numpy array or bytes; reader.readtext can accept PIL/image.
        # Try readtext returning paragraphs (detail=0)
        return "\n".join(reader.readtext(image_bytes, detail=0, paragraph=True)).strip()
    except Exception as e:
        logger.debug("OCR failed for image: %s", e)
        return ""

def parse_and_extract_content(filename: str, file_bytes: bytes) -> List[SlideData]:
    file_ext = os.path.splitext(filename)[1].lower()
    raw_slides: List[Dict[str, Any]] = []
    if file_ext == ".pptx":
        if not _HAS_PPTX:
            raise RuntimeError("python-pptx not installed. Install: pip install python-pptx")
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
                # images: python-pptx shape.image.blob if present
                try:
                    if hasattr(shape, "image") and getattr(shape, "image") is not None:
                        images.append(shape.image.blob)
                except Exception:
                    # some shapes raise
                    pass
            raw_slides.append({"index": i, "text": "\n".join(texts).strip(), "images": images})
    elif file_ext == ".pdf":
        if not _HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF not installed. Install: pip install PyMuPDF")
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            images: List[bytes] = []
            try:
                for img in page.get_images(full=True):
                    xref = img[0]
                    try:
                        imgdict = doc.extract_image(xref)
                        images.append(imgdict["image"])
                    except Exception:
                        # fallback: use page pixmap as image
                        pass
            except Exception:
                pass
            if not text and not images:
                # snapshot of page as png
                try:
                    pix = page.get_pixmap(dpi=150)
                    images.append(pix.tobytes("png"))
                except Exception:
                    pass
            raw_slides.append({"index": i, "text": text, "images": images})
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    ocr_reader = get_easyocr_reader()
    processed: List[SlideData] = []
    for s in raw_slides:
        ocr_texts = []
        for img in s["images"]:
            try:
                t = ocr_image_bytes(img, ocr_reader)
                if t:
                    ocr_texts.append(t)
            except Exception:
                continue
        ocr_text = "\n".join(ocr_texts).strip()
        processed.append({"index": s["index"], "text": s["text"] or "", "images": s["images"], "ocr_text": ocr_text})
    return processed

# ----------------------------
# LLM / OpenRouter call wrapper
# ----------------------------
def call_openrouter(system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 1500, temperature: float = 0.1) -> str:
    api_key = st.session_state.get("OPENROUTER_API_KEY") or DEFAULT_OPENROUTER_KEY
    if not api_key:
        raise APIError("OpenRouter API key not configured. Set OPENROUTER_API_KEY env or in Settings.")
    headers = {"Authorization": f"Bearer {api_key}"}
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
        # OpenRouter: 'choices' -> [ { 'message': { 'content': "..." } } ]
        choices = body.get("choices", [])
        if not choices:
            raise APIError("No choices returned from OpenRouter")
        content = choices[0].get("message", {}).get("content") or choices[0].get("text")
        if not content:
            raise APIError("Empty content from OpenRouter response")
        return content.strip()
    except requests.exceptions.RequestException as e:
        logger.exception("OpenRouter request failed: %s", e)
        raise APIError(f"OpenRouter request failed: {e}")

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
# RAG Q&A
# ----------------------------
def answer_question_with_rag(query: str, uploads: List[Dict]) -> str:
    if not query:
        return "Please provide a question."
    try:
        model = get_embedding_model()
    except Exception as e:
        return "Embeddings not available: " + str(e)
    q_emb = embed_texts([query], model)
    found_chunks = []
    # each upload: must have 'index' (VectorIndex) and 'chunks' list
    for upload in uploads:
        idx: VectorIndex = upload.get("index")
        if idx and idx.dimension:
            dists, ids = idx.search(q_emb, k=min(TOP_K, len(upload.get("chunks", [])) or 1))
            for j in ids[0]:
                if j is None or j == -1:
                    continue
                try:
                    found_chunks.append(upload["chunks"][int(j)])
                except Exception:
                    continue
    if not found_chunks:
        return "Couldn't find relevant information in your documents."
    context = "\n---\n".join(found_chunks[:TOP_K*3])
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
# Streamlit UI
# ----------------------------
APP_CSS = """
/* Minimal theming to improve readability */
body { font-family: Inter, sans-serif; }
"""

def initialize_session_state():
    defaults = {
        "uploads": [],  # list of dicts {filename, file_bytes, status_msg, slides_data, chunks, embeddings, index, index_built, db_id}
        "OPENROUTER_API_KEY": st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else os.getenv("OPENROUTER_API_KEY", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def handle_file_processing(upload_idx: int):
    upload = st.session_state.uploads[upload_idx]
    filename = upload["filename"]
    try:
        with st.spinner(f"Processing {filename} ..."):
            # parse
            upload["status_msg"] = "Extracting content..."
            slides = parse_and_extract_content(filename, upload["file_bytes"])
            upload["slides_data"] = slides
            upload["slide_count"] = len(slides)
            # aggregate text (slide text + OCR)
            full_text = "\n\n".join(((s["text"] or "") + "\n" + (s["ocr_text"] or "")).strip() for s in slides).strip()
            upload["full_text"] = full_text
            upload["status_msg"] = "Chunking..."
            upload["chunks"] = chunk_text(full_text, max_chars=1200) or ([full_text] if full_text else [])
            # embeddings + index
            if _HAS_SENTENCE_TRANSFORMERS:
                upload["status_msg"] = "Embedding chunks..."
                model = get_embedding_model()
                upload["embeddings"] = embed_texts(upload["chunks"], model)
                upload["index"] = VectorIndex(upload["embeddings"], upload["chunks"])
                upload["index_built"] = True
            else:
                upload["index_built"] = False
                upload["status_msg"] = "Embeddings unavailable"
            # store in DB
            conn = get_db_connection()
            with conn:
                cur = conn.execute("INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)",
                                   (filename, int(time.time()), json.dumps({"slide_count": upload["slide_count"]})))
                upload["db_id"] = cur.lastrowid
            upload["status_msg"] = f"Ready ({len(upload['chunks'])} chunks)"
    except Exception as e:
        logger.exception("Processing failed: %s", e)
        upload["status_msg"] = f"Error: {e}"

# UI render helpers
def render_home():
    st.markdown(f"## {APP_TITLE}")
    st.write(APP_SUBTITLE)
    st.info("Upload PPTX or PDF files and generate lessons, flashcards, quizzes, or ask questions (RAG).")

def render_upload_tab():
    st.header("Upload & Process")
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
                    "db_id": None
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
                        st.markdown(f"**Slide {s['index']+1}**")
                        if s["text"]:
                            st.write(s["text"])
                        if s["ocr_text"]:
                            st.caption("OCR text:")
                            st.write(s["ocr_text"])
                        st.markdown("---")
            if cols[3].button("Delete", key=f"del_{i}"):
                # remove and db cleanup if present
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
    selected = st.selectbox("Choose upload", ["<none>"] + [u["filename"] for u in st.session_state.uploads])
    if selected == "<none>":
        st.info("Select a processed upload to generate a lesson.")
        return
    upload = next((u for u in st.session_state.uploads if u["filename"] == selected), None)
    if not upload:
        st.error("Upload not found.")
        return
    text = upload.get("full_text") or "\n".join(upload.get("chunks") or [])
    if not text.strip():
        st.warning("This document has no extracted text.")
        return
    with st.form("lesson_form"):
        level = st.selectbox("Max level to generate up to", ["Beginner->Advanced", "Beginner only", "Beginner+Intermediate"])
        q = st.number_input("Example quiz questions", min_value=1, max_value=10, value=3)
        submit = st.form_submit_button("Generate lesson")
    if submit:
        try:
            prompt_context = text if level == "Beginner->Advanced" else (text if "Beginner" in level else text)
            lesson = generate_multilevel_lesson(prompt_context)
            st.success("Lesson generated")
            st.text_area("Generated lesson", lesson, height=400)
        except APIError as e:
            st.error(f"LLM error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

def render_mcq_tab():
    st.header("Generate MCQs")
    selected = st.selectbox("Choose upload for MCQs", ["<none>"] + [u["filename"] for u in st.session_state.uploads], key="mcq_select")
    if selected == "<none>":
        st.info("Select a processed upload")
        return
    upload = next((u for u in st.session_state.uploads if u["filename"] == selected), None)
    if not upload:
        st.error("Upload not found")
        return
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
                        qtext = obj.get("question")
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
    selected = st.selectbox("Choose upload for flashcards", ["<none>"] + [u["filename"] for u in st.session_state.uploads], key="fc_select")
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
                        qtext = obj.get("question", "")[:2000]
                        ans = obj.get("answer", "")[:2000]
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
        for q, opts_json, correct_index in rows:
            opts = json.loads(opts_json)
            ans = st.radio(q, opts, key=f"quiz_{hash(q)}")
            if st.button("Submit answer", key=f"sub_{hash(q)}"):
                if opts.index(ans) == correct_index:
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

# Main
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
    initialize_session_state()

    st.sidebar.title(APP_TITLE)
    st.sidebar.markdown(APP_SUBTITLE)
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
# OCR (EasyOCR) Management
# ------------------------------

@st.cache_resource(show_spinner="Loading OCR model...")
def get_easyocr_reader(lang_list: List[str] = ["en"]) -> Optional["easyocr.Reader"]:
    """Loads and caches an EasyOCR reader instance."""
    if not _HAS_EASYOCR or easyocr is None:
        logger.warning("EasyOCR not installed, OCR functionality will be disabled.")
        return None
    try:
        logger.info(f"Loading EasyOCR reader for languages: {lang_list}")
        return easyocr.Reader(lang_list, gpu=False)
    except Exception as e:
        st.error(f"Failed to initialize EasyOCR: {e}")
        logger.exception("EasyOCR initialization failed.")
        return None

def ocr_image_bytes(image_bytes: bytes, reader: "easyocr.Reader") -> str:
    """Runs OCR on a single image's bytes and returns the combined text."""
    if not image_bytes or not reader:
        return ""
    try:
        # Use paragraph=True for better text block grouping.
        text_list = reader.readtext(image_bytes, detail=0, paragraph=True)
        return "\n".join(text_list).strip()
    except Exception as e:
        logger.warning(f"EasyOCR failed for an image: {e}", exc_info=True)
        return ""


# ------------------------------
# File Parsers (PPTX, PDF)
# ------------------------------

def _extract_from_pptx(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Robustly extracts text and images from a PPTX file."""
    if not _HAS_PPTX or Presentation is None:
        raise ModuleNotFoundError("Please install 'python-pptx' to process PPTX files.")
    slides_content = []
    prs = Presentation(io.BytesIO(file_bytes))
    for i, slide in enumerate(prs.slides):
        texts, images = [], []
        for shape in slide.shapes:
            if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                texts.append(shape.text.strip())
            if hasattr(shape, "image"):
                images.append(shape.image.blob)
        slides_content.append({"index": i, "text": "\n".join(filter(None, texts)), "images": images})
    return slides_content

def _extract_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Robustly extracts text and images from a PDF file, with OCR fallback."""
    if not _HAS_PYMUPDF or fitz is None:
        raise ModuleNotFoundError("Please install 'PyMuPDF' to process PDF files.")
    slides_content = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            images = [doc.extract_image(img[0])["image"] for img in page.get_images(full=True)]
            # If page is image-based (no text, no extracted images), render it for OCR.
            if not text and not images:
                pix = page.get_pixmap(dpi=150)
                images.append(pix.tobytes("png"))
            slides_content.append({"index": i, "text": text, "images": images})
    return slides_content


# ------------------------------
# Main Content Extraction Pipeline
# ------------------------------

def parse_and_extract_content(filename: str, file_bytes: bytes) -> List[SlideData]:
    """
    Main pipeline to parse a file, extract text/images, and perform OCR.
    This is the primary function to call after a file is uploaded.
    """
    file_ext = os.path.splitext(filename)[1].lower()
    logger.info(f"Starting content extraction for '{filename}' ({file_ext}).")

    try:
        if file_ext == ".pptx":
            raw_slides = _extract_from_pptx(file_bytes)
        elif file_ext == ".pdf":
            raw_slides = _extract_from_pdf(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: '{file_ext}'. Please upload a PPTX or PDF.")
    except Exception as e:
        st.error(f"Failed to parse '{filename}': {e}")
        logger.exception(f"Error parsing file {filename}.")
        return []

    # Initialize OCR reader (will be fetched from cache if already loaded)
    ocr_reader = get_easyocr_reader(["en"])
    processed_slides: List[SlideData] = []

    st.progress(0, text="Processing slides...")
    for i, slide in enumerate(raw_slides):
        ocr_texts = [ocr_image_bytes(img_bytes, ocr_reader) for img_bytes in slide["images"]]
        processed_slides.append({
            "index": slide["index"],
            "text": slide["text"],
            "images": slide["images"],
            "ocr_text": "\n".join(filter(None, ocr_texts)).strip()
        })
        st.progress((i + 1) / len(raw_slides), text=f"Processing slide {i+1}/{len(raw_slides)}...")

    logger.info(f"Successfully extracted content from {len(processed_slides)} slides.")
    return processed_slides

import re

# ------------------------------
# Text Processing Utilities
# ------------------------------

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Splits text into semantic chunks, respecting paragraphs and sentences.
    Each chunk is guaranteed to be under `max_chars`.

    Args:
        text: The input text to be chunked.
        max_chars: The maximum character length for any single chunk.

    Returns:
        A list of text chunks.
    """
    if not text or not isinstance(text, str):
        return []

    # Split text into paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        # If a single paragraph is too long, split it by sentences
        if len(p) > max_chars:
            # Add the current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # Simple sentence split, can be improved with more complex regex if needed
            sentences = re.split(r'(?<=[.!?])\s+', p)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk += sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            if current_chunk: # Add the last sentence chunk
                 chunks.append(current_chunk.strip())
                 current_chunk = ""
            continue

        # Otherwise, add paragraphs to the current chunk
        if len(current_chunk) + len(p) + 1 <= max_chars:
            current_chunk += p + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def extract_json_from_text(text: str) -> Optional[Any]:
    """
    Extracts a JSON object or array from a string, ignoring markdown code fences.
    Uses a robust regex to find the first complete JSON structure.

    Args:
        text: The string potentially containing an embedded JSON object or array.

    Returns:
        The parsed Python object (dict or list), or None if no valid JSON is found.
    """
    if not text or not isinstance(text, str):
        return None

    # Regex to find JSON wrapped in optional ```json ... ``` fences
    # It looks for the content between the first '{' or '[' and the last '}' or ']'
    json_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)

    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extracted JSON string: {e}\nString was: {json_str[:200]}...")

    logger.debug("No valid JSON object or array found in the text.")
    return None


# ------------------------------
# LLM API Communication (OpenRouter)
# ------------------------------

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    is_json: bool = False
) -> str:
    """
    Calls the OpenRouter Chat API and returns the response content.

    Args:
        system_prompt: The system message to guide the model's behavior.
        user_prompt: The user's message or query.
        model: The model identifier to use (e.g., 'gpt-4o-mini').
        max_tokens: The maximum number of tokens to generate.
        temperature: The sampling temperature for generation (0.0 to 2.0).
        is_json: If True, adds a 'response_format' field for models that support it.

    Returns:
        The content of the model's response as a string.

    Raises:
        APIError: If the API key is missing or an API request fails.
    """
    api_key = st.session_state.get("OPENROUTER_API_KEY") or DEFAULT_OPENROUTER_KEY
    if not api_key:
        raise APIError("OpenRouter API key is not configured.")

    headers = {"Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if is_json:
        body["response_format"] = {"type": "json_object"}

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=45)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise APIError("Received an empty or invalid response from the API.")
        return content.strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}. Response: {e.response.text if e.response else 'N/A'}")
        raise APIError(f"Network error or API failure: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse API response: {e}. Raw response: {response.text}")
        raise APIError("Failed to parse a valid response from the API.")


# ------------------------------
# Content Generation Prompts
# ------------------------------

# Prompts are defined as constants for clarity and easy modification.
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
1.  **Concept Explanation**: Start from first principles and progressively increase complexity.
2.  **Worked Examples**: Provide two distinct examples: one numerical (with explicit steps and SI units) and one conceptual.
3.  **Common Mistakes**: List 2-3 common pitfalls or misunderstandings students have.
4.  **Practice MCQs**: Write 5 short multiple-choice questions with the correct answer clearly indicated.
5.  **Flashcard Q&A**: Generate 8 concise question/answer pairs suitable for flashcards.
Your explanation must be clear, thorough, and tailored to the specified audience.
"""

PROMPT_MCQ_JSON = """
You are an AI assistant that generates high-quality multiple-choice questions (MCQs) from a given text.
Your response MUST be a single, valid JSON array. Each object in the array represents one MCQ and must have these exact keys:
- "question": A string containing the question text.
- "options": An array of 4 strings representing the possible answers.
- "answer_index": A 0-based integer indicating the correct option.
Do not include any explanations, introductory text, or markdown formatting. Only the JSON array.
"""

PROMPT_FLASHCARDS_JSON = """
You are an AI assistant that extracts concise question-and-answer pairs from text, suitable for flashcards.
Your response MUST be a single, valid JSON array of objects. Each object must have these exact keys:
- "question": A string for the front of the card (max 80 chars).
- "answer": A string for the back of the card (max 200 chars).
Do not include any explanations, introductory text, or markdown formatting. Only the JSON array.
"""

# ------------------------------
# Content Generation Functions
# ------------------------------

def generate_multilevel_lesson(slide_text: str, related_texts: str = "") -> str:
    user_prompt = f"Based on the following slide text and related context, generate a multi-level lesson.\n\n**SLIDE TEXT:**\n{slide_text}\n\n**RELATED CONTEXT:**\n{related_texts}"
    return call_openrouter(PROMPT_LESSON_MULTILEVEL, user_prompt, max_tokens=1500, temperature=0.2)

def generate_deep_lesson(slide_text: str, related_texts: str = "") -> str:
    user_prompt = f"Based on the following slide text and related context, generate a deep-dive lesson.\n\n**SLIDE TEXT:**\n{slide_text}\n\n**RELATED CONTEXT:**\n{related_texts}"
    return call_openrouter(PROMPT_LESSON_DEEP, model="gpt-4o-mini", max_tokens=3000, temperature=0.1)

def generate_mcq_set_from_text(text: str, qcount: int = 5) -> List[Dict]:
    user_prompt = f"Create exactly {qcount} MCQs based on the following text.\n\n**TEXT:**\n{text}"
    try:
        response_text = call_openrouter(PROMPT_MCQ_JSON, user_prompt, max_tokens=800, temperature=0.0, is_json=True)
        parsed_json = extract_json_from_text(response_text)
        return parsed_json if isinstance(parsed_json, list) else []
    except APIError as e:
        st.warning(f"Could not generate quiz questions: {e}")
        return []

def generate_flashcards_from_text(text: str, n: int = 10) -> List[Dict]:
    user_prompt = f"Extract up to {n} flashcards from the text below.\n\n**TEXT:**\n{text}"
    try:
        response_text = call_openrouter(PROMPT_FLASHCARDS_JSON, user_prompt, max_tokens=1000, temperature=0.0, is_json=True)
        parsed_json = extract_json_from_text(response_text)
        return parsed_json if isinstance(parsed_json, list) else []
    except APIError as e:
        st.warning(f"Could not generate flashcards: {e}")
        return []

# ------------------------------
# Spaced Repetition (SM-2 Algorithm)
# ------------------------------

def sm2_update_card(
    easiness: float = 2.5,
    interval: int = 1,
    repetitions: int = 0,
    quality: int = 0
) -> Tuple[float, int, int, int]:
    """
    Updates card metadata based on the SM-2 spaced repetition algorithm.

    Args:
        easiness: The current easiness factor (E-Factor) of the card.
        interval: The current interval in days before the card is reviewed again.
        repetitions: The number of times the card has been successfully recalled.
        quality: The user's recall quality rating (0-5, where >=3 is a pass).

    Returns:
        A tuple containing (new_easiness, new_interval, new_repetitions, next_review_timestamp).
    """
    quality = max(0, min(5, quality)) # Ensure quality is within [0, 5]

    if quality < 3:
        # Failed recall: reset repetitions and interval
        repetitions = 0
        interval = 1
    else:
        # Successful recall
        repetitions += 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = max(1, round(interval * easiness))

    # Update easiness factor
    easiness += (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    easiness = max(1.3, easiness)

    # Calculate next review date (in seconds since epoch)
    day_in_seconds = 86400
    next_review_timestamp = int(time.time()) + interval * day_in_seconds
    return easiness, interval, repetitions, next_review_timestamp

# ------------------------------
# Export and Text-to-Speech (TTS)
# ------------------------------

def anki_export_tsv(upload_id: int, conn: sqlite3.Connection) -> Optional[Tuple[str, bytes]]:
    """
    Exports flashcards for a given upload as a TSV file in-memory.

    Args:
        upload_id: The ID of the upload whose flashcards to export.
        conn: The active database connection.

    Returns:
        A tuple of (filename, file_bytes) or None if an error occurs.
    """
    try:
        cursor = conn.execute("SELECT question, answer FROM flashcards WHERE upload_id = ?", (upload_id,))
        rows = cursor.fetchall()
        if not rows:
            st.info("No flashcards found for this upload to export.")
            return None

        output = io.StringIO()
        for q, a in rows:
            # Sanitize by replacing tabs and newlines, which can break TSV format
            q_sanitized = (q or "").replace("\t", " ").replace("\n", "<br>")
            a_sanitized = (a or "").replace("\t", " ").replace("\n", "<br>")
            output.write(f"{q_sanitized}\t{a_sanitized}\n")

        tsv_bytes = output.getvalue().encode("utf-8")
        filename = f"slidetutor_anki_deck_{upload_id}.txt"
        return filename, tsv_bytes
    except sqlite3.Error as e:
        logger.exception(f"Database error during Anki export for upload {upload_id}: {e}")
        st.error("A database error occurred during the export.")
        return None

def text_to_speech(text: str, lang: str = "en") -> Tuple[str, bytes]:
    """
    Synthesizes text to speech (MP3) in memory using gTTS.

    Args:
        text: The text to convert to speech.
        lang: The language of the text (e.g., 'en', 'hi').

    Returns:
        A tuple of (filename, mp3_bytes).

    Raises:
        RuntimeError: If gTTS is not installed.
        ValueError: If no text is provided.
    """
    if not _HAS_GTTS or gTTS is None:
        raise RuntimeError("Text-to-Speech library not found. Please run: pip install gTTS")
    if not text or not text.strip():
        raise ValueError("Cannot generate audio from empty text.")

    try:
        mp3_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return "lesson_audio.mp3", mp3_fp.getvalue()
    except Exception as e:
        logger.exception(f"Text-to-Speech generation failed: {e}")
        raise RuntimeError(f"Failed to generate audio: {e}")
    
# ------------------------------
# Streamlit UI - Main Application
# ------------------------------

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")

    # --- STATE INITIALIZATION ---
    # Initialize all session state keys at the beginning to prevent errors.
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

    # --- STYLING ---
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

    # --- HEADER ---
    render_header()

    # --- TABS (MAIN NAVIGATION) ---
    tab_names = ["🏠 Home", "📤 Upload", "📚 Lessons", "💬 Chat Q&A", "📝 Quizzes", "🃏 Flashcards", "⚙️ Settings"]
    home, upload, lessons, chat, quizzes, flashcards, settings = st.tabs(tab_names)

    with home:
        render_home_tab()
    with upload:
        render_upload_tab()
    with lessons:
        render_lessons_tab()
    with chat:
        render_chat_tab()
    with quizzes:
        render_quizzes_tab()
    with flashcards:
        render_flashcards_tab()
    with settings:
        render_settings_tab()

# ------------------------------
# UI Components & Tabs
# ------------------------------

APP_CSS = """
    :root {
        --bg: #071025; --bg-2: #06101A; --text: #E6F0FA; --muted: #9AA6B2;
        --accent: #2AB7A9; --accent-2: #4D7CFE; --radius: 12px;
    }
    .stApp { background-color: var(--bg); color: var(--text); }
    .stTabs [role="tab"] { font-weight: 600; color: var(--muted); }
    .stTabs [aria-selected="true"] { color: var(--text); }
    .stButton>button { border-radius: 8px; }
    .stExpander { border-radius: var(--radius); border: 1px solid rgba(255,255,255,0.05); }
    h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        padding: 24px; border-radius: var(--radius);
        border: 1px solid rgba(255,255,255,0.03);
        box-shadow: 0 8px 28px rgba(2,6,12,0.5);
    }
    .small-muted { color: var(--muted); font-size: 14px; }
"""

def render_header():
    """Renders the top header of the application."""
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="font-size: 2.5rem; font-weight: 700;">🎓</div>
            <div>
                <h1 style="margin:0; padding:0;">{APP_TITLE}</h1>
                <p class="small-muted" style="margin:0; padding:0;">{APP_SUBTITLE}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def get_active_upload():
    """Helper to get the currently selected upload object from session state."""
    idx = st.session_state.get("active_upload_idx")
    uploads = st.session_state.get("uploads", [])
    if idx is not None and 0 <= idx < len(uploads):
        return uploads[idx]
    return None

def render_home_tab():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Welcome to SlideTutor! 🚀")
    st.markdown(
        "Your personal AI tutor for mastering lecture slides. "
        "Transform any **PPTX** or **PDF** file into interactive learning modules."
    )
    st.markdown(
        """
        **How it works:**
        1.  **Upload**: Go to the **📤 Upload** tab and add your slide decks.
        2.  **Process**: Click **"Build Index"** to let the AI analyze and understand the content.
        3.  **Learn**: Use the **📚 Lessons**, **💬 Chat Q&A**, **📝 Quizzes**, and **🃏 Flashcards** tabs to study smarter.
        """
    )
    st.success("Ready to start? Head over to the **📤 Upload** tab!")
    st.markdown("</div>", unsafe_allow_html=True)

def render_upload_tab():
    st.markdown("### 1. Upload Your Files")
    st.markdown("<p class='small-muted'>Add one or more PPTX or PDF files. The content will be extracted and prepared for AI processing.</p>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=["pdf", "pptx"]
    )

    if uploaded_files:
        existing_filenames = {u["filename"] for u in st.session_state.uploads}
        for file in uploaded_files:
            if file.name not in existing_filenames:
                with st.spinner(f"Processing '{file.name}'..."):
                    try:
                        new_upload = process_new_upload(file)
                        st.session_state.uploads.append(new_upload)
                        st.success(f"✅ Successfully processed '{file.name}'!")
                    except Exception as e:
                        st.error(f"❌ Failed to process '{file.name}': {e}")
            else:
                 st.warning(f"⚠️ File '{file.name}' has already been uploaded and was skipped.")

    if not st.session_state.uploads:
        st.info("No files uploaded yet. Use the uploader above to get started.")
        return

    st.markdown("---")
    st.markdown("### 2. Build Search Index")
    st.markdown("<p class='small-muted'>After uploading, you must build a vector index for a file to enable the Lessons, Chat, and Quiz features. This allows the AI to search and understand the document's content.</p>", unsafe_allow_html=True)

    for i, upload in enumerate(st.session_state.uploads):
        with st.expander(f"**{upload['filename']}** ({upload.get('status_msg', 'Ready')})", expanded=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**Slides/Pages:** {upload['slide_count']}")
                st.write(f"**Text Chunks:** {len(upload['chunks'])}")
                if upload.get("index_built"):
                    emb_shape = upload.get("embeddings", np.array([[]])).shape
                    st.write(f"**Index:** Built ({emb_shape[0]} vectors, {emb_shape[1]} dims)")

            with col2:
                if st.button("Build Index", key=f"build_{i}", use_container_width=True, disabled=upload.get("index_built", False)):
                    with st.spinner(f"Building vector index for '{upload['filename']}'... This may take a moment."):
                        build_vector_index(upload)
                    st.rerun()

            with col3:
                if st.button("🗑️ Delete", key=f"del_{i}", use_container_width=True):
                    st.session_state.uploads.pop(i)
                    st.rerun()

def render_lessons_tab():
    uploads = [u for u in st.session_state.uploads if u.get("index_built")]
    if not uploads:
        st.warning("Please upload a file and build its index on the **📤 Upload** tab first.", icon="📤")
        return

    upload_options = {u["filename"]: i for i, u in enumerate(uploads)}
    selected_filename = st.selectbox("Select a processed document:", options=upload_options.keys())

    if selected_filename:
        idx = upload_options[selected_filename]
        st.session_state.active_upload_idx = idx
        upload = uploads[idx]

        st.markdown("### Generate a Lesson")
        st.markdown("<p class='small-muted'>Choose a lesson style and the AI will generate a tailored study guide based on the document content.</p>", unsafe_allow_html=True)

        if st.button("Generate Multi-Level Lesson", use_container_width=True):
            with st.spinner("🧑‍🏫 The AI teacher is preparing your multi-level lesson..."):
                try:
                    full_text = upload["full_text"]
                    lesson = generate_multilevel_lesson(full_text)
                    st.markdown(lesson)
                except APIError as e:
                    st.error(f"Failed to generate lesson: {e}")

def render_chat_tab():
    uploads = [u for u in st.session_state.uploads if u.get("index_built")]
    if not uploads:
        st.warning("Please upload a file and build its index on the **📤 Upload** tab to use the chat.", icon="📤")
        return

    st.markdown("### Chat with Your Documents")
    st.markdown("<p class='small-muted'>Ask questions about your uploaded content. The AI will use the document as context to find the best answer.</p>", unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = answer_question_with_rag(prompt, uploads)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except APIError as e:
                    st.error(f"Could not get an answer: {e}")

def render_quizzes_tab():
    upload = get_active_upload()
    if not upload or not upload.get("index_built"):
        st.warning("Please select a processed document in the **📚 Lessons** tab first.", icon="📚")
        return

    st.markdown(f"### Generate a Quiz for: **{upload['filename']}**")
    q_count = st.slider("Number of questions:", min_value=3, max_value=10, value=5)

    if st.button("Generate Quiz", use_container_width=True):
        with st.spinner("Generating quiz questions..."):
            mcqs = generate_mcq_set_from_text(upload['full_text'], q_count)

        if not mcqs:
            st.error("Failed to generate a quiz. Please try again.")
            return

        for i, mcq in enumerate(mcqs):
            st.markdown(f"**Question {i+1}:** {mcq['question']}")
            user_choice = st.radio("Options:", mcq['options'], key=f"q_{i}", index=None)
            # This part can be expanded to score the quiz.

def render_flashcards_tab():
    uploads = st.session_state.get("uploads", [])
    if not uploads:
        st.warning("Please upload a document on the **📤 Upload** tab first.", icon="📤")
        return

    st.markdown("### Flashcards & Spaced Repetition")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Generate Flashcards")
        upload_options = {u["filename"]: i for i, u in enumerate(uploads)}
        selected_filename = st.selectbox("Select document to generate cards from:", options=upload_options.keys())
        if selected_filename:
            idx = upload_options[selected_filename]
            upload = uploads[idx]
            if st.button("Generate & Save Flashcards", key=f"gen_fc_{idx}"):
                with st.spinner(f"Generating flashcards for {upload['filename']}..."):
                    cards = generate_flashcards_from_text(upload['full_text'], n=20)
                    add_flashcards_to_db(upload, cards)
                st.success(f"Generated and saved {len(cards)} flashcards!")

    with col2:
        st.markdown("#### Practice Due Cards")
        if st.button("Load Due Cards for Practice"):
            st.session_state.due_cards = get_due_flashcards()
            st.session_state.current_card_idx = 0
            if not st.session_state.due_cards:
                st.info("🎉 No cards are due for review right now!")
            st.rerun()

    if st.session_state.due_cards:
        render_flashcard_practice_ui()

def render_flashcard_practice_ui():
    """Renders the UI for a single flashcard practice session."""
    st.markdown("---")
    idx = st.session_state.current_card_idx
    cards = st.session_state.due_cards

    if idx >= len(cards):
        st.success("✨ Session complete! You've reviewed all due cards.")
        st.session_state.due_cards = []
        return

    card = cards[idx]
    st.markdown(f"##### Card {idx + 1} of {len(cards)}")
    with st.container(border=True):
        st.markdown(f"**Question:**\n> {card['question']}")
        if "show_answer" not in card: card["show_answer"] = False

        if st.button("Show Answer", key=f"show_{card['id']}"):
            card["show_answer"] = True

        if card["show_answer"]:
            st.markdown(f"**Answer:**\n> {card['answer']}")
            st.markdown("How well did you recall this?")
            c1, c2, c3, c4 = st.columns(4)
            quality_map = {"Again (Incorrect)": 1, "Hard": 3, "Good": 4, "Easy": 5}
            for i, (label, quality) in enumerate(quality_map.items()):
                if st.columns(4)[i].button(label, key=f"{label}_{card['id']}", use_container_width=True):
                    update_flashcard_review(card, quality)
                    st.session_state.current_card_idx += 1
                    st.rerun()

def render_settings_tab():
    st.markdown("### Settings")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### API Configuration")
    api_key = st.text_input(
        "OpenRouter API Key",
        value=st.session_state.OPENROUTER_API_KEY,
        type="password",
        help="Your API key is stored temporarily in the session state."
    )
    if api_key != st.session_state.OPENROUTER_API_KEY:
        st.session_state.OPENROUTER_API_KEY = api_key
        st.success("API Key updated for this session.")
    st.markdown("<p class='small-muted'>You can get a free API key from <a href='https://openrouter.ai/'>OpenRouter.ai</a>. It's recommended to set this as an environment variable for production.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Core Logic & Backend Functions
# ------------------------------

def process_new_upload(uploaded_file) -> Dict:
    """Reads, parses, and processes a single uploaded file."""
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

    # Use the robust parsing pipeline defined previously
    slides_data = parse_and_extract_content(filename, file_bytes)

    # Combine all text and chunk it
    full_text = "\n\n".join(
        (s.get('text', '') + "\n" + s.get('ocr_text', '')).strip() for s in slides_data
    ).strip()
    chunks = chunk_text(full_text)

    upload = {
        "filename": filename,
        "slide_count": len(slides_data),
        "slides_data": slides_data,
        "full_text": full_text,
        "chunks": chunks,
        "index_built": False,
        "status_msg": "Processed",
        "embeddings": None,
        "index": None,
    }
    # Persist the upload record to the database
    conn = get_db_connection()
    with conn:
        cursor = conn.execute(
            "INSERT INTO uploads (filename, uploaded_at, meta) VALUES (?, ?, ?)",
            (filename, int(time.time()), json.dumps({"slide_count": len(slides_data)}))
        )
        upload['db_id'] = cursor.lastrowid
    return upload

def build_vector_index(upload: Dict):
    """Computes embeddings and builds the vector index for an upload."""
    if not upload or not upload.get("chunks"):
        upload["status_msg"] = "Error: No text chunks to index."
        st.error(upload["status_msg"])
        return

    try:
        model = get_embedding_model()
        embeddings = embed_texts(upload["chunks"], model)
        upload["embeddings"] = embeddings
        upload["index"] = VectorIndex(embeddings, upload["chunks"])
        upload["index_built"] = True
        shape = embeddings.shape
        upload["status_msg"] = f"Index Ready ({shape[0]} vectors)"
    except Exception as e:
        upload["status_msg"] = f"Index Failed: {e}"
        logger.exception(f"Failed to build index for {upload['filename']}")
        st.error(upload["status_msg"])

def answer_question_with_rag(query: str, indexed_uploads: List[Dict]) -> str:
    """Performs Retrieval-Augmented Generation to answer a question."""
    if not query: return "Please ask a question."

    # 1. Retrieval: Find relevant chunks from all indexed documents
    model = get_embedding_model()
    query_embedding = embed_texts([query], model)
    all_chunks = []
    for upload in indexed_uploads:
        if upload.get("index"):
            distances, indices = upload["index"].search(query_embedding, k=TOP_K)
            for i in indices[0]:
                if i != -1:
                    all_chunks.append(upload["chunks"][i])

    if not all_chunks:
        return "I couldn't find any relevant information in your documents to answer that question."

    # 2. Augmentation: Create a prompt with the retrieved context
    context = "\n---\n".join(all_chunks)
    system_prompt = "You are a helpful AI assistant. Answer the user's question based *only* on the provided context. If the answer is not in the context, say so. Be concise."
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    # 3. Generation: Get the answer from the LLM
    return call_openrouter(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=500)

def add_flashcards_to_db(upload: Dict, cards: List[Dict]):
    """Saves a list of generated flashcards to the database."""
    upload_id = get_upload_db_id(upload)
    if not upload_id:
        st.error("Could not resolve upload ID to save flashcards.")
        return
    conn = get_db_connection()
    with conn:
        for card in cards:
            if "question" in card and "answer" in card:
                conn.execute(
                    "INSERT INTO flashcards (upload_id, question, answer, next_review) VALUES (?, ?, ?, ?)",
                    (upload_id, card["question"], card["answer"], int(time.time()))
                )

def get_due_flashcards() -> List[Dict]:
    """Fetches all flashcards that are due for review."""
    conn = get_db_connection()
    now = int(time.time())
    cursor = conn.execute(
        "SELECT id, question, answer, easiness, interval, repetitions FROM flashcards WHERE next_review <= ? ORDER BY next_review ASC", (now,)
    )
    # Convert rows to dictionaries for easier handling in the UI
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

def update_flashcard_review(card: Dict, quality: int):
    """Updates a flashcard's review data in the database using the SM-2 algorithm."""
    easiness, interval, reps, next_review = sm2_update_card(
        card['easiness'], card['interval'], card['repetitions'], quality
    )
    conn = get_db_connection()
    with conn:
        conn.execute(
            """UPDATE flashcards
               SET easiness = ?, interval = ?, repetitions = ?, next_review = ?
               WHERE id = ?""",
            (easiness, interval, reps, next_review, card['id'])
        )

# --- APP ENTRY POINT ---
if __name__ == "__main__":
    main()

## --------------------------------------------------------------------------------
# Replacement UI Block - From Header to the end of the file
# This is a complete, modular, and robust UI implementation.
# It assumes the backend functions from the previous steps are defined.
# --------------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all necessary keys in Streamlit's session state."""
    state_defaults = {
        "uploads": [],
        "chat_history": [],
        "viewing_upload_idx": None,
        "due_cards": [],
        "current_card_idx": 0,
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def handle_file_processing(upload_index: int):
    """The core pipeline for processing a single uploaded file."""
    upload = st.session_state.uploads[upload_index]
    filename = upload["filename"]
    file_bytes = upload["file_bytes"]
    progress_bar = st.progress(0, text=f"Starting processing for '{filename}'...")

    try:
        # 1. Parse slides and perform OCR
        progress_bar.progress(0.1, text=f"Extracting content from '{filename}'...")
        slides_data = parse_and_extract_content(filename, file_bytes)
        upload["slides_data"] = slides_data
        upload["slide_count"] = len(slides_data)

        # 2. Consolidate and chunk text
        progress_bar.progress(0.4, text="Combining and chunking text...")
        full_text = "\n\n".join(
            (s.get('text', '') + "\n" + s.get('ocr_text', '')).strip() for s in slides_data
        ).strip()
        upload["full_text"] = full_text
        upload["chunks"] = chunk_text(full_text)

        # 3. Build vector index
        progress_bar.progress(0.6, text=f"Building search index for '{filename}'...")
        build_vector_index(upload) # This function computes embeddings and creates the index

        upload["processed"] = True
        upload["status_msg"] = f"✅ Ready ({len(upload['chunks'])} chunks indexed)"
        progress_bar.progress(1.0, text=f"'{filename}' processed successfully!")
        time.sleep(1.5)
        progress_bar.empty()

    except Exception as e:
        upload["status_msg"] = f"❌ Error: {e}"
        logger.exception(f"Failed to process {filename}")
        st.error(f"An error occurred while processing '{filename}': {e}")
        progress_bar.empty()

@st.dialog("Slide Viewer", width="large")
def render_slide_viewer_modal():
    """Renders the slide viewer UI inside a modal dialog."""
    idx = st.session_state.get("viewing_upload_idx")
    if idx is None:
        st.error("No upload selected for viewing.")
        return

    upload = st.session_state.uploads[idx]
    slides = upload.get("slides_data", [])
    num_slides = len(slides)

    if "viewer_page" not in st.session_state:
        st.session_state.viewer_page = 0

    page_num = st.session_state.viewer_page

    # --- Navigation ---
    st.markdown(f"#### {upload['filename']}")
    c1, c2, c3 = st.columns([2, 5, 2])
    if c1.button("⬅️ Previous", use_container_width=True, disabled=(page_num <= 0)):
        st.session_state.viewer_page -= 1
        st.rerun()
    if c3.button("Next ➡️", use_container_width=True, disabled=(page_num >= num_slides - 1)):
        st.session_state.viewer_page += 1
        st.rerun()

    page_num_display = st.session_state.viewer_page + 1
    new_page = c2.slider(
        "Go to slide", 1, max(1, num_slides), page_num_display,
        label_visibility="collapsed"
    )
    if new_page != page_num_display:
        st.session_state.viewer_page = new_page - 1
        st.rerun()

    # --- Slide Display ---
    if 0 <= page_num < num_slides:
        slide = slides[page_num]
        image_bytes = slide["images"][0] if slide["images"] else None

        if image_bytes:
            st.image(image_bytes, caption=f"Slide {page_num + 1}", use_container_width=True)
        else:
            st.info("No visual for this slide. Displaying extracted text instead.")
            st.markdown(f"<div class='card'>{slide.get('text', 'No text extracted.')}</div>", unsafe_allow_html=True)

        with st.expander("View Extracted Text"):
            full_slide_text = (slide.get('text', '') + "\n" + slide.get('ocr_text', '')).strip()
            st.text(full_slide_text or "No text was extracted from this slide.")

def render_main_ui():
    """The primary function to render the entire Streamlit UI."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    initialize_session_state()

    # --- Custom CSS and Header ---
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
    render_header()

    # --- Main Navigation Tabs ---
    tab_list = ["🏠 Home", "📤 Upload & Process", "📚 Lessons", "💬 Chat Q&A", "📝 Quizzes", "🃏 Flashcards", "⚙️ Settings"]
    home_tab, upload_tab, lessons_tab, chat_tab, quizzes_tab, flashcards_tab, settings_tab = st.tabs(tab_list)

    with home_tab:
        render_home_tab()

    with upload_tab:
        st.header("Upload & Process Files")
        st.markdown("<p class='small-muted'>Step 1: Upload your files. Step 2: Click 'Process' to analyze them.</p>", unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Select one or more PPTX or PDF files",
            type=["pptx", "pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            existing_filenames = {u["filename"] for u in st.session_state.uploads}
            for file in uploaded_files:
                if file.name not in existing_filenames:
                    st.session_state.uploads.append({
                        "filename": file.name,
                        "file_bytes": file.getvalue(),
                        "processed": False,
                        "status_msg": "📎 Ready to process",
                    })

        if not st.session_state.uploads:
            st.info("Upload a file to get started.")
        else:
            st.markdown("---")
            # Create a copy for safe iteration while deleting
            for i, upload in enumerate(list(st.session_state.uploads)):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.subheader(upload["filename"])
                        st.markdown(f"<p class='small-muted'>{upload['status_msg']}</p>", unsafe_allow_html=True)
                    with c2:
                        btn_cols = st.columns(3)
                        if btn_cols[0].button("⚙️ Process", key=f"proc_{i}", use_container_width=True, disabled=upload.get("processed", False)):
                            handle_file_processing(i)
                            st.rerun()
                        if btn_cols[1].button("👁️ View", key=f"view_{i}", use_container_width=True, disabled=not upload.get("processed", False)):
                            st.session_state.viewing_upload_idx = i
                            render_slide_viewer_modal()
                        if btn_cols[2].button("🗑️ Delete", key=f"del_{i}", use_container_width=True):
                            st.session_state.uploads.pop(i)
                            st.rerun()
    
    with lessons_tab:
        render_lessons_tab()

    with chat_tab:
        render_chat_tab()

    with quizzes_tab:
        render_quizzes_tab()

    with flashcards_tab:
        render_flashcards_tab()

    with settings_tab:
        render_settings_tab()


# --- APP ENTRY POINT ---
if __name__ == "__main__":
    # Ensure all backend functions from previous steps are defined above this call
    render_main_ui()

# --------------------------
# --------------------------
# Lessons, Chat Q&A, Quizzes, Flashcards, Export, Progress, Settings
# Unified robust replacement block
# --------------------------
# --------------------------
# --------------------------------------------------------------------------------
# Replacement UI Block - Lessons, Chat, Quizzes, and other tabs
# This is a complete, modular, and robust UI implementation.
# It assumes your backend functions (e.g., generate_multilevel_lesson,
# call_openrouter_chat, etc.) are defined elsewhere in the script.
# --------------------------------------------------------------------------------

# --- UI HELPER FUNCTIONS ---

def render_upload_selector(context_key: str) -> Optional[Dict]:
    """Renders a selectbox for choosing an upload and returns the selected object."""
    uploads = st.session_state.get("uploads", [])
    if not uploads:
        st.info("No documents have been uploaded yet. Please go to the 'Upload & Process' tab.")
        return None

    # Filter for uploads that have been processed and indexed for full functionality
    processed_uploads = [u for u in uploads if u.get("index_built")]
    if not processed_uploads:
        st.warning("No documents have been processed and indexed. Please build an index in the 'Upload & Process' tab to enable this feature.")
        return None

    upload_options = {u["id"]: u["filename"] for u in processed_uploads}
    selected_id = st.selectbox(
        "Select a processed document to work with:",
        options=list(upload_options.keys()),
        format_func=lambda k: upload_options.get(k, "Unknown"),
        key=f"selector_{context_key}"
    )
    return next((u for u in processed_uploads if u["id"] == selected_id), None)

def build_context_from_selection(upload: Dict, context_mode: str, slide_idx: int) -> str:
    """Builds the text context for LLM generation based on the user's choice."""
    if context_mode == "Indexed search":
        if not upload.get("index_built", False):
            st.warning("Index is not built. Falling back to 'Whole document' context.")
            return "\n\n".join(s.get("text", "") for s in upload.get("slides", []))

        chunks = upload.get("chunks", [])
        slides = upload.get("slides", [])
        
        # Determine the seed text for the search query
        seed_text = ""
        if slides:
             # Find the text of the specified slide to guide the search
             seed_text = next((s.get("text", "") for s in slides if s.get("index") == slide_idx), chunks[0] if chunks else "")
        
        if not seed_text:
             return "\n\n".join(s.get("text", "") for s in slides) # Fallback if no seed text found

        with st.spinner("Searching for relevant context..."):
            try:
                model = get_embedding_model()
                query_embedding = embed_texts([seed_text], model)
                distances, indices = upload["index"].search(query_embedding, k=TOP_K)

                found_chunks = [chunks[i] for i in indices[0] if i != -1 and 0 <= i < len(chunks)]
                return "\n\n---\n\n".join(found_chunks)
            except Exception as e:
                st.error(f"Index search failed: {e}. Using the whole document as a fallback.")
                return "\n\n".join(s.get("text", "") for s in slides)

    elif context_mode == "Specific slide":
        return next((s.get("text", "") for s in upload.get("slides", []) if s.get("index") == slide_idx), "")
    else:  # "Whole document"
        return "\n\n".join(s.get("text", "") for s in upload.get("slides", []))


# --- TAB-SPECIFIC RENDER FUNCTIONS ---

def render_lessons_tab(active: bool):
    """Renders the UI for the Lessons tab."""
    if not active: return
    st.header("📚 Generate Custom Lessons")
    upload = render_upload_selector("lessons")
    if not upload: return

    slides = upload.get("slides", [])
    max_slide_idx = len(slides) - 1 if slides else 0

    st.markdown("##### Lesson Configuration")
    c1, c2 = st.columns(2)
    context_mode = c1.radio(
        "Source of Context",
        ["Indexed search", "Whole document", "Specific slide"],
        key="lessons_context_mode",
        help="Indexed search is fastest and most relevant. 'Whole document' may be slow."
    )
    slide_idx = c2.number_input(
        "Focus Slide/Page Number (1-based)",
        min_value=1,
        max_value=max(1, max_slide_idx + 1),
        value=1,
        key="lessons_slide_idx",
        help="Select the slide to use as the main topic for the lesson."
    ) - 1  # Adjust to 0-based index

    deep_lesson = st.checkbox("Produce a more detailed, in-depth lesson", key="lessons_deep")

    if st.button("✨ Generate Lesson", use_container_width=True, type="primary"):
        context = build_context_from_selection(upload, context_mode, slide_idx)
        if not context.strip():
            st.error("The selected context is empty. Cannot generate a lesson.")
            return

        with st.spinner("AI Tutor is crafting your lesson... This may take a moment."):
            try:
                # Assuming `generate_deep_lesson` and `generate_multilevel_lesson` are defined elsewhere
                if deep_lesson:
                    lesson_text = generate_deep_lesson(context, context)
                else:
                    lesson_text = generate_multilevel_lesson(context, context)
                st.session_state[f"lesson_{upload['id']}"] = lesson_text
            except Exception as e:
                st.error(f"Failed to generate lesson: {e}")
                logger.exception("Lesson generation failed")

    if f"lesson_{upload['id']}" in st.session_state:
        st.markdown("---")
        st.subheader("Your Generated Lesson")
        st.markdown(st.session_state[f"lesson_{upload['id']}"])


def render_chat_tab(active: bool):
    """Renders the UI for the Chat Q&A tab."""
    if not active: return
    st.header("💬 Chat with Your Documents")
    upload = render_upload_selector("chat")
    if not upload: return

    chat_history_key = f"chat_history_{upload['id']}"
    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []

    for message in st.session_state[chat_history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask a question about {upload['filename']}..."):
        st.session_state[chat_history_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and formulating an answer..."):
                try:
                    # Assuming `answer_question_with_rag` is defined elsewhere
                    response = answer_question_with_rag(prompt, [upload])
                    st.markdown(response)
                    st.session_state[chat_history_key].append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

def render_quizzes_tab(active: bool):
    """Renders the UI for the Quizzes tab."""
    if not active: return
    st.header("📝 Auto-Generate Quizzes")
    upload = render_upload_selector("quizzes")
    if not upload: return

    if st.button("Generate 5-Question Quiz", use_container_width=True, type="primary"):
        with st.spinner("AI is creating your quiz..."):
            try:
                # Quizzes benefit from the full context of the document
                full_text_context = "\n\n".join(s.get("text", "") for s in upload.get("slides", []))
                # Assuming `generate_mcq_set_from_text` is defined elsewhere
                mcqs = generate_mcq_set_from_text(full_text_context, qcount=5)
                st.session_state[f"quiz_{upload['id']}"] = mcqs if mcqs else []
            except Exception as e:
                st.error(f"Failed to generate quiz: {e}")
                st.session_state[f"quiz_{upload['id']}"] = []

    quiz_key = f"quiz_{upload['id']}"
    if st.session_state.get(quiz_key):
        st.markdown("---")
        st.subheader("Take the Quiz!")
        
        with st.form(key=f"quiz_form_{upload['id']}"):
            user_answers = []
            for i, q in enumerate(st.session_state[quiz_key]):
                st.markdown(f"**Question {i+1}:** {q['question']}")
                answer = st.radio(f"Options for Q{i+1}", q["options"], key=f"q_{i}", index=None, label_visibility="collapsed")
                user_answers.append(answer)
            
            submitted = st.form_submit_button("Submit Answers")

        if submitted:
            score = 0
            st.markdown("---")
            st.subheader("Quiz Results")
            for i, q in enumerate(st.session_state[quiz_key]):
                correct_answer = q["options"][q["answer_index"]]
                user_answer = user_answers[i]
                if user_answer == correct_answer:
                    score += 1
                    st.success(f"**Question {i+1}: Correct!**")
                else:
                    st.error(f"**Question {i+1}: Incorrect.** The correct answer was: **{correct_answer}**")
            st.metric(label="Your Final Score", value=f"{score} / {len(st.session_state[quiz_key])}")


# --- Main UI Router ---
# This block replaces the entire `if/elif active_tab == ...` structure in your original code.
# It assumes `active_tab` is managed by your app's navigation logic (e.g., st.tabs).

active_tab_name = st.session_state.get("active_tab", "Home")

# Render the specific UI for each tab based on the active tab's name
render_lessons_tab(active_tab_name == "Lessons")
render_chat_tab(active_tab_name == "Chat Q&A")
render_quizzes_tab(active_tab_name == "Quizzes")

# The following are placeholders to ensure the block is complete, as in your original file structure.
if active_tab_name == "Flashcards":
    st.header("🃏 Flashcards & Spaced Repetition")
    st.info("The fully interactive Flashcards tab is available in the complete application UI.")

if active_tab_name == "Export":
    st.header("📤 Export")
    st.info("The Export tab is available in the complete application UI.")

if active_tab_name == "Progress":
    st.header("📊 Progress")
    st.info("The Progress tab is available in the complete application UI.")

if active_tab_name == "Settings":
    st.header("⚙️ Settings")
    st.info("The Settings tab is available in the complete application UI.")


# --------------------------------------------------------------------------------
# Replacement UI Block - Flashcards, Export, Progress, and Settings Tabs
# This is a complete, modular, and robust UI implementation.
# It assumes your backend functions (e.g., generate_flashcards_from_text,
# sm2_update_card, anki_export_tsv, etc.) are defined elsewhere in the script.
# --------------------------------------------------------------------------------

# --- UI HELPER FUNCTIONS ---

def render_upload_selector(context_key: str, anki_export: bool = False) -> Optional[Dict]:
    """Renders a selectbox for choosing an upload and returns the selected object."""
    uploads = st.session_state.get("uploads", [])
    if not uploads:
        st.info("No documents have been uploaded yet. Please go to the 'Upload & Process' tab.")
        return None

    # For most tabs, we require a processed index. For exports, any upload is fine.
    if not anki_export:
        valid_uploads = [u for u in uploads if u.get("processed")]
        if not valid_uploads:
            st.warning("No documents have been processed. Please use the 'Process' button in the 'Upload & Process' tab first.")
            return None
    else:
        valid_uploads = uploads

    upload_options = {u["id"]: u["filename"] for u in valid_uploads}
    selected_id = st.selectbox(
        "Select a document to work with:",
        options=list(upload_options.keys()),
        format_func=lambda k: upload_options.get(k, "Unknown"),
        key=f"selector_{context_key}"
    )
    return next((u for u in valid_uploads if u["id"] == selected_id), None)

# --- TAB-SPECIFIC RENDER FUNCTIONS ---

def render_flashcards_tab(active: bool):
    """Renders the UI for the Flashcards and Spaced Repetition tab."""
    if not active: return
    st.header("🃏 Flashcards & Spaced Repetition")
    upload = render_upload_selector("flashcards")
    if not upload: return

    c1, c2 = st.columns([1, 1])

    # --- Generation Column ---
    with c1:
        st.subheader("Generate New Cards")
        generation_mode = st.radio(
            "Generate from:",
            ["Entire Document", "Specific Slide/Page"],
            key=f"fc_gen_mode_{upload['id']}"
        )
        context_text = ""
        if generation_mode == "Specific Slide/Page":
            slides = upload.get("slides_data", [])
            max_slide_idx = len(slides) - 1 if slides else 0
            slide_num = st.number_input(
                "Slide/Page Number (1-based)",
                min_value=1, max_value=max(1, max_slide_idx + 1), value=1,
                key=f"fc_slide_num_{upload['id']}"
            )
            slide_idx = slide_num - 1
            if 0 <= slide_idx < len(slides):
                slide = slides[slide_idx]
                context_text = (slide.get('text', '') + "\n" + slide.get('ocr_text', '')).strip()
        else:
            context_text = upload.get("full_text", "")

        if st.button("Generate & Save Flashcards", use_container_width=True, type="primary"):
            if not context_text.strip():
                st.error("The selected source text is empty. Cannot generate flashcards.")
            else:
                with st.spinner("AI is creating flashcards..."):
                    try:
                        cards = generate_flashcards_from_text(context_text, n=20)
                        if not cards:
                            st.warning("The AI did not generate any flashcards from this text.")
                        else:
                            add_flashcards_to_db(upload, cards)
                            st.success(f"Successfully generated and saved {len(cards)} new flashcards!")
                    except Exception as e:
                        st.error(f"Failed to generate or save flashcards: {e}")
                        logger.exception("Flashcard generation failed")

    # --- Practice Column ---
    with c2:
        st.subheader("Practice Due Cards")
        if st.button("Load Due Cards for Practice", use_container_width=True):
            st.session_state.due_cards = get_due_flashcards(upload)
            st.session_state.current_card_idx = 0
            if not st.session_state.due_cards:
                st.info("🎉 No cards are due for review for this document!")
            st.rerun()

    # --- Practice UI ---
    if st.session_state.get("due_cards"):
        render_flashcard_practice_ui()

def render_flashcard_practice_ui():
    """Renders the interactive UI for a flashcard review session."""
    st.markdown("---")
    idx = st.session_state.current_card_idx
    cards = st.session_state.due_cards

    if idx >= len(cards):
        st.success("✨ Session Complete! You've reviewed all due cards.")
        st.balloons()
        st.session_state.due_cards = []
        return

    card = cards[idx]
    st.markdown(f"##### Card {idx + 1} of {len(cards)}")
    
    with st.container(border=True):
        st.markdown(f"**Question:**\n> {card['question']}")
        
        show_answer_key = f"show_answer_{card['id']}"
        if show_answer_key not in st.session_state:
            st.session_state[show_answer_key] = False

        if not st.session_state[show_answer_key]:
            if st.button("Show Answer", key=f"show_btn_{card['id']}", use_container_width=True):
                st.session_state[show_answer_key] = True
                st.rerun()
        else:
            st.markdown("---")
            st.markdown(f"**Answer:**\n> {card['answer']}")
            st.markdown("**How well did you recall this?**")
            
            cols = st.columns(4)
            quality_map = {"Again (Incorrect)": 1, "Hard": 3, "Good": 4, "Easy": 5}
            for i, (label, quality) in enumerate(quality_map.items()):
                if cols[i].button(label, key=f"{label}_{card['id']}", use_container_width=True):
                    try:
                        update_flashcard_review(card, quality)
                        st.session_state.current_card_idx += 1
                        st.session_state[show_answer_key] = False  # Reset for the next card
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update card: {e}")

def render_export_tab(active: bool):
    """Renders the UI for the Export tab."""
    if not active: return
    st.header("📤 Export Data")
    upload = render_upload_selector("export", anki_export=True) # Anki export can work on any upload
    if not upload: return

    st.subheader(f"Exporting from: {upload['filename']}")

    # --- Anki Flashcard Export ---
    st.markdown("##### Anki Deck Export")
    if st.button("Export Flashcards for Anki (.tsv)", use_container_width=True):
        try:
            result = anki_export_tsv(upload['db_id'], get_db_connection())
            if result:
                filename, data = result
                st.download_button(
                    label="📥 Download Anki File",
                    data=data,
                    file_name=filename,
                    mime="text/tab-separated-values"
                )
            else:
                st.warning("No flashcards found in the database for this document to export.")
        except Exception as e:
            st.error(f"Anki export failed: {e}")

    # --- Raw Data Export ---
    st.markdown("##### Raw Data Export")
    if st.button("Export Extracted Content (.json)", use_container_width=True):
        try:
            json_data = json.dumps(upload.get("slides_data", []), indent=2).encode('utf-8')
            st.download_button(
                label="📥 Download JSON File",
                data=json_data,
                file_name=f"{upload['filename']}_extracted.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"JSON export failed: {e}")

def render_progress_tab(active: bool):
    """Renders the UI for the Progress tab, showing DB stats."""
    if not active: return
    st.header("📊 Progress & Analytics")

    try:
        conn = get_db_connection()
        uploads_count = conn.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
        flashcards_count = conn.execute("SELECT COUNT(*) FROM flashcards").fetchone()[0]
        quizzes_count = conn.execute("SELECT COUNT(*) FROM quizzes").fetchone()[0]
        
        now = int(time.time())
        due_cards_count = conn.execute("SELECT COUNT(*) FROM flashcards WHERE next_review <= ?", (now,)).fetchone()[0]

        st.metric(label="Total Documents Uploaded", value=uploads_count)
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Total Flashcards Created", value=flashcards_count)
        c2.metric(label="Flashcards Due for Review", value=due_cards_count)
        c3.metric(label="Total Quizzes Created", value=quizzes_count)

    except Exception as e:
        st.error(f"Could not load progress from the database: {e}")

def render_settings_tab(active: bool):
    """Renders the UI for the Settings tab."""
    if not active: return
    st.header("⚙️ Settings & Diagnostics")

    with st.container(border=True):
        st.subheader("API Configuration")
        api_key = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.OPENROUTER_API_KEY,
            type="password",
            help="Your API key is stored temporarily in the session state."
        )
        if api_key != st.session_state.OPENROUTER_API_KEY:
            st.session_state.OPENROUTER_API_KEY = api_key
            st.success("API Key updated for this session.")

    with st.container(border=True):
        st.subheader("Model & Search Configuration")
        emb_model = st.text_input("Embedding Model Name", value=EMBEDDING_MODEL_NAME)
        top_k = st.number_input("Search Top-K", min_value=1, max_value=20, value=TOP_K, help="Number of relevant chunks to retrieve for context.")

        if st.button("Save Model Settings"):
            # These would typically be saved to a config or session state
            st.success("Settings saved for this session.")
            # In a real app, you would update st.session_state here.
            st.session_state.EMBEDDING_MODEL_NAME = emb_model
            st.session_state.TOP_K = top_k

    with st.expander("System Diagnostics"):
        st.write({
            "FAISS Available": _HAS_FAISS,
            "PyMuPDF (PDF Parsing) Available": _HAS_PYMUPDF,
            "EasyOCR (Image Text) Available": _HAS_EASYOCR,
            "Sentence Transformers (Embeddings) Available": _HAS_SENTENCE_TRANSFORMERS,
            "gTTS (Text-to-Speech) Available": _HAS_GTTS,
        })

# --- Main UI Router ---
# This block replaces the entire `if/elif active_tab == ...` structure.
active_tab_name = st.session_state.get("active_tab", "Home")

render_flashcards_tab(active_tab_name == "Flashcards")
render_export_tab(active_tab_name == "Export")
render_progress_tab(active_tab_name == "Progress")
render_settings_tab(active_tab_name == "Settings")
