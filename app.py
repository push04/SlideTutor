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
/* ----------------------------------------------------------
   SlideTutor — Professional UI theme
   - Modern, accessible dark theme with careful spacing, focus states,
     responsive layouts, and component utility classes.
   - Drop this string into your app and apply via st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
   ---------------------------------------------------------- */

/* Import a reliable variable font (Inter) for legible UI text */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root{
  /* Color palette (dark theme) */
  --bg: #061227;            /* page background */
  --bg-2: #07162b;          /* subtle panel background */
  --card: rgba(255,255,255,0.03);
  --muted: #9AA6B2;
  --text: #E6F0FA;
  --accent: #6C5CE7;        /* primary gradient start */
  --accent-2: #4b2bd0;      /* primary gradient end */
  --accent-3: #2AB7A9;      /* secondary accent (eg. success highlights) */
  --success: #2AB76A;
  --danger: #FF6B6B;
  --glass: rgba(255,255,255,0.025);
  --radius: 12px;
  --radius-sm: 8px;
  --shadow-1: 0 6px 20px rgba(2,6,12,0.6);
  --shadow-2: 0 4px 12px rgba(2,8,20,0.45);
  --spacing: 16px;
  --max-width: 1200px;
  --accent-gradient: linear-gradient(90deg, var(--accent), var(--accent-2));
  --transition-fast: 150ms ease;
  --transition-medium: 240ms ease;
}

/* Base + layout */
html, body, .stApp {
  height: 100%;
  background: radial-gradient(1200px 600px at 10% 10%, rgba(92,72,172,0.08), transparent 10%),
              linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  line-height: 1.45;
  -webkit-tap-highlight-color: transparent;
}

/* Container padding tuned for a clean, airy feel */
.block-container {
  padding: 28px 32px !important;
  max-width: var(--max-width);
  margin: 0 auto;
}

/* Card styles */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.018), rgba(255,255,255,0.008));
  border-radius: var(--radius);
  padding: calc(var(--spacing) * 0.875);
  margin-bottom: 18px;
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: var(--shadow-1);
  transition: transform var(--transition-fast), box-shadow var(--transition-fast);
}
.card:hover { transform: translateY(-4px); box-shadow: var(--shadow-2); }

/* Card header / meta */
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.card-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 0; }
.card-sub { font-size: 0.92rem; color: var(--muted); margin: 0; }

/* Elegant app header (logo + title) */
.app-header {
  display:flex;
  align-items:center;
  gap:14px;
  margin-bottom:18px;
}
.app-logo {
  width:48px; height:48px;
  display:inline-flex; align-items:center; justify-content:center;
  border-radius:10px;
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
  box-shadow: 0 6px 18px rgba(11,12,30,0.5);
  font-size:1.4rem;
}
.app-title { font-size:1.4rem; font-weight:700; margin:0; }
.app-subtitle { color:var(--muted); font-size:0.95rem; margin:0; }

/* Sidebar refinements (Streamlit's sidebar) */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.006));
  border-right: 1px solid rgba(255,255,255,0.03);
  padding-top: 20px;
}
[data-testid="stSidebar"] .stButton>button { width:100%; }

/* Tabs / navigation */
.css-1v3fvcr, .stTabs { /* fallback for older streamlit classes */
  display:block;
}

/* File uploader / preview */
.uploader {
  border: 1px dashed rgba(255,255,255,0.04);
  padding: 14px;
  border-radius: 10px;
  background: linear-gradient(180deg, rgba(255,255,255,0.01), transparent);
}
.uploader .hint { color: var(--muted); font-size:0.95rem; }

/* Compact file list rows */
.file-row {
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding:10px; border-radius:8px; background:var(--glass); margin-bottom:8px;
}
.file-meta { display:flex; align-items:center; gap:10px; }
.file-name { font-weight:600; color:var(--text); }
.file-small { color:var(--muted); font-size:0.92rem; }

/* Slide preview block */
.preview-slide {
  display:flex; gap:12px; align-items:flex-start;
  background: rgba(255,255,255,0.015);
  border-radius: var(--radius-sm);
  padding: 10px;
  border: 1px solid rgba(255,255,255,0.02);
}
.slide-thumb {
  width: 120px; height: 80px; border-radius:8px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  display:flex; align-items:center; justify-content:center; color:var(--muted); font-size:0.92rem; overflow:hidden;
  flex-shrink:0;
}
.slide-text { font-size:0.95rem; color:var(--text); }

/* Buttons: primary / secondary / ghost */
.stButton>button {
  border-radius: 10px !important;
  padding: 8px 14px !important;
  background: var(--accent-gradient) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 6px 18px rgba(77,92,230,0.12);
  transition: transform var(--transition-fast), box-shadow var(--transition-fast), opacity var(--transition-fast);
}
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(77,92,230,0.18); }
.stButton>button:active { transform: translateY(0); }
.stButton>button[disabled] { opacity: 0.6; transform:none; box-shadow:none; cursor:not-allowed; }

/* Utility variants */
.btn-secondary {
  background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.04);
  box-shadow: none !important;
}
.btn-ghost {
  background: transparent !important;
  color: var(--muted) !important;
  border: 1px solid rgba(255,255,255,0.02) !important;
}

/* Badges and small labels */
.badge {
  display:inline-block; padding:4px 8px; font-size:0.82rem; border-radius:999px; background:rgba(255,255,255,0.02); color:var(--muted);
}

/* Muted & helper text */
.small-muted { color: var(--muted); font-size: 0.92rem; }

/* Input / textarea styling */
textarea, input, .stTextInput, .stTextArea {
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,0.03) !important;
  background: rgba(255,255,255,0.01) !important;
  padding: 10px !important;
  color: var(--text) !important;
  transition: box-shadow var(--transition-fast);
}
textarea:focus, input:focus, .stTextInput:focus, .stTextArea:focus {
  outline: none !important;
  box-shadow: 0 6px 20px rgba(76,84,255,0.12);
  border-color: rgba(108,92,231,0.22) !important;
}

/* Chat bubbles */
.chat-container { display:flex; flex-direction:column; gap:10px; }
.chat-message { max-width:80%; padding:12px 14px; border-radius:12px; line-height:1.4; }
.chat-user { align-self:flex-end; background: linear-gradient(90deg,#2b2b2b, #1f2430); color:#fff; border: 1px solid rgba(255,255,255,0.02); }
.chat-assistant { align-self:flex-start; background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); color:var(--text); border:1px solid rgba(255,255,255,0.02); }

/* Flashcards */
.flashcard { background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.008)); padding: 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.02); box-shadow: var(--shadow-2); }
.flashcard-controls { display:flex; gap:8px; margin-top:10px; }

/* Progress / loaders */
.progress-wrap { background: rgba(255,255,255,0.02); border-radius: 10px; padding: 6px; }
.progress-bar { height:10px; border-radius:10px; background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02)); overflow:hidden; }
.progress-bar > span { display:block; height:100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); border-radius:10px; }

/* Small utilities */
.kbd { background: rgba(255,255,255,0.02); border-radius:6px; padding:3px 6px; font-weight:600; font-size:0.85rem; color:var(--muted); border:1px solid rgba(255,255,255,0.02); }
.center { display:flex; justify-content:center; align-items:center; }

/* Focus-visible for accessibility */
:focus, .stButton>button:focus, .stTextInput:focus, .stTextArea:focus {
  outline: 3px solid rgba(108,92,231,0.12);
  outline-offset: 2px;
}
:focus:not(:focus-visible) { outline: none; }

/* Low-motion for accessibility */
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}

/* Responsive tweaks */
@media (max-width: 900px) {
  .slide-thumb { width: 90px; height: 64px; }
  .block-container { padding: 18px 16px !important; }
  .app-title { font-size: 1.12rem; }
}

/* Light theme override (optional) - apply .light-theme on body */
.light-theme {
  --bg: #f6f7fb;
  --bg-2: #eef2fb;
  --card: rgba(12,18,30,0.02);
  --text: #071225;
  --muted: #4b5563;
  --accent: #3366ff;
  --accent-2: #4b2bd0;
  background: linear-gradient(180deg, #f6f8fc 0%, #eef2fb 100%);
}
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
