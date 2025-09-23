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

# prefer Streamlit secrets, fall back to environment variable
DEFAULT_OPENROUTER_API_KEY = (
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
    # resolve API key: session -> secrets/env
    try:
        api_key = st.session_state.get("OPENROUTER_API_KEY") or DEFAULT_OPENROUTER_API_KEY
    except Exception:
        api_key = DEFAULT_OPENROUTER_API_KEY

    if not api_key:
        raise RuntimeError("OpenRouter API key not configured. Set OPENROUTER_API_KEY in Streamlit secrets or paste into Settings.")

    if not _HAS_REQUESTS or requests is None:
        raise RuntimeError("requests library not available. Install 'requests' to call the LLM API.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        # prefer chat messages format (OpenRouter compatibility); keep simple chat shape
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": int(max_tokens or 1200),
        "temperature": float(temperature or 0.1)
    }

    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        logger.exception("Network/API error calling OpenRouter: %s", e)
        raw = ""
        try:
            raw = getattr(e, "response", None) and e.response.text or ""
        except Exception:
            raw = ""
        raise RuntimeError(f"LLM API call failed: {e}\n{raw}") from e

    try:
        data = resp.json()
    except Exception:
        logger.exception("Failed to parse JSON from LLM response")
        raise RuntimeError("Failed to decode LLM response as JSON.")

    # Extract content from different possible shapes
    content = None
    try:
        choices = data.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                # typical chat shape: message.content
                msg = first.get("message") or {}
                content = msg.get("content") or first.get("text") or first.get("message", {}).get("content")
        if not content:
            # fallback shapes
            if "output" in data:
                content = data["output"]
            elif "result" in data:
                content = data["result"]
            elif "text" in data:
                content = data["text"]
    except Exception:
        content = None

    if not content:
        logger.error("LLM missing expected fields; response truncated: %s", str(data)[:2000])
        raise RuntimeError("LLM response missing expected content. See logs.")

    # ensure string
    if not isinstance(content, str):
        try:
            content = json.dumps(content)
        except Exception:
            content = str(content)

    return content.strip()


# ----------------------------
# Content generation wrappers (enhanced, more robust prompts)
# ----------------------------

# NOTE:
# - Prompts request both a human-readable Markdown section (for display)
#   and a strict JSON payload wrapped in triple-backticks (```json ... ```)
#   so your UI can render the Markdown and your code can reliably parse the JSON.
# - Each prompt gives precise output schema, length limits, and safety instructions
#   (e.g., say "CANNOT_ANSWER" if the information isn't present).
# - Keep temperature low for deterministic outputs when you call the LLM.

PROMPT_LESSON_MULTILEVEL = (
    "You are an experienced subject-matter teacher and curriculum designer.\n\n"
    "TASK: Create a *multi-level lesson* from the provided TEXT. Produce TWO output parts:\n\n"
    "1) A human-friendly MARKDOWN lesson (suitable for immediate display). Structure it clearly with headings\n"
    "   and short paragraphs. Include examples and tips so a student can learn from it.\n\n"
    "2) A strict JSON summary wrapped in triple-backticks labeled as ```json``` (so it can be parsed).\n"
    "   The JSON object MUST follow this schema exactly:\n"
    "   {\n"
    "     \"title\": string,                # short title (<= 80 chars)\n"
    "     \"levels\": {                     # object with 3 keys: beginner, intermediate, advanced\n"
    "        \"Beginner\": {\n"
    "            \"explanation\": string,  # concise explanation (50-300 chars)\n"
    "            \"worked_example\": string, # one worked example (<= 400 chars)\n"
    "            \"tips\": [string,...]    # 2-3 short actionable tips\n"
    "        },\n"
    "        \"Intermediate\": { ... },\n"
    "        \"Advanced\": { ... }\n"
    "     },\n"
    "     \"short_quiz\": [                 # exactly 3 short quiz items (Q/A pairs)\n"
    "        {\"question\": string, \"answer\": string}\n"
    "     ]\n"
    "   }\n\n"
    "REQUIREMENTS / GUIDELINES:\n"
    "- Use only the provided TEXT as your source. Do NOT introduce outside facts.\n"
    "- If the TEXT does not contain enough information, for any missing piece return the string \"CANNOT_ANSWER\" for that field.\n"
    "- Keep each explanation concise and focused (see recommended char limits above).\n"
    "- The Markdown section should mirror the JSON structure with clear H2/H3 headings.\n"
    "- Language: same language as the TEXT (default English). Tone: clear, teacherly, friendly.\n"
    "- Avoid lists larger than 5 items. Keep examples minimal and directly relevant.\n\n"
    "OUTPUT: First the Markdown lesson, then the JSON block. Do NOT output any other stray text."
)

PROMPT_MCQ_JSON = (
    "You are an AI that reliably generates high-quality multiple-choice questions (MCQs).\n\n"
    "TASK: From the provided TEXT, create the requested number of MCQs. Reply ONLY with a valid JSON array\n"
    "wrapped in triple-backticks labeled ```json``` so it can be parsed. The array should contain objects with the\n"
    "following keys (strict):\n\n"
    "  {\n"
    "    \"question\": string,             # concise stem (<= 200 chars)\n"
    "    \"options\": [string, string, string, string],  # exactly 4 options, unique, each <= 120 chars\n"
    "    \"answer_index\": integer,       # 0-based index of the correct option\n"
    "    \"explanation\": string,         # short explanation of why the correct option is right (<= 200 chars)\n"
    "    \"distractor_rationale\": [string, string, string], # brief note why each incorrect option is plausible (3 items)\n"
    "    \"source\": string | null        # optional: slide/page reference or 'CANNOT_ANSWER'\n"
    "  }\n\n"
    "REQUIREMENTS / GUIDELINES:\n"
    "- Produce exactly the number of MCQs requested. If you cannot reach that count, return as many as possible and add a top-level\n"
    "  JSON field `\"note\":\"GENERATED_N_LESS_THAN_REQUESTED\"` (where N is how many you produced).\n"
    "- Use only the TEXT as your source. Do NOT invent facts. If the answer cannot be supported by the TEXT, mark the question's\n"
    "  `explanation` as \"CANNOT_ANSWER\" and prefer NOT to include that question unless you can create plausible but text-grounded\n"
    "  items.\n"
    "- Ensure options are similar length and avoid giving clues in wording (no 'all of the above'/'none of the above').\n"
    "- Try to vary difficulty: label at least one question as 'easy', one 'medium', and one 'hard' in the explanation if multiple are\n"
    "  requested.\n"
    "- Do not include any additional fields outside the schema. The response MUST be valid JSON.\n\n"
    "OUTPUT: Only the JSON array in a ```json``` block."
)

PROMPT_FLASHCARDS_JSON = (
    "You are an AI that extracts clear, concise flashcards from the provided TEXT.\n\n"
    "TASK: Return a JSON array (wrapped in ```json```) of flashcard objects with this exact schema:\n\n"
    "  [\n"
    "    {\n"
    "      \"question\": string,   # short, focused prompt (<= 160 chars)\n"
    "      \"answer\": string,     # concise correct answer (<= 300 chars)\n"
    "      \"source\": string|null,# optional slide/page index or 'CANNOT_ANSWER'\n"
    "      \"tags\": [string,...]  # optional short tags (e.g., ['formula','definition'])\n"
    "    }\n"
    "  ]\n\n"
    "REQUIREMENTS / GUIDELINES:\n"
    "- Prefer atomic Q/A pairs (one fact or concept per card).\n"
    "- Avoid verbatim copying of long paragraphs; summarize when possible.\n"
    "- If the TEXT lacks enough info, produce as many valid flashcards as possible and stop.\n"
    "- No more than 50 flashcards at once unless explicitly requested.\n"
    "- Language should match the input TEXT.\n\n"
    "OUTPUT: Only the JSON array wrapped in ```json```; do not include extra commentary."
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
:root{
  /* theme tokens */
  --bg-0: #051022;
  --bg-1: #061428;
  --panel: rgba(255,255,255,0.02);
  --muted: #9AA6B2;
  --text: #E6F0FA;
  --accent-1: #6C5CE7;
  --accent-2: #4b2bd0;
  --radius: 12px;
  --shadow-soft: 0 8px 36px rgba(2,6,12,0.55);
  --max-width: 1180px;
  --top-offset: 84px;   /* safe top offset to avoid Streamlit top bar overlap */
  --side-gutter: 28px;  /* horizontal gutter on desktop */
  --bg-panel: rgba(255,255,255,0.012);
}

/* GLOBAL */
html, body, .stApp {
  min-height: 100%;
  background: linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
  color: var(--text);
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  line-height: 1.45;
  box-sizing: border-box;
  overflow-x: hidden; /* prevent horizontal scroll caused by overflowing elements */
}

/* Main container: horizontal gutters + centered content */
.stApp .block-container {
  padding: calc(var(--top-offset)) var(--side-gutter) 36px !important;
  max-width: var(--max-width);
  margin-left: auto;
  margin-right: auto;
  width: calc(100% - (var(--side-gutter) * 2));
  transition: width 160ms ease;
  box-sizing: border-box;
}

/* larger gutters for very wide screens to keep content centered and readable */
@media (min-width: 1600px) {
  :root { --side-gutter: 96px; --max-width: 1400px; }
  .stApp .block-container { padding-left: var(--side-gutter) !important; padding-right: var(--side-gutter) !important; }
}

/* CARD (reusable) */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.014), rgba(255,255,255,0.008));
  border-radius: var(--radius);
  padding: 18px;
  margin-bottom: 18px;
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: var(--shadow-soft);
  overflow: visible;
  transition: transform 160ms ease, box-shadow 160ms ease;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 18px 48px rgba(2,6,12,0.55); }

/* HEADER (global header rendered once by render_header()) */
.app-header {
  display:flex;
  align-items:center;
  gap:14px;
  margin-bottom:12px;
  flex-wrap:wrap;
  position:relative;
  z-index: 50;
}
.app-logo {
  width:56px;
  height:56px;
  border-radius:12px;
  display:inline-grid;
  place-items:center;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  font-size:1.4rem;
}
.app-title {
  font-size:1.6rem;
  font-weight:700;
  margin:0;
  line-height:1.12;
  max-width: 68%;
  overflow-wrap:break-word;
  word-break:break-word;
}
.app-sub {
  color:var(--muted);
  margin:0;
  font-size:0.98rem;
  max-width: 68%;
  overflow-wrap:break-word;
}

/* HERO block - balanced horizontal & vertical spacing */
.hero {
  display:flex;
  gap:22px;
  align-items:center;
  padding:20px;
  border-radius:14px;
  background: linear-gradient(90deg, rgba(76,58,199,0.06), rgba(75,43,208,0.03));
  border: 1px solid rgba(255,255,255,0.02);
  align-items:flex-start;
}
.hero-left { flex:1; min-width:280px; }
.hero-right { width:320px; min-width:220px; }

/* HERO content typography */
.hero .heading { font-size:1.8rem; font-weight:700; margin:0 0 8px 0; letter-spacing: -0.02em; }
.hero .sub { color:var(--muted); margin:0 0 12px 0; max-width: 68ch; }

/* Metrics panel inside hero (vertical layout on small screens) */
.metrics {
  display:flex;
  flex-direction:column;
  gap:8px;
  align-items:flex-start;
}
.metric-title { font-size:0.88rem; color:var(--muted); }
.metric-value { font-size:1.6rem; font-weight:700; line-height:1; }

/* Uploader & preview */
.uploader {
  border:1px dashed rgba(255,255,255,0.04);
  padding:12px;
  border-radius:10px;
  background:var(--bg-panel);
  overflow:visible;
}
.file-row {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  padding:10px;
  border-radius:8px;
  background:rgba(255,255,255,0.01);
  margin-bottom:8px;
  word-break: break-word;
}
.file-name { font-weight:600; color:var(--text); }
.file-small { color:var(--muted); font-size:0.92rem; }

/* Slide preview */
.preview-slide {
  display:flex;
  gap:12px;
  align-items:flex-start;
  padding:10px;
  border-radius:8px;
  background:rgba(255,255,255,0.012);
  border:1px solid rgba(255,255,255,0.02);
  overflow:hidden;
}
.slide-thumb {
  width:160px;
  height:92px;
  border-radius:8px;
  display:flex;
  align-items:center;
  justify-content:center;
  color:var(--muted);
  overflow:hidden;
}
.slide-text { font-size:0.95rem; color:var(--text); }

/* Buttons & inputs — polished */
.stButton>button {
  border-radius:12px !important;
  padding:10px 16px !important;
  background: linear-gradient(90deg, var(--accent-1), var(--accent-2)) !important;
  color:#fff !important;
  border:none !important;
  box-shadow:0 10px 28px rgba(77,92,230,0.12);
  white-space: nowrap;
  transition: transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease;
}
.stButton>button:hover { transform: translateY(-3px); box-shadow:0 18px 40px rgba(77,92,230,0.14); }
.stButton>button:active { transform: translateY(-1px); }

/* Inputs */
textarea, input, .stTextInput, .stTextArea {
  border-radius:10px !important;
  border:1px solid rgba(255,255,255,0.03) !important;
  background: rgba(255,255,255,0.008) !important;
  padding:10px !important;
  color:var(--text) !important;
  box-sizing: border-box;
}

/* Small helpers */
.small-muted { color:var(--muted); font-size:0.92rem; }
.kbd { background: rgba(255,255,255,0.02); border-radius:6px; padding:4px 8px; font-weight:600; font-size:0.85rem; color:var(--muted); }

/* Make long text wrap cleanly */
* { word-wrap: break-word; }

/* Horizontal layout helpers (avoid side-cropping) */
.page-row {
  display:flex;
  gap:24px;
  align-items:flex-start;
  width:100%;
  box-sizing:border-box;
}
.page-col { flex:1; min-width:220px; }

/* Card grid: responsive columns */
.grid-4 { display:grid; grid-template-columns: repeat(4, 1fr); gap:16px; }
.grid-3 { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; }
.grid-2 { display:grid; grid-template-columns: repeat(2, 1fr); gap:12px; }
@media (max-width: 1100px) {
  .grid-4, .grid-3 { grid-template-columns: repeat(2, 1fr); }
  .hero-right { display:none; }
  .app-title { font-size:1.35rem; max-width:100%; }
  .app-sub { font-size:0.95rem; max-width:100%; }
  .stApp .block-container { padding: calc(var(--top-offset) - 16px) 20px 20px !important; }
}
@media (max-width: 720px) {
  .grid-2 { grid-template-columns: repeat(1, 1fr); }
  .hero { flex-direction:column; gap:12px; align-items:flex-start; }
  :root { --top-offset: 64px; --side-gutter: 16px; }
  .stApp .block-container { padding: calc(var(--top-offset) - 20px) 14px 14px !important; }
}

/* safety: ensure block-container box-sizing override (avoid collisions) */
.stApp .block-container, .block-container { box-sizing: border-box; }
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
    """
    Clean, non-duplicating home/hero UI:
    - DOES NOT render the global header/logo (render_header() already does that)
    - Uses compact hero, metrics column, CTA buttons, and expanders to avoid congestion
    """
    try:
        uploads = st.session_state.get("uploads", []) or []
        total_uploads = len(uploads)
        total_slides = sum(int(u.get("slide_count", 0) or 0) for u in uploads)
        fc_db = st.session_state.get("flashcards_db", {})
        total_flashcards = sum(len(v) for v in fc_db.values()) if isinstance(fc_db, dict) else 0

        # HERO (compact — no duplicate logo/title)
        st.markdown("<div class='card hero'>", unsafe_allow_html=True)
        left, right = st.columns([2, 1])
        with left:
            st.markdown("<h2 style='margin:0 0 8px 0;'>Learn faster from slides — smart, visual, practical.</h2>", unsafe_allow_html=True)
            st.markdown("<div class='small-muted' style='margin-bottom:12px'>Upload PDFs/PPTX, extract text & images, build semantic indexes for RAG Q&A, auto-generate lessons, quizzes & flashcards, and practice with SM-2.</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1,1,1])
            if c1.button("Upload files", key="home_cta_upload"):
                st.session_state["_navigate_to"] = "Upload"
                safe_rerun()
            if c2.button("Try Chat Q&A", key="home_cta_chat"):
                st.session_state["_navigate_to"] = "Chat Q&A"
                safe_rerun()
            if c3.button("Generate Lesson", key="home_cta_lesson"):
                st.session_state["_navigate_to"] = "Lessons"
                safe_rerun()
        with right:
            # compact metrics (stacked)
            try:
                st.markdown("<div style='display:flex;flex-direction:column;gap:6px;align-items:flex-start'>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.9rem;color:var(--muted)'>Uploads</div><div style='font-size:1.6rem;font-weight:700'>{total_uploads}</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.9rem;color:var(--muted)'>Slides parsed</div><div style='font-size:1.6rem;font-weight:700'>{total_slides}</div>", unsafe_allow_html=True)
                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:0.9rem;color:var(--muted)'>Flashcards</div><div style='font-size:1.6rem;font-weight:700'>{total_flashcards}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                st.metric("Uploads", total_uploads)
                st.metric("Slides parsed", total_slides)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recent uploads expander (hidden by default to reduce congestion)
        with st.expander("Recent uploads & preview", expanded=False):
            if not uploads:
                st.markdown("<div class='small-muted'>No uploads yet — go to Upload tab.</div>", unsafe_allow_html=True)
            else:
                for up in reversed(uploads[-3:]):
                    st.markdown(f"**{up.get('filename','(file)')}** — slides: {up.get('slide_count',0)}")
                    slides = up.get("slides_data") or []
                    if slides:
                        s0 = slides[0]
                        imgs = s0.get("images") or []
                        if imgs:
                            try:
                                st.image(io.BytesIO(imgs[0]), width=260)
                            except Exception:
                                pass
                        if s0.get("text"):
                            st.write(s0.get("text")[:800] + ("..." if len(s0.get("text","")) > 800 else ""))
                    st.markdown("---")

        # Features as a compact grid (collapsed details inside expanders to avoid clutter)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### What you can do")
        cols = st.columns([1,1,1,1])
        cols[0].markdown("**Extract**\n\n<div class='small-muted'>Text, images, OCR from slides</div>", unsafe_allow_html=True)
        cols[1].markdown("**Index**\n\n<div class='small-muted'>Build embeddings for semantic search</div>", unsafe_allow_html=True)
        cols[2].markdown("**Generate**\n\n<div class='small-muted'>Lessons, MCQs, flashcards</div>", unsafe_allow_html=True)
        cols[3].markdown("**Practice**\n\n<div class='small-muted'>SM-2 spaced repetition & Anki export</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Quick actions: act on a selected upload (kept small)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Quick actions")
        if uploads:
            sel_name = st.selectbox("Choose upload", [u["filename"] for u in uploads], key="home_quick_select")
            sel_upload = next((u for u in uploads if u.get("filename") == sel_name), None)
            b1, b2, b3 = st.columns([1,1,1])
            if b1.button("Build Index", key=f"home_build_{_sanitize_key(sel_name)}"):
                try:
                    build_vector_index_safe(sel_upload, force=False)
                    st.success(sel_upload.get("status_msg","Index attempted"))
                    safe_rerun()
                except Exception:
                    st.error("Index build failed. Check logs.")
            if b2.button("Preview slides", key=f"home_preview_{_sanitize_key(sel_name)}"):
                try:
                    if sel_upload:
                        for s in sel_upload.get("slides_data", [])[:4]:
                            st.markdown(f"**Slide {s.get('index',0)+1}**")
                            if s.get("images"):
                                for img in s.get("images")[:1]:
                                    st.image(io.BytesIO(img), width=260)
                            if s.get("text"):
                                st.write(s.get("text")[:600])
                            st.markdown("---")
                    else:
                        st.info("No slides to preview.")
                except Exception:
                    st.error("Preview failed.")
            if b3.button("Generate 10 flashcards", key=f"home_genfc_{_sanitize_key(sel_name)}"):
                try:
                    cards = generate_flashcards_from_text(sel_upload.get("full_text",""), n=10)
                    saved = add_flashcards_to_db_safe(sel_upload, cards)
                    st.success(f"Saved {saved} flashcards")
                except Exception:
                    st.error("Flashcard gen failed.")
        else:
            st.markdown("<div class='small-muted'>No uploads — use the Upload tab to add files.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # small footer hint
        st.markdown("<div class='small-muted' style='margin-top:8px'>Tip: expand 'Recent uploads' to preview slides; use quick actions for common tasks.</div>", unsafe_allow_html=True)

    except Exception as e:
        logger.exception("render_home failed: %s", e)
        st.markdown(f"### {APP_TITLE} — {APP_SUBTITLE}")
        st.markdown("Upload PDFs or PPTX and explore features in the tabs above.")



def render_upload_tab():
    st.header("Upload & Process")
    st.markdown("<div class='card'>Upload PPTX or PDF files. Then process to extract text & OCR, optionally build embeddings (if available).</div>", unsafe_allow_html=True)

    auto_index = st.checkbox("Auto-build index after upload (attempt)", value=True, key="auto_build_index_opt")
    uploaded_files = st.file_uploader("PPTX / PDF — multiple", type=["pptx", "pdf"], accept_multiple_files=True, key="uploader")

    if uploaded_files:
        existing = {u["filename"] for u in st.session_state.uploads}
        for f in uploaded_files:
            if f.name not in existing:
                up = process_new_upload_safe(f)
                st.session_state.uploads.append(up)
                st.success(f"Added: {f.name}")
                # auto-build optionally
                if auto_index:
                    try:
                        build_vector_index_safe(up)
                        st.info(up.get("status_msg", "Index attempted"))
                    except Exception:
                        logger.exception("Auto index failed")
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

            # show file preview (images first)
            if up.get("slides_data"):
                if st.button("Preview (first 10 slides as images)", key=f"preview_img_{_sanitize_key(up['filename'])}"):
                    for s in up.get("slides_data", [])[:10]:
                        st.markdown(f"**Slide {s['index']+1}**")
                        imgs = s.get("images") or []
                        if imgs:
                            # show up to 3 images as thumbnails
                            imgs_to_show = imgs[:3]
                            cols2 = st.columns(len(imgs_to_show))
                            for c_idx, img_bytes in enumerate(imgs_to_show):
                                try:
                                    cols2[c_idx].image(io.BytesIO(img_bytes), width=220, use_container_width=False)
                                except Exception:
                                    try:
                                        cols2[c_idx].write("Image preview not available")
                                    except Exception:
                                        pass
                        else:
                            # fallback to text / OCR
                            if s.get("text"):
                                st.write(s["text"])
                            if s.get("ocr_text"):
                                st.caption("OCR text:")
                                st.write(s["ocr_text"])
                        st.markdown("---")

            # delete action
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

    # select document
    opts = {u["filename"]: idx for idx, u in enumerate(uploads)}
    sel = st.selectbox("Select a processed document", list(opts.keys()))
    up = uploads[opts[sel]]

    # options for generation/display
    col1, col2 = st.columns([1, 1])
    show_markdown = col1.checkbox("Render LLM Markdown output (if present)", value=True)
    show_structured = col2.checkbox("Render structured lesson (parsed JSON)", value=True)

    # interactive controls
    st.markdown("---")
    if st.button("Generate Multi-level Lesson"):
        with st.spinner("Generating lesson..."):
            try:
                raw = generate_multilevel_lesson(up.get("full_text", ""))
            except Exception as e:
                logger.exception("Lesson generation failed: %s", e)
                st.error(f"Lesson generation failed: {e}")
                return

        if not raw or not str(raw).strip():
            st.warning("LLM returned empty output.")
            return

        # Attempt to extract JSON object from the LLM output
        parsed_json = None
        try:
            parsed_json = extract_json_from_text(raw)
        except Exception:
            parsed_json = None

        # Try to isolate the markdown portion (anything outside a ```json``` block or trailing JSON)
        md_part = str(raw)
        try:
            # remove triple-backtick JSON blocks if present
            md_part = re.sub(r'```json[\s\S]*?```', '', md_part, flags=re.IGNORECASE).strip()
            # also remove a trailing raw JSON object if present (fallback)
            md_part = re.sub(r'\n*\{[\s\S]*\}\s*$', '', md_part).strip()
        except Exception:
            md_part = str(raw)

        # Show helpful UI to user
        st.success("Lesson generated (preview below).")

        # Render Markdown part if requested and not empty
        if show_markdown and md_part:
            st.markdown("#### Lesson (LLM markdown preview)")
            try:
                # Use markdown first for nicer formatting, fallback to write
                st.markdown(md_part)
            except Exception:
                st.write(md_part)
            st.markdown("---")

        # If we parsed a JSON object and user wants structured view, render it nicely
        if show_structured and parsed_json and isinstance(parsed_json, dict):
            try:
                # Save parsed JSON in session for later use/export
                st.session_state["last_parsed_lesson"] = parsed_json

                title = parsed_json.get("title") or parsed_json.get("topic") or "Lesson"
                st.markdown(f"### {title}")

                levels = parsed_json.get("levels") or {}
                if isinstance(levels, dict) and levels:
                    # show level cards horizontally where space permits
                    level_keys = list(levels.keys())
                    # create responsive columns (up to 3)
                    ncols = min(3, max(1, len(level_keys)))
                    cols = st.columns(ncols)
                    for i, lvl in enumerate(level_keys):
                        data = levels.get(lvl) or {}
                        c = cols[i % ncols]
                        with c:
                            st.markdown(f"#### {lvl}")
                            exp = data.get("explanation") or data.get("summary") or "CANNOT_ANSWER"
                            st.markdown(f"**Explanation:**\n\n{exp}")
                            we = data.get("worked_example") or "CANNOT_ANSWER"
                            st.markdown(f"**Worked example:**\n\n{we}")
                            tips = data.get("tips") or []
                            if tips and isinstance(tips, (list, tuple)):
                                st.markdown("**Tips:**")
                                for t in tips[:5]:
                                    st.markdown(f"- {t}")
                            else:
                                st.markdown("**Tips:** CANNOT_ANSWER")
                else:
                    st.info("No structured levels found in parsed JSON.")

                # Short quiz rendering
                short_quiz = parsed_json.get("short_quiz") or parsed_json.get("quiz") or []
                if short_quiz and isinstance(short_quiz, list):
                    st.markdown("---")
                    st.markdown("#### Short Quiz")
                    for qi, item in enumerate(short_quiz):
                        q = item.get("question") if isinstance(item, dict) else (item[0] if isinstance(item, (list, tuple)) else None)
                        a = item.get("answer") if isinstance(item, dict) else (item[1] if isinstance(item, (list, tuple)) else None)
                        if not q:
                            continue
                        card_key = f"lesson_quiz_show_{_sanitize_key(str(sel))}_{qi}"
                        st.markdown(f"**Q{qi+1}. {q}**")
                        if st.button("Show Answer", key=card_key):
                            st.markdown(f"**Answer:** {a or 'CANNOT_ANSWER'}")
                        st.markdown("---")
                else:
                    # no quiz found
                    pass

                # allow download of structured JSON
                try:
                    jbytes = json.dumps(parsed_json, indent=2, ensure_ascii=False).encode("utf-8")
                    st.download_button("Download lesson JSON", data=jbytes, file_name=f"lesson_{_sanitize_key(title)[:32]}.json", mime="application/json")
                except Exception:
                    pass

            except Exception as e:
                logger.exception("Rendering parsed lesson failed: %s", e)
                st.warning("Parsed lesson exists but failed to render nicely. Showing raw JSON below.")
                st.json(parsed_json)

        else:
            # If structured view not requested or parsing failed, show safe fallback
            if parsed_json and not show_structured:
                st.info("Lesson JSON parsed but structured view is disabled (enable 'Render structured lesson').")
            elif not parsed_json:
                st.warning("Could not parse structured JSON from the LLM response. Falling back to any markdown output (above) or raw JSON below.")
            # as final fallback, show raw (prettified) JSON, but do not dump inline raw text
            try:
                parsed_try = json.loads(re.sub(r'[\r\n]+$', '', raw)) if isinstance(raw, str) and raw.strip().startswith("{") else None
            except Exception:
                parsed_try = None
            if parsed_try:
                st.json(parsed_try)
            else:
                # last resort: attempt to pretty-print any JSON-like substring
                j = extract_json_from_text(raw)
                if j:
                    st.json(j)
                else:
                    # show a compact preview to avoid cluttering UI with a huge raw blob
                    st.markdown("#### Raw LLM output (preview)")
                    st.text(raw[:4000] + ("..." if len(raw) > 4000 else ""))

    # End generate button handling

    # If user has a previously parsed lesson in session, allow them to re-open it
    if st.session_state.get("last_parsed_lesson") and st.checkbox("Show last parsed lesson from this session", value=False):
        pj = st.session_state.get("last_parsed_lesson")
        try:
            st.markdown(f"### {pj.get('title','Lesson')}")
            st.json(pj)
        except Exception:
            st.write(pj)


def render_chat_tab():
    st.header("Ask Questions (RAG)")
    indexed = [u for u in st.session_state.uploads if u.get("index_built")]
    if not indexed:
        st.warning("No indexed uploads. Build an index on the Upload tab to enable RAG.")
        return

    opts = {u['filename']: u for u in indexed}
    chosen = st.multiselect("Which indexed documents to search (multi-select)", list(opts.keys()), default=list(opts.keys()))
    docs = [opts[n] for n in chosen] if chosen else indexed

    with st.form("rag_form"):
        prompt = st.text_area("Ask a question about your documents", key="rag_input_area", height=120)
        top_k_local = st.number_input("Context chunks to retrieve (top_k)", min_value=1, max_value=20, value=TOP_K)
        submitted = st.form_submit_button("Get answer")
    if submitted:
        if not prompt or not prompt.strip():
            st.warning("Provide a question.")
        else:
            with st.spinner("Searching..."):
                ans = answer_question_with_rag_safe(prompt, docs, top_k=top_k_local)
                st.markdown("**Answer:**")
                # prefer markdown so newlines render cleanly
                try:
                    st.markdown(ans)
                except Exception:
                    st.write(ans)
                # show simple context snippets used (best-effort)
                with st.expander("Context snippets used (first matches)"):
                    snippets_shown = 0
                    for up in docs:
                        chunks = (up.get("chunks") or [])[:5]
                        if not chunks:
                            continue
                        st.markdown(f"**{up.get('filename')}**")
                        for c in chunks[:3]:
                            st.write(c[:800] + ("..." if len(c) > 800 else ""))
                            snippets_shown += 1
                            if snippets_shown >= 6:
                                break
                        if snippets_shown >= 6:
                            break


def render_quizzes_tab():
    st.header("Generate & Take MCQs")
    uploads = [u for u in st.session_state.uploads if u.get("processed")]
    if not uploads:
        st.info("Process an upload first on the Upload tab.")
        return

    # select upload
    opts = {u["filename"]: idx for idx, u in enumerate(uploads)}
    sel = st.selectbox("Select document", list(opts.keys()), key="quiz_doc_select")
    up = uploads[opts[sel]]

    # generation options
    gen_col1, gen_col2, gen_col3 = st.columns([2, 2, 1])
    qcount = gen_col1.slider("Number of MCQs to generate", 1, 20, 5, key="gen_qcount")
    temp_opt = gen_col2.slider("LLM temperature (lower = more deterministic)", 0.0, 1.0, 0.0, 0.1, key="gen_temp")
    shuffle_before = gen_col3.checkbox("Shuffle questions", value=True, key="gen_shuffle")

    # store latest generated mcqs in session so user can interact without re-generating
    if "latest_generated_mcqs" not in st.session_state or st.session_state.get("latest_generated_for") != up.get("filename"):
        st.session_state["latest_generated_mcqs"] = []
        st.session_state["latest_generated_for"] = None

    if st.button("Generate MCQs", key=f"gen_mcq_{_sanitize_key(up['filename'])}"):
        with st.spinner("Generating MCQs from document..."):
            try:
                mcqs = generate_mcq_set_from_text(up.get("full_text", ""), qcount=qcount)
            except Exception as e:
                logger.exception("MCQ generation call failed: %s", e)
                mcqs = []
            if not mcqs:
                st.warning("No MCQs returned from the model.")
                st.session_state["latest_generated_mcqs"] = []
                st.session_state["latest_generated_for"] = None
            else:
                # normalize structure and guard missing fields
                normalized = []
                for obj in mcqs:
                    try:
                        qtext = str(obj.get("question") or "").strip()
                        options = list(obj.get("options") or [])
                        # ensure options are strings
                        options = [str(x) for x in options]
                        if not options:
                            # fallback: try to parse 'answers' or 'choices'
                            continue
                        answer_index = int(obj.get("answer_index", 0)) if obj.get("answer_index") is not None else 0
                        answer_index = max(0, min(len(options) - 1, answer_index))
                        normalized.append({"question": qtext, "options": options, "answer_index": answer_index})
                    except Exception:
                        continue
                if not normalized:
                    st.warning("MCQs returned were malformed and could not be used.")
                    st.session_state["latest_generated_mcqs"] = []
                    st.session_state["latest_generated_for"] = None
                else:
                    if shuffle_before:
                        import random
                        random.shuffle(normalized)
                    st.session_state["latest_generated_mcqs"] = normalized
                    st.session_state["latest_generated_for"] = up.get("filename")
                    st.success(f"Generated {len(normalized)} MCQs (stored in session).")

    # show generated MCQs (interactive)
    mcqs = st.session_state.get("latest_generated_mcqs", [])
    if mcqs:
        st.markdown("### Preview / Take generated MCQs")
        # create a unique namespace for radio keys so they survive reruns
        answers = {}
        for qi, q in enumerate(mcqs):
            qkey = f"gen_mcq_{_sanitize_key(up['filename'])}_{qi}"
            st.markdown(f"**Q{qi+1}. {q.get('question','(no text)')}**")
            opts = q.get("options") or []
            if not opts:
                st.write("_No options for this question_")
                continue
            # if the saved answer_index might point to an option that gets shuffled later, we only use it for scoring
            answers[qi] = st.radio(f"Select answer (Q{qi+1})", options=opts, key=qkey)
            st.markdown("---")

        if st.button("Submit generated answers", key=f"submit_generated_{_sanitize_key(up['filename'])}"):
            score = 0
            details = []
            for qi, q in enumerate(mcqs):
                opts = q.get("options") or []
                correct_idx = int(q.get("answer_index", 0)) if opts else 0
                correct_opt = opts[correct_idx] if (opts and 0 <= correct_idx < len(opts)) else (opts[0] if opts else None)
                picked = st.session_state.get(f"gen_mcq_{_sanitize_key(up['filename'])}_{qi}")
                ok = (picked == correct_opt)
                if ok:
                    score += 1
                details.append((qi + 1, ok, correct_opt, picked))
            st.success(f"You scored {score} / {len(mcqs)}")
            for d in details:
                qno, ok, correct_opt, picked = d
                if ok:
                    st.write(f"Q{qno}: ✅ Correct")
                else:
                    st.write(f"Q{qno}: ❌ Wrong — correct: **{correct_opt}**, you chose: {picked}")

        # save generated MCQs to DB (grouped by upload id)
        if st.button("Save generated MCQs to DB", key=f"save_generated_{_sanitize_key(up['filename'])}"):
            try:
                conn = get_db_connection()
                now = int(time.time())
                saved = 0
                with conn:
                    for obj in mcqs:
                        qtext = obj.get("question", "")
                        opts = obj.get("options", [])
                        ans = int(obj.get("answer_index", 0)) if opts else 0
                        conn.execute(
                            "INSERT INTO quizzes (upload_id, question, options, correct_index, created_at) VALUES (?, ?, ?, ?, ?)",
                            (str(up.get("db_id") or up.get("filename")), qtext, json.dumps(opts), ans, now)
                        )
                        saved += 1
                st.success(f"Saved {saved} MCQs to DB (upload_id={up.get('db_id') or up.get('filename')}).")
            except Exception:
                logger.exception("Save MCQs failed")
                st.warning("Could not save MCQs to DB. Check logs.")

    # allow taking saved MCQs from DB for this upload
    st.markdown("### Saved MCQs (from DB)")
    try:
        conn = get_db_connection()
        cur = conn.execute("SELECT id, question, options, correct_index, created_at FROM quizzes WHERE upload_id = ? ORDER BY created_at DESC", (str(up.get("db_id") or up.get("filename")),))
        saved_rows = cur.fetchall() or []
    except Exception:
        saved_rows = []
    if not saved_rows:
        st.info("No saved MCQs for this document (you can generate and save above).")
        return

    # show a compact list and option to take a subset
    take_count = st.number_input("How many recent saved MCQs to take", min_value=1, max_value=len(saved_rows), value=min(10, len(saved_rows)), step=1, key="take_saved_count")
    take_button = st.button("Take saved MCQs", key=f"take_saved_{_sanitize_key(up['filename'])}")
    if take_button:
        # prepare questions
        selected_rows = saved_rows[:int(take_count)]
        quiz_items = []
        for r in selected_rows:
            try:
                qid = r[0]
                qtext = r[1] or ""
                opts = json.loads(r[2]) if r[2] else []
                correct_index = int(r[3] or 0)
                # sanitize
                opts = [str(x) for x in opts] if opts else []
                if not opts:
                    continue
                quiz_items.append({"db_id": qid, "question": qtext, "options": opts, "answer_index": correct_index})
            except Exception:
                continue
        if not quiz_items:
            st.warning("Saved MCQs were malformed and cannot be taken.")
        else:
            # store in session under a unique key then rerun to render the interactive quiz
            st.session_state[f"taking_saved_mcqs_for_{_sanitize_key(up['filename'])}"] = quiz_items
            safe_rerun()

    # if session has quiz to take, render it
    session_key = f"taking_saved_mcqs_for_{_sanitize_key(up['filename'])}"
    if st.session_state.get(session_key):
        items = st.session_state.get(session_key)
        st.markdown(f"#### Taking {len(items)} saved MCQs from DB (upload: {up.get('filename')})")
        user_answers = {}
        for i, it in enumerate(items):
            st.markdown(f"**Q{i+1}. {it.get('question','(no text)')}**")
            user_answers[i] = st.radio(f"Choose (saved_{i})", options=it.get("options", []), key=f"saved_mcq_radio_{_sanitize_key(up['filename'])}_{i}")
            st.markdown("---")
        if st.button("Submit saved-quiz answers", key=f"submit_saved_{_sanitize_key(up['filename'])}"):
            score = 0
            details = []
            for i, it in enumerate(items):
                opts = it.get("options", [])
                correct_idx = int(it.get("answer_index", 0)) if opts else 0
                correct_opt = opts[correct_idx] if (opts and 0 <= correct_idx < len(opts)) else (opts[0] if opts else None)
                picked = st.session_state.get(f"saved_mcq_radio_{_sanitize_key(up['filename'])}_{i}")
                ok = (picked == correct_opt)
                if ok:
                    score += 1
                details.append((i + 1, ok, correct_opt, picked))
            st.success(f"You scored {score} / {len(items)}")
            for d in details:
                qno, ok, correct_opt, picked = d
                if ok:
                    st.write(f"Q{qno}: ✅ Correct")
                else:
                    st.write(f"Q{qno}: ❌ Wrong — correct: **{correct_opt}**, you chose: {picked}")
            # clear the session quiz after completion
            del st.session_state[session_key]

def render_flashcards_tab():
    st.header("Flashcards")
    uploads = st.session_state.uploads
    if not uploads:
        st.info("Upload files first.")
        return

    sel = st.selectbox("Choose document", [u["filename"] for u in uploads], key="fc_doc_select")
    up = next((u for u in uploads if u["filename"] == sel), None)
    if up is None:
        st.warning("Selected upload not found in session.")
        return

    # generation options
    gen_cols = st.columns([2, 2, 1])
    num_cards = gen_cols[0].number_input("Max flashcards to generate", min_value=5, max_value=200, value=20, step=5, key="fc_gen_count")
    per_slide = gen_cols[1].slider("Max per slide (best-effort)", 1, 5, 2, key="fc_per_slide")
    preview_only = gen_cols[2].checkbox("Preview only (don't save)", value=False, key="fc_preview_only")

    if st.button("Generate Flashcards", key=f"gen_fc_{_sanitize_key(up['filename'])}"):
        with st.spinner("Generating flashcards..."):
            cards = generate_flashcards_from_text(up.get("full_text", ""), n=int(num_cards))
            if not cards:
                st.warning("No flashcards returned.")
            else:
                # optionally save
                if preview_only:
                    st.info(f"Previewing {min(len(cards), 10)} flashcards (preview-only).")
                    for c in cards[:10]:
                        st.markdown(f"**Q:** {c.get('question')}")
                        st.caption(f"A: {c.get('answer')}")
                else:
                    saved = add_flashcards_to_db_safe(up, cards)
                    st.success(f"Saved {saved} flashcards to DB.")

    # management & practice
    manage_col1, manage_col2 = st.columns(2)
    if manage_col1.button("Load due cards for practice"):
        st.session_state.due_cards = get_due_flashcards_safe()
        st.session_state.current_card_idx = 0
        safe_rerun()

    if manage_col2.button("Preview 5 flashcards (from DB)"):
        preview = get_due_flashcards_safe(upload=up, limit=5)
        if preview:
            for c in preview:
                st.markdown(f"**Q:** {c.get('question')}")
                st.caption(f"A: {c.get('answer')}")
        else:
            st.info("No flashcards to preview for this upload.")

    # export
    if st.button("Export all flashcards for this upload (Anki TSV)"):
        uid = up.get("db_id") or up.get("filename")
        try:
            res = anki_export_tsv(uid, conn=get_db_connection())
        except Exception:
            res = anki_export_tsv(uid, conn=None)
        if res:
            fname, b = res
            st.download_button("Download Anki TSV", data=b, file_name=fname, mime="text/tab-separated-values")
        else:
            st.warning("No flashcards found for this upload to export.")

    # practice UI
    if st.session_state.get("due_cards"):
        render_flashcard_practice_ui()

def render_flashcard_practice_ui():
    cards = st.session_state.get("due_cards", [])
    if not cards:
        st.info("No due cards to practice now.")
        return

    idx = st.session_state.get("current_card_idx", 0)
    total = len(cards)
    if idx >= total:
        st.success("✨ You finished all due cards in this session.")
        st.session_state["due_cards"] = []
        st.session_state["current_card_idx"] = 0
        return

    card = cards[idx]
    st.markdown(f"##### Card {idx+1} of {total}")
    st.write(card.get("question"))
    # allow toggle answer without triggering rerun of main flow
    show_key = f"fc_show_answer_{card.get('id')}"
    if show_key not in st.session_state:
        st.session_state[show_key] = False
    if st.button("Show Answer", key=f"show_btn_{card.get('id')}"):
        st.session_state[show_key] = True
    if st.session_state.get(show_key):
        st.markdown("**Answer:**")
        st.write(card.get("answer"))

    # make review buttons compact and confirm action
    col_again, col_hard, col_good, col_easy = st.columns([1,1,1,1])
    if col_again.button("Again", key=f"again_{card.get('id')}"):
        update_flashcard_review_safe(card, 1)
        st.session_state["current_card_idx"] = idx + 1
        safe_rerun()
    if col_hard.button("Hard", key=f"hard_{card.get('id')}"):
        update_flashcard_review_safe(card, 3)
        st.session_state["current_card_idx"] = idx + 1
        safe_rerun()
    if col_good.button("Good", key=f"good_{card.get('id')}"):
        update_flashcard_review_safe(card, 4)
        st.session_state["current_card_idx"] = idx + 1
        safe_rerun()
    if col_easy.button("Easy", key=f"easy_{card.get('id')}"):
        update_flashcard_review_safe(card, 5)
        st.session_state["current_card_idx"] = idx + 1
        safe_rerun()

def render_settings_tab():
    st.header("Settings & Exports")
    key = st.text_input("OpenRouter API Key (session only)", value=st.session_state.get("OPENROUTER_API_KEY",""), type="password")
    if key != st.session_state.get("OPENROUTER_API_KEY"):
        st.session_state["OPENROUTER_API_KEY"] = key
        st.success("API key saved in session (not persisted).")

    st.markdown("#### Export flashcards (Anki TSV)")
    uploads = st.session_state.get("uploads", []) or []
    if uploads:
        # show user-friendly filenames via index selection (robust if filenames duplicate)
        idx = st.selectbox("Select upload", options=list(range(len(uploads))),
                           format_func=lambda i: uploads[i].get("filename", f"upload_{i}"),
                           key="export_select")
        if st.button("Export Anki TSV", key="export_anki_btn"):
            selected = uploads[int(idx)]
            uid = selected.get("db_id") or selected.get("filename")
            conn = None
            try:
                conn = get_db_connection()
            except Exception:
                conn = None
            res = anki_export_tsv(uid, conn)
            if not res:
                st.warning("No flashcards for selected upload.")
            else:
                fname, b = res
                st.download_button("Download Anki TSV", data=b, file_name=fname, mime="text/tab-separated-values")
    else:
        st.info("No uploads available for export.")

    st.divider()
    st.subheader("Diagnostics")

    statuses = [
        ("OpenRouter API Key", bool(st.session_state.get("OPENROUTER_API_KEY") or DEFAULT_OPENROUTER_API_KEY)),
        ("NumPy", _HAS_NUMPY),
        ("Requests", _HAS_REQUESTS),
        ("PyMuPDF (fitz)", _HAS_PYMUPDF),
        ("python-pptx", _HAS_PPTX),
        ("Sentence-Transformers", _HAS_SENTENCE_TRANSFORMERS),
        ("FAISS", _HAS_FAISS),
        ("EasyOCR", _HAS_EASYOCR),
        ("Pillow (PIL)", _HAS_PIL),
        ("gTTS", _HAS_GTTS),
    ]
    cols = st.columns(3)
    for i, (name, ok) in enumerate(statuses):
        with cols[i % 3]:
            st.markdown(f"{'✅' if ok else '⚠️'} **{name}**")

    with st.expander("Publish checklist"):
        st.markdown("- Add OPENROUTER_API_KEY to Streamlit secrets (or paste it above for session use).")
        st.markdown("- Optional: install extras you need (PyMuPDF for PDFs, python-pptx for PPTX, EasyOCR for image OCR, gTTS for audio).")
        st.markdown("- Run locally with: `streamlit run app.py` and verify Diagnostics are green.")
        st.markdown("- Deploy to Streamlit Community Cloud or your container platform; this app uses no top-level imports, easing cold starts.")


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
