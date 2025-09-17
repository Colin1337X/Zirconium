# server.py — username+password auth; providers (ollama/openrouter/chutes); profiles/threads;
# themes/branding; RAG; host-only admin; telemetry dashboard.  (NO refusal detection)

from flask import Flask, request, Response, send_from_directory, jsonify, stream_with_context, session
import os, io, json, time, uuid, pathlib, requests, re, math, sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ---------- Optional PDF parser ----------
try:
    import PyPDF2  # pip install PyPDF2 (optional)
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# ---------- Paths ----------
BASE = pathlib.Path(__file__).parent.resolve()
DATA = BASE / "data"; DATA.mkdir(exist_ok=True)
CONV_DIR = DATA / "conversations"; CONV_DIR.mkdir(exist_ok=True)
MEM_FILE = DATA / "memory.json"
LORE_DIR = DATA / "lore"; LORE_DIR.mkdir(exist_ok=True)
RAG_DIR = DATA / "rag"; RAG_DIR.mkdir(exist_ok=True)
PROVIDERS_FILE = DATA / "providers.json"
SERVER_STATE_FILE = DATA / "server_state.json"
DB_PATH = DATA / "app.db"

# ---------- Branding (Themes) ----------
BRAND_DIR = DATA / "branding"; BRAND_DIR.mkdir(exist_ok=True)
THEMES_DIR = BRAND_DIR / "themes"; THEMES_DIR.mkdir(exist_ok=True)
BRANDING_FILE = BRAND_DIR / "branding_themes.json"

DEFAULT_THEME = {
    "slug": "default",
    "app_name": "Local Terminal Chat",
    "colors": {
        "bg": "#0a0c0a", "fg": "#c8facc", "muted": "#8fb99b",
        "panel": "#0f110f", "border": "#1a1f1a", "accent": "#62ff80",
        "danger": "#ff6b6b", "link": "#7fe0ff"
    },
    "radius": 8,
    "font_stack": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
}
DEFAULT_BRANDING_STATE = {
    "active_theme": "default",
    "default_theme": "default",
    "themes": { "default": DEFAULT_THEME }
}

def theme_dir(slug: str) -> pathlib.Path:
    p = THEMES_DIR / slug
    p.mkdir(exist_ok=True, parents=True)
    return p

def load_branding_state():
    if not BRANDING_FILE.exists():
        BRANDING_FILE.write_text(json.dumps(DEFAULT_BRANDING_STATE, indent=2))
        theme_dir("default")
        return DEFAULT_BRANDING_STATE
    try:
        data = json.loads(BRANDING_FILE.read_text())
        if "themes" not in data or not isinstance(data["themes"], dict) or not data["themes"]:
            data = DEFAULT_BRANDING_STATE
        data.setdefault("active_theme","default")
        data.setdefault("default_theme","default")
        for slug in list(data["themes"].keys()):
            theme_dir(slug)
        return data
    except Exception:
        return DEFAULT_BRANDING_STATE

def save_branding_state(state):
    BRANDING_FILE.write_text(json.dumps(state, indent=2))

def get_theme(slug):
    st = load_branding_state()
    th = st["themes"].get(slug)
    if not th: return None
    td = theme_dir(slug)
    t = th.copy(); t["slug"] = slug
    t["logo_url"] = f"/branding/themes/{slug}/logo.png" if (td/"logo.png").exists() else None
    t["favicon_url"] = f"/branding/themes/{slug}/favicon.ico" if (td/"favicon.ico").exists() else None
    t["custom_css_url"] = f"/branding/themes/{slug}/custom.css" if (td/"custom.css").exists() else None
    return t

def get_active_theme():
    st = load_branding_state()
    slug = st.get("active_theme","default")
    return get_theme(slug) or get_theme("default")

# ---------- Flask ----------
app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=False)

AUTH_TOKEN = os.environ.get("CHAT_AUTH_TOKEN")  # optional LAN gate

# ---------- Host-only admin detection ----------
HOST_ADMIN_IPS = set(ip.strip() for ip in os.environ.get("HOST_ADMIN_IPS", "127.0.0.1,::1").split(",") if ip.strip())
def is_host_admin_request():
    remote = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    remote = remote.split(",")[0].strip()
    return remote in HOST_ADMIN_IPS

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not is_host_admin_request():
            return ("Forbidden", 403)
        admin_key = os.environ.get("HOST_ADMIN_KEY")
        if admin_key and request.headers.get("X-Admin-Key") != admin_key:
            return ("Forbidden", 403)
        return f(*args, **kwargs)
    return wrap

# ---------- DB ----------
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def db_init():
    con = db(); cur = con.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE,
      email TEXT,
      pass_hash TEXT NOT NULL,
      created_at INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS profiles(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      name TEXT NOT NULL,
      kind TEXT NOT NULL,
      settings_json TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS threads(
      id TEXT PRIMARY KEY,
      profile_id INTEGER NOT NULL,
      title TEXT,
      created_at INTEGER NOT NULL,
      FOREIGN KEY(profile_id) REFERENCES profiles(id)
    );
    CREATE TABLE IF NOT EXISTS clients(
      id TEXT PRIMARY KEY,
      user_id INTEGER NOT NULL,
      app_kind TEXT NOT NULL,
      user_agent TEXT,
      os TEXT,
      device TEXT,
      ip TEXT,
      first_seen INTEGER NOT NULL,
      last_seen INTEGER NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)
    try:
        cur.execute("UPDATE users SET username = email WHERE (username IS NULL OR username='') AND email IS NOT NULL")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    except Exception:
        pass
    con.commit(); con.close()
db_init()

# ---------- Server state ----------
def load_state():
    if not SERVER_STATE_FILE.exists():
        SERVER_STATE_FILE.write_text(json.dumps({"locked": False}, indent=2))
    try: return json.loads(SERVER_STATE_FILE.read_text())
    except Exception: return {"locked": False}

def save_state(st):
    SERVER_STATE_FILE.write_text(json.dumps(st, indent=2))

# ---------- Memory ----------
DEFAULT_MEMORY = {"stable_facts": [], "summary": ""}
if not MEM_FILE.exists():
    MEM_FILE.write_text(json.dumps(DEFAULT_MEMORY, ensure_ascii=False, indent=2), "utf-8")

def load_json(path, default):
    try: return json.loads(path.read_text("utf-8"))
    except Exception: return default
def save_json(path, obj): path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")
def load_memory(): return load_json(MEM_FILE, DEFAULT_MEMORY)
def save_memory(mem): save_json(MEM_FILE, mem)

# ---------- Providers ----------
PROVIDERS_DEFAULT = {
    "current": "ollama",
    "ollama": { "api_base": os.environ.get("OLLAMA","http://127.0.0.1:11434"), "api_key": None },
    "openrouter": { "api_base": "https://openrouter.ai/api/v1", "api_key": os.environ.get("OPENROUTER_API_KEY") },
    "chutes": { "api_base": os.environ.get("CHUTES_API_BASE", "https://api.chutes.ai/v1"), "api_key": os.environ.get("CHUTES_API_KEY") }
}
EMBED_MODEL = os.environ.get("EMBED_MODEL","nomic-embed-text")
PROVIDERS_FILE = DATA / "providers.json"

def load_providers():
    if not PROVIDERS_FILE.exists():
        save_json(PROVIDERS_FILE, PROVIDERS_DEFAULT); return PROVIDERS_DEFAULT
    cfg = load_json(PROVIDERS_FILE, PROVIDERS_DEFAULT)
    allowed = {"current","ollama","openrouter","chutes"}
    cfg = {k:v for k,v in cfg.items() if k in allowed or k=="current"}
    if cfg.get("current") not in ("ollama","openrouter","chutes"): cfg["current"] = "ollama"
    for k in ("ollama","openrouter","chutes"):
        cfg.setdefault(k, PROVIDERS_DEFAULT[k])
    save_json(PROVIDERS_FILE, cfg)
    return cfg

def save_providers(cfg):
    out = {
        "current": cfg.get("current","ollama"),
        "ollama": cfg.get("ollama", PROVIDERS_DEFAULT["ollama"]),
        "openrouter": cfg.get("openrouter", PROVIDERS_DEFAULT["openrouter"]),
        "chutes": cfg.get("chutes", PROVIDERS_DEFAULT["chutes"]),
    }
    save_json(PROVIDERS_FILE, out)

def headers_for(provider, cfg):
    h = {"Content-Type":"application/json"}
    if provider in ("openrouter","chutes") and cfg.get("api_key"):
        h["Authorization"] = f"Bearer {cfg['api_key']}"
    return h

def chat_stream_ollama(model, messages):
    base = load_providers()["ollama"]["api_base"].rstrip("/")
    body = {"model": model, "messages": messages, "stream": True}
    return requests.post(f"{base}/api/chat", json=body, stream=True, timeout=None)

def chat_stream_openai_like(provider_name, model, messages, extra=None):
    cfg = load_providers()[provider_name]
    base = (cfg.get("api_base") or "").rstrip("/")
    url = f"{base}/chat/completions"
    payload = {"model": model, "messages": messages, "stream": True}
    if extra: payload.update(extra)
    return requests.post(url, json=payload, headers=headers_for(provider_name, cfg), stream=True, timeout=None)

# ---------- Conversations ----------
CONV_DIR.mkdir(exist_ok=True)
def conv_path(tid): return CONV_DIR / f"{tid}.json"
def load_conv(tid): return load_json(conv_path(tid), {"messages": []})
def save_conv(tid, data): save_json(conv_path(tid), data)

# ---------- Optional LAN gate ----------
@app.before_request
def optional_lan_gate():
    if request.path.startswith("/api/") and AUTH_TOKEN:
        if request.headers.get("Authorization","") != f"Bearer {AUTH_TOKEN}":
            return ("Unauthorized", 401)

# ---------- Auth helpers ----------
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not session.get("uid"):
            return ("Unauthorized", 401)
        return f(*args, **kwargs)
    return wrap
def current_user_id(): return session.get("uid")

# ---------- System prompts ----------
RPG_GUIDE = """
You are the game engine + world narrator. User messages may include an "actions" array (structured events).
ALWAYS respond in two channels:
1) NARRATION: brief, sensory-rich description of what happens next.
2) STATE: minimal JSON that summarizes outcomes for the UI.
STATE={"turn_summary":"...", "events":[{"type":"world|npc|system|damage|status|loot","text":"...","value":number?,"target":null|"name"}]}
Keep STATE concise; don't duplicate the full narration. Respect whisper/shout. Sequence queued actions crisply.
"""

def inject_memory(base_messages):
    mem = load_memory()
    facts = "\n".join(f"- {x}" for x in mem.get("stable_facts", [])[:24])
    summary = mem.get("summary", "")
    sys = {"role":"system","content":
        "You are a local assistant. Persist helpful context between chats.\n"
        "Stable facts:\n" + (facts or "(none)") + "\n\n" +
        "Rolling summary:\n" + (summary or "(empty)") + "\n\n" +
        "Use these sparingly; verify before assuming."
    }
    sys_rpg = {"role":"system","content": RPG_GUIDE}
    return [sys, sys_rpg] + base_messages

# ---------- Chunking / embeddings ----------
def chunk_text(txt, max_chars=1200):
    paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf)+len(p)+1 <= max_chars:
            buf = (buf+"\n\n"+p).strip()
        else:
            if buf: chunks.append(buf)
            buf = p
    if buf: chunks.append(buf)
    return chunks

def embed_many(base_url, chunks):
    try:
        r = requests.post(f"{base_url.rstrip('/')}/api/embeddings",
                          json={"model": EMBED_MODEL, "input": chunks}, timeout=180)
        r.raise_for_status(); data = r.json()
        return (data.get("embeddings") or data.get("data")) or []
    except Exception:
        return []

def cosim(a,b):
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(x*x for x in b)) or 1.0
    return sum(x*y for x,y in zip(a,b))/(na*nb)

def keyword_score(q,t):
    qs = set(re.findall(r"[a-zA-Z']{3,}", q.lower()))
    ts = set(re.findall(r"[a-zA-Z']{3,}", t.lower()))
    return len(qs & ts) / (1 + len(qs))

# ---------- Lore retrieval ----------
def load_lore(name):
    p = LORE_DIR / f"{name}.json"
    if not p.exists(): return None
    return load_json(p, None)

def retrieve_lore(names, query_text, k=6):
    prov = load_providers()
    ollama_base = prov["ollama"]["api_base"]
    results = []
    for nm in names or []:
        db = load_lore(nm)
        if not db: continue
        chunks = db.get("chunks", [])
        any_emb = any("emb" in c for c in chunks)
        if any_emb:
            qv = embed_many(ollama_base, [query_text])
            if qv:
                qv = qv[0]
                for c in chunks:
                    results.append((cosim(qv, c.get("emb") or []), c["text"], nm))
            else:
                for c in chunks:
                    results.append((keyword_score(query_text, c["text"]), c["text"], nm))
        else:
            for c in chunks:
                results.append((keyword_score(query_text, c["text"]), c["text"], nm))
    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:k]
    if not top: return None
    return "\n\n".join([f"[{nm}] {txt}" for _, txt, nm in top])

# ---------- RAG storage & retrieval ----------
def rag_db_path(profile_id, thread_id):
    p = RAG_DIR / f"{profile_id}_{thread_id}.json"
    if not p.exists():
        save_json(p, {"docs": []})
    return p

def load_rag(profile_id, thread_id):
    return load_json(rag_db_path(profile_id, thread_id), {"docs":[]})

def save_rag(profile_id, thread_id, data):
    save_json(rag_db_path(profile_id, thread_id), data)

def extract_text_from_upload(storage):
    name = storage.filename or f"file_{int(time.time())}"
    raw = storage.read()
    lower = name.lower()
    try:
        if lower.endswith((".txt", ".md", ".csv", ".json", ".log", ".html", ".css", ".js")):
            return name, raw.decode("utf-8", errors="ignore")
        if lower.endswith(".pdf") and HAS_PDF:
            text = []
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            for pg in reader.pages:
                try: text.append(pg.extract_text() or "")
                except Exception: pass
            return name, "\n\n".join(t for t in text if t)
        return name, raw.decode("utf-8", errors="ignore")
    except Exception:
        return name, ""

def retrieve_rag(profile_id, thread_id, query_text, k=8):
    db = load_rag(profile_id, thread_id)
    if not db.get("docs"): return None
    prov = load_providers()
    ollama_base = prov["ollama"]["api_base"]
    results=[]
    any_emb = any(any("emb" in c for c in d.get("chunks",[])) for d in db["docs"])
    qv = None
    if any_emb:
        qvecs = embed_many(ollama_base, [query_text])
        if qvecs: qv = qvecs[0]
    for d in db["docs"]:
        for c in d.get("chunks", []):
            score = cosim(qv, c["emb"]) if (qv is not None and "emb" in c) else keyword_score(query_text, c["text"])
            results.append((score, d["name"], c["text"]))
    results.sort(key=lambda x:x[0], reverse=True)
    top = results[:k]
    if not top: return None
    return "\n\n".join([f"[DOC:{nm}] {txt}" for _, nm, txt in top])

# ---------- Static & Branding ----------
@app.route("/")
def index(): return send_from_directory("static","index.html")

@app.route("/<path:path>")
def static_files(path): return send_from_directory("static", path)

@app.route("/branding/<path:filename>")
def branding_files(filename): return send_from_directory(BRAND_DIR, filename)

@app.route("/branding/themes/<slug>/<path:filename>")
def branding_theme_files(slug, filename): return send_from_directory(theme_dir(slug), filename)

@app.route("/api/branding", methods=["GET"])
def api_branding_active(): return jsonify(get_active_theme())

# ---------- Admin page ----------
@app.route("/admin")
@admin_required
def admin_page(): return send_from_directory("static", "admin.html")

# ---------- Public auth (USERNAME + PASSWORD) ----------
@app.route("/api/signup", methods=["POST"])
def signup():
    if load_state().get("locked"): return ("Logins disabled by admin", 403)
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    pw = data.get("password") or ""
    if not username or not pw: return ("Bad Request", 400)
    con = db(); cur = con.cursor()
    try:
        cur.execute("INSERT INTO users(username, pass_hash, created_at) VALUES(?,?,?)",
                    (username, generate_password_hash(pw), int(time.time())))
        uid = cur.lastrowid
        now = int(time.time())
        for name, kind in [("Assistant","assistant"), ("Brainstorm","brainstorm"), ("RPGs","rpg")]:
            cur.execute("INSERT INTO profiles(user_id,name,kind,settings_json,created_at) VALUES(?,?,?,?,?)",
                        (uid, name, kind, json.dumps({"default_model":"llama3.1:8b"}), now))
        con.commit()
        session["uid"] = uid
        return jsonify({"ok": True})
    except sqlite3.IntegrityError:
        return jsonify({"error":"Username already exists"}), 409
    finally:
        con.close()

@app.route("/api/login", methods=["POST"])
def login():
    if load_state().get("locked"): return ("Logins disabled by admin", 403)
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    pw = data.get("password") or ""
    con = db(); cur = con.cursor()
    cur.execute("SELECT id, pass_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone(); con.close()
    if not row or not check_password_hash(row["pass_hash"], pw):
        return ("Unauthorized", 401)
    session["uid"] = row["id"]
    return jsonify({"ok": True})

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear(); return jsonify({"ok": True})

# ---------- Providers & models ----------
@app.route("/api/providers", methods=["GET"])
def get_providers():
    cfg = load_providers()
    out = {"current": cfg.get("current","ollama")}
    for name in ("ollama","openrouter","chutes"):
        out[name] = {k:v for k,v in cfg.get(name,{}).items() if k != "api_key"}
    return jsonify(out)

@app.route("/api/provider_config", methods=["POST"])
@login_required
def set_provider_config_user():
    incoming = request.get_json(force=True)
    cfg = load_providers()
    cur = incoming.get("current")
    if cur in ("ollama","openrouter","chutes"): cfg["current"] = cur
    for name in ("ollama","openrouter","chutes"):
        if name in incoming: cfg[name].update({k:v for k,v in incoming[name].items()})
    save_providers(cfg); return jsonify({"ok": True})

@app.route("/api/admin/provider_config", methods=["POST"])
@admin_required
def set_provider_config_admin():
    incoming = request.get_json(force=True)
    cfg = load_providers()
    cur = incoming.get("current")
    if cur in ("ollama","openrouter","chutes"): cfg["current"] = cur
    for name in ("ollama","openrouter","chutes"):
        if name in incoming: cfg[name].update({k:v for k,v in incoming[name].items()})
    save_providers(cfg); return jsonify({"ok": True})

@app.route("/api/models", methods=["GET"])
def list_models():
    cfg = load_providers()
    try:
        r = requests.get(f"{cfg['ollama']['api_base'].rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status(); data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        return jsonify({"models": models})
    except Exception:
        return jsonify({"models": []})

# ---------- Admin: lock/unlock & themes ----------
@app.route("/api/admin/status", methods=["GET"])
@admin_required
def admin_status():
    return jsonify({"locked": bool(load_state().get("locked"))})

@app.route("/api/admin/lock", methods=["POST"])
@admin_required
def admin_lock():
    st = load_state(); st["locked"] = True; save_state(st); return jsonify({"ok": True,"locked": True})

@app.route("/api/admin/unlock", methods=["POST"])
@admin_required
def admin_unlock():
    st = load_state(); st["locked"] = False; save_state(st); return jsonify({"ok": True,"locked": False})

@app.route("/api/admin/branding_themes", methods=["GET"])
@admin_required
def admin_branding_themes():
    st = load_branding_state()
    out = [get_theme(slug) for slug in st["themes"].keys()]
    return jsonify({"active_theme": st.get("active_theme"), "default_theme": st.get("default_theme"), "themes": out})

@app.route("/api/admin/branding_select", methods=["POST"])
@admin_required
def admin_branding_select():
    js = request.get_json(force=True)
    which = js.get("which"); slug = (js.get("slug") or "").strip().lower()
    st = load_branding_state()
    if slug not in st["themes"]: return jsonify({"error":"unknown theme"}), 404
    if which == "active": st["active_theme"] = slug
    elif which == "default": st["default_theme"] = slug
    else: return jsonify({"error":"which must be 'active' or 'default'"}), 400
    save_branding_state(st); return jsonify({"ok": True})

@app.route("/api/admin/branding_theme", methods=["POST"])
@admin_required
def admin_branding_theme_upsert():
    js = request.get_json(force=True)
    slug = (js.get("slug") or "").strip().lower()
    if not re.match(r"^[a-z0-9\-]{1,32}$", slug): return jsonify({"error":"slug invalid"}), 400
    st = load_branding_state()
    base = st["themes"].get(slug, {"slug": slug})
    for k in ("app_name","radius","font_stack","colors"):
        if k in js: base[k] = js[k]
    st["themes"][slug] = base; save_branding_state(st); theme_dir(slug)
    return jsonify({"ok": True, "theme": get_theme(slug)})

@app.route("/api/admin/branding_theme_rename", methods=["POST"])
@admin_required
def admin_branding_theme_rename():
    js = request.get_json(force=True)
    old = (js.get("old_slug") or "").strip().lower()
    new = (js.get("new_slug") or "").strip().lower()
    if not re.match(r"^[a-z0-9\-]{1,32}$", new): return jsonify({"error":"new slug invalid"}), 400
    st = load_branding_state()
    if old not in st["themes"]: return jsonify({"error":"unknown theme"}), 404
    if new in st["themes"]: return jsonify({"error":"target exists"}), 409
    st["themes"][new] = st["themes"].pop(old); st["themes"][new]["slug"] = new
    (THEMES_DIR/old).rename(THEMES_DIR/new)
    if st["active_theme"] == old: st["active_theme"] = new
    if st["default_theme"] == old: st["default_theme"] = new
    save_branding_state(st)
    return jsonify({"ok": True, "theme": get_theme(new)})

@app.route("/api/admin/branding_theme", methods=["DELETE"])
@admin_required
def admin_branding_theme_delete():
    slug = (request.args.get("slug") or "").strip().lower()
    st = load_branding_state()
    if slug not in st["themes"]: return jsonify({"error":"unknown theme"}), 404
    if slug in ("default",) or slug in (st["active_theme"], st["default_theme"]):
        return jsonify({"error":"cannot delete default/active theme"}), 400
    st["themes"].pop(slug, None)
    try:
        d = THEMES_DIR/slug
        for p in d.glob("*"):
            try: p.unlink()
            except: pass
        d.rmdir()
    except: pass
    save_branding_state(st)
    return jsonify({"ok": True})

@app.route("/api/admin/branding_theme_duplicate", methods=["POST"])
@admin_required
def admin_branding_theme_duplicate():
    js = request.get_json(force=True)
    src = (js.get("src_slug") or "").strip().lower()
    dst = (js.get("dst_slug") or "").strip().lower()
    if not re.match(r"^[a-z0-9\-]{1,32}$", dst): return jsonify({"error":"dst slug invalid"}), 400
    st = load_branding_state()
    if src not in st["themes"]: return jsonify({"error":"unknown src"}), 404
    if dst in st["themes"]: return jsonify({"error":"dst exists"}), 409
    st["themes"][dst] = json.loads(json.dumps(st["themes"][src])); st["themes"][dst]["slug"] = dst
    save_branding_state(st)
    sdir, ddir = theme_dir(src), theme_dir(dst)
    for name in ("logo.png","favicon.ico","custom.css"):
        sp = sdir/name
        if sp.exists(): (ddir/name).write_bytes(sp.read_bytes())
    return jsonify({"ok": True, "theme": get_theme(dst)})

@app.route("/api/admin/branding_theme_assets", methods=["POST"])
@admin_required
def admin_branding_theme_assets():
    slug = (request.args.get("slug") or "").strip().lower()
    st = load_branding_state()
    if slug not in st["themes"]: return jsonify({"error":"unknown theme"}), 404
    td = theme_dir(slug); changed=[]
    if "logo" in request.files:
        f = request.files["logo"]
        if f and f.filename.lower().endswith((".png",".webp",".jpg",".jpeg",".gif",".svg",".ico")):
            (td/"logo.png").write_bytes(f.read()); changed.append("logo")
    if "favicon" in request.files:
        f = request.files["favicon"]
        if f and f.filename.lower().endswith((".ico",".png")):
            (td/"favicon.ico").write_bytes(f.read()); changed.append("favicon")
    if "custom_css" in request.files:
        f = request.files["custom_css"]
        if f and f.filename.lower().endswith(".css"):
            (td/"custom.css").write_bytes(f.read()); changed.append("custom_css")
    return jsonify({"ok": True, "changed": changed})

# ---------- Profiles & threads ----------
@app.route("/api/me", methods=["GET"])
@login_required
def me():
    uid = current_user_id()
    con = db(); cur = con.cursor()
    cur.execute("SELECT id, username, created_at FROM users WHERE id=?", (uid,))
    u = cur.fetchone()
    cur.execute("SELECT id,name,kind,settings_json FROM profiles WHERE user_id=? ORDER BY id", (uid,))
    profs = [{"id":r["id"],"name":r["name"],"kind":r["kind"],"settings":json.loads(r["settings_json"])} for r in cur.fetchall()]
    con.close()
    return jsonify({"user":{"id":u["id"],"username":u["username"]}, "profiles":profs})

@app.route("/api/profiles", methods=["POST"])
@login_required
def create_profile():
    uid = current_user_id()
    data = request.get_json(force=True)
    name = (data.get("name") or "Profile").strip()
    kind = (data.get("kind") or "custom").strip()
    settings = data.get("settings") or {}
    con = db(); cur = con.cursor()
    cur.execute("INSERT INTO profiles(user_id,name,kind,settings_json,created_at) VALUES(?,?,?,?,?)",
                (uid, name, kind, json.dumps(settings), int(time.time())))
    pid = cur.lastrowid; con.commit(); con.close()
    return jsonify({"id": pid, "name": name, "kind": kind})

@app.route("/api/threads", methods=["GET"])
@login_required
def list_threads():
    uid = current_user_id()
    profile_id = int(request.args.get("profile_id"))
    con = db(); cur = con.cursor()
    cur.execute("""SELECT t.id,t.title,t.created_at 
                   FROM threads t JOIN profiles p ON p.id=t.profile_id 
                   WHERE p.user_id=? AND p.id=? ORDER BY t.created_at DESC""",
                (uid, profile_id))
    rows = [dict(r) for r in cur.fetchall()]
    con.close(); return jsonify({"threads": rows})

@app.route("/api/new_thread", methods=["POST"])
@login_required
def new_thread():
    uid = current_user_id()
    data = request.get_json(force=True)
    profile_id = int(data.get("profile_id"))
    title = (data.get("title") or "").strip() or "Untitled"
    con = db(); cur = con.cursor()
    cur.execute("SELECT id FROM profiles WHERE id=? AND user_id=?", (profile_id, uid))
    if not cur.fetchone():
        con.close(); return ("Forbidden", 403)
    tid = uuid.uuid4().hex[:12]
    cur.execute("INSERT INTO threads(id,profile_id,title,created_at) VALUES(?,?,?,?)",
                (tid, profile_id, title, int(time.time())))
    con.commit(); con.close()
    save_conv(tid, {"messages": []})
    return jsonify({"thread_id": tid, "title": title})

# ---------- Telemetry (clients heartbeat) ----------
def _guess_os(ua: str) -> str:
    u = (ua or "").lower()
    if "windows" in u: return "Windows"
    if "mac os" in u or "macintosh" in u: return "macOS"
    if "iphone" in u or "ios" in u: return "iOS"
    if "android" in u: return "Android"
    if "linux" in u: return "Linux"
    return "Unknown"
def _guess_device(ua: str) -> str:
    u = (ua or "").lower()
    if "ipad" in u or "tablet" in u: return "Tablet"
    if "mobile" in u or "iphone" in u or "android" in u: return "Mobile"
    return "Desktop"

@app.route("/api/telemetry/heartbeat", methods=["POST"])
@login_required
def telemetry_heartbeat():
    js = request.get_json(force=True)
    app_kind = (js.get("app_kind") or "web").strip().lower()
    if app_kind not in ("web","desktop"): app_kind = "web"
    client_id = js.get("client_id") or uuid.uuid4().hex[:16]
    ua = request.headers.get("User-Agent","")
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    os_name = (js.get("os") or _guess_os(ua))
    device = (js.get("device") or _guess_device(ua))
    now = int(time.time()); uid = current_user_id()
    con = db(); cur = con.cursor()
    cur.execute("SELECT id FROM clients WHERE id=? AND user_id=?", (client_id, uid))
    row = cur.fetchone()
    if row:
        cur.execute("""UPDATE clients SET app_kind=?, user_agent=?, os=?, device=?, ip=?, last_seen=?
                       WHERE id=? AND user_id=?""",
                    (app_kind, ua, os_name, device, ip, now, client_id, uid))
    else:
        cur.execute("""INSERT INTO clients(id,user_id,app_kind,user_agent,os,device,ip,first_seen,last_seen)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (client_id, uid, app_kind, ua, os_name, device, ip, now, now))
    con.commit(); con.close()
    return jsonify({"ok": True, "client_id": client_id})

@app.route("/api/admin/clients", methods=["GET"])
@admin_required
def admin_list_clients():
    now = int(time.time())
    con = db(); cur = con.cursor()
    cur.execute("""SELECT c.id, c.user_id, u.username, c.app_kind, c.os, c.device, c.ip,
                          c.first_seen, c.last_seen, c.user_agent
                   FROM clients c JOIN users u ON u.id=c.user_id
                   ORDER BY c.last_seen DESC LIMIT 500""")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    out = []
    for r in rows:
        uptime = max(0, r["last_seen"] - r["first_seen"])
        online = (now - r["last_seen"]) <= 45
        r["uptime_sec"] = uptime
        r["online"] = online
        out.append(r)
    return jsonify({"clients": out, "now": now})

# ---------- Chat ----------
def call_local_summarize(text, chat_model="llama3.1:8b"):
    base = load_providers()["ollama"]["api_base"]
    prompt = ("Summarize the following conversation into <= 120 words, focusing on user goals, preferences, ongoing tasks, "
              "and facts likely to remain true next time. Neutral tone; avoid ephemeral details.\n\n"+text)
    r = requests.post(f"{base.rstrip('/')}/api/generate",
                      json={"model": chat_model, "prompt": prompt, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json().get("response","")

def extract_facts_heuristic(text):
    facts, seen = [], set()
    for line in text.splitlines():
        s = line.strip().rstrip(".")
        if not s: continue
        l = s.lower()
        if l.startswith(("i ","i'm","i am","my ","we ","we're","we are")) and 6 <= len(s) <= 140:
            k = s.lower()
            if k not in seen:
                seen.add(k); facts.append(s)
        if len(facts) >= 8: break
    return facts

@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    payload = request.get_json(force=True)
    model = payload.get("model") or "llama3.1:8b"
    provider = payload.get("provider") or load_providers().get("current","ollama")
    profile_id = int(payload.get("profile_id"))
    thread_id = payload.get("thread_id")
    lore_names = payload.get("lore_names") or []
    user_msgs = payload.get("messages", [])
    options = payload.get("options")

    con = db(); cur = con.cursor()
    cur.execute("""SELECT t.id FROM threads t 
                   JOIN profiles p ON p.id=t.profile_id 
                   WHERE t.id=? AND p.user_id=? AND p.id=?""",
                (thread_id, current_user_id(), profile_id))
    ok = cur.fetchone(); con.close()
    if not ok: return ("Forbidden", 403)

    conv = load_conv(thread_id)
    conv["messages"].extend(user_msgs)
    recent_msgs = conv["messages"][-18:]

    # Parse action JSON
    try:
        last = user_msgs[-1]["content"]
        if last.lstrip().startswith("{"):
            obj = json.loads(last)
            if isinstance(obj, dict) and "actions" in obj:
                user_msgs[-1]["content"] = "USER_ACTIONS:\n" + json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    user_turn_text = recent_msgs[-1]["content"] if recent_msgs else ""
    rag_ctx = retrieve_rag(profile_id, thread_id, user_turn_text, k=8)
    lore_ctx = retrieve_lore(lore_names, user_turn_text, k=6)

    base_messages = inject_memory(recent_msgs)
    if rag_ctx:
        base_messages = [{"role":"system", "content":"Attached Documents (high priority factual context):\n"+rag_ctx}] + base_messages
    if lore_ctx:
        base_messages = [{"role":"system", "content":"Lore Context (authoritative canon; prefer over guesses):\n"+lore_ctx}] + base_messages

    # Upstream stream
    try:
        if provider=="ollama":
            upstream = chat_stream_ollama(model, base_messages); mode="ollama"
        elif provider in ("openrouter","chutes"):
            upstream = chat_stream_openai_like(provider, model, base_messages, extra=options); mode="openai"
        else:
            return jsonify({"error": f"Unknown provider '{provider}'"}), 400
    except Exception as e:
        return jsonify({"error": f"Upstream init failed: {e}"}), 500

    def stream_back():
        acc=[]
        if mode=="ollama":
            for line in upstream.iter_lines(decode_unicode=True):
                if not line: continue
                try:
                    obj = json.loads(line)
                    txt = (obj.get("message") or {}).get("content") or ""
                    if txt: acc.append(txt); yield f"data: {txt}\n\n"
                    if obj.get("done"): yield "event: done\ndata: {}\n\n"; break
                except Exception:
                    yield f"data: {line}\n\n"
        else:
            for raw in upstream.iter_lines(decode_unicode=True):
                if not raw: continue
                if raw.startswith("data: "):
                    data = raw[6:]
                    if data.strip() == "[DONE]": yield "event: done\ndata: {}\n\n"; break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices",[{}])[0].get("delta",{}).get("content")
                        if delta: acc.append(delta); yield f"data: {delta}\n\n"
                    except Exception: pass

        # persist assistant turn
        full = "".join(acc)
        conv["messages"].append({"role":"assistant","content": full})
        save_conv(thread_id, conv)

        # update memory
        try:
            transcript = "\n".join(m["role"] + ": " + m["content"] for m in conv["messages"][-40:])
            mem = load_memory()
            mem["summary"] = call_local_summarize(transcript)[:1000]
            newfacts = extract_facts_heuristic(transcript)
            merged = newfacts + [f for f in mem.get("stable_facts", []) if f not in newfacts]
            mem["stable_facts"] = merged[:40]
            save_memory(mem)
        except Exception:
            pass

    headers = {"Content-Type":"text/event-stream; charset=utf-8",
               "Cache-Control":"no-cache","Connection":"keep-alive"}
    return Response(stream_with_context(stream_back()), headers=headers)

# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)