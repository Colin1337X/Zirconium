# server.py — username+password auth; providers (ollama/openrouter/chutes); branding/themes; RAG;
# host-only admin; telemetry dashboard; customizable MOTD; RPG status tracker (RimWorld-ish).

from flask import Flask, request, Response, send_from_directory, jsonify, stream_with_context, session
import os, io, json, time, uuid, pathlib, requests, re, math, sqlite3, platform, shutil, socket, random

from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ---------- Optional deps ----------
try:
    import PyPDF2
    HAS_PDF = True
except Exception:
    HAS_PDF = False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

# ---------- Paths ----------
BASE = pathlib.Path(__file__).parent.resolve()
DATA = BASE / "data"; DATA.mkdir(exist_ok=True)
CONV_DIR = DATA / "conversations"; CONV_DIR.mkdir(exist_ok=True)
MEM_FILE = DATA / "memory.json"
LORE_DIR = DATA / "lore"; LORE_DIR.mkdir(exist_ok=True)
RAG_DIR = DATA / "rag"; RAG_DIR.mkdir(exist_ok=True)
BRAND_DIR = DATA / "branding"; BRAND_DIR.mkdir(exist_ok=True)
THEMES_DIR = BRAND_DIR / "themes"; THEMES_DIR.mkdir(exist_ok=True)
BRANDING_FILE = BRAND_DIR / "branding_themes.json"
PROVIDERS_FILE = DATA / "providers.json"
SERVER_STATE_FILE = DATA / "server_state.json"
DB_PATH = DATA / "app.db"

# New config files
MOTD_FILE = DATA / "motd.json"  # ascii + list items
RPG_DIR = DATA / "rpg"; RPG_DIR.mkdir(exist_ok=True)

# ---------- Branding (Themes) ----------
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

# ---------- Utilities ----------
def load_json(path, default):
    try: return json.loads(path.read_text("utf-8"))
    except Exception: return default
def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")

# ---------- Server state & memory ----------
def load_state():
    if not SERVER_STATE_FILE.exists():
        SERVER_STATE_FILE.write_text(json.dumps({"locked": False}, indent=2))
    try: return json.loads(SERVER_STATE_FILE.read_text())
    except Exception: return {"locked": False}
def save_state(st): SERVER_STATE_FILE.write_text(json.dumps(st, indent=2))

DEFAULT_MEMORY = {"stable_facts": [], "summary": ""}
if not MEM_FILE.exists():
    MEM_FILE.write_text(json.dumps(DEFAULT_MEMORY, ensure_ascii=False, indent=2), "utf-8")
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

# ---------- Lore & RAG ----------
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

# ---------- Auth ----------
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

# ---------- MOTD Config ----------
DEFAULT_MOTD = {
    "ascii": r"""
   _           _        _        _           _ _       
  | |         | |      | |      | |         | | |      
  | |     __ _| | _____| |_ __ _| |__   __ _| | | ___  
  | |    / _` | |/ / _ \ __/ _` | '_ \ / _` | | |/ _ \ 
  | |___| (_| |   <  __/ || (_| | | | | (_| | | | (_) |
  |______\__,_|_|\_\___|\__\__,_|_| |_|\__,_|_|_|\___/ 
""".strip("\n"),
    "list": [
        "Welcome to your local AI terminal.",
        "Tip: Drag & drop files into the chat for on-the-fly RAG.",
        "Admin: /admin (host-only)."
    ]
}
if not MOTD_FILE.exists():
    save_json(MOTD_FILE, DEFAULT_MOTD)

@app.route("/api/admin/motd", methods=["GET"])
@admin_required
def admin_motd_get():
    return jsonify(load_json(MOTD_FILE, DEFAULT_MOTD))

@app.route("/api/admin/motd", methods=["POST"])
@admin_required
def admin_motd_set():
    js = request.get_json(force=True)
    ascii_art = str(js.get("ascii") or DEFAULT_MOTD["ascii"])[:20000]
    items = js.get("list") or []
    if not isinstance(items, list): return jsonify({"error":"list must be array"}), 400
    items = [str(x)[:200] for x in items][:20]
    save_json(MOTD_FILE, {"ascii": ascii_art, "list": items})
    return jsonify({"ok": True})

def _fmt_bytes(n):
    for unit in ['B','KB','MB','GB','TB','PB']:
        if n < 1024.0: return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"

def _uptime_str():
    if HAS_PSUTIL:
        secs = int(time.time() - psutil.boot_time())
    else:
        secs = None
        if os.name == "posix" and pathlib.Path("/proc/uptime").exists():
            try: secs = int(float(pathlib.Path("/proc/uptime").read_text().split()[0]))
            except Exception: pass
        if secs is None:
            started = int((DATA / ".server_started").read_text()) if (DATA/".server_started").exists() else int(time.time())
            secs = int(time.time() - started)
            (DATA/".server_started").write_text(str(started))
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    out=[]
    if d: out.append(f"{d}d")
    if h: out.append(f"{h}h")
    if m: out.append(f"{m}m")
    out.append(f"{s}s")
    return " ".join(out)

def _suggest_aesthetic_from_model(start_text, lore_snips):
    """Try to get tags like: neon|cyberpunk|parchment|noir|forest|terminal"""
    cfg = load_providers(); cur = cfg.get("current","ollama")
    try:
        if cur == "ollama":
            base = cfg["ollama"]["api_base"].rstrip("/")
            prompt = (
                "Given the following text and lore, output 1-3 short aesthetic tags (lowercase, hyphenated), comma-separated. "
                "Examples: neon, cyberpunk, parchment, noir, mossy-ruin, terminal, retro-green.\n\n"
                f"TEXT:\n{start_text}\n\nLORE:\n{lore_snips}\n\nTAGS: "
            )
            r = requests.post(f"{base}/api/generate",
                              json={"model":"llama3.1:8b","prompt":prompt,"stream":False}, timeout=12)
            r.raise_for_status()
            raw = r.json().get("response","")
            tags = [t.strip().lower() for t in raw.split(",") if 1<=len(t.strip())<=24][:3]
            return tags or None
        else:
            # OpenRouter/Chutes minimal call
            url = load_providers()[cur]["api_base"].rstrip("/") + "/chat/completions"
            body = {"model":"gpt-3.5-turbo","messages":[
                {"role":"system","content":"Output only comma-separated short aesthetic tags (lowercase)."},
                {"role":"user","content": f"TEXT:\n{start_text}\n\nLORE:\n{lore_snips}\n\nTAGS:"}
            ], "stream": False}
            r = requests.post(url, json=body, headers=headers_for(cur, load_providers()), timeout=12)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            tags = [t.strip().lower() for t in txt.split(",") if 1<=len(t.strip())<=24][:3]
            return tags or None
    except Exception:
        return None

def _heuristic_aesthetic(start_text, lore_snips):
    t = (start_text + " " + lore_snips).lower()
    buckets = [
        (["neon","cyber","night","neon-lit","matrix","neotokyo"], "neon"),
        (["forest","grove","moss","druid","herb","sprout"], "mossy-ruin"),
        (["ink","scroll","ancient","parchment","manuscript"], "parchment"),
        (["noir","rain","cigarette","detective","alley"], "noir"),
        (["terminal","retro","vt100","green"], "terminal"),
        (["sand","dune","desert","spice"], "dunes"),
    ]
    hits = [tag for keys, tag in buckets if any(k in t for k in keys)]
    return list(dict.fromkeys(hits))[:3] or ["terminal"]

@app.route("/api/motd", methods=["GET"])
def api_motd():
    th = get_active_theme() or DEFAULT_THEME
    app_name = th.get("app_name","Local Terminal Chat")

    cfg = load_json(MOTD_FILE, DEFAULT_MOTD)
    ascii_art = cfg.get("ascii") or DEFAULT_MOTD["ascii"]
    items = list(cfg.get("list") or [])

    # dynamic system items
    host = socket.gethostname()
    os_name = f"{platform.system()} {platform.release()}"
    py = platform.python_version()
    cpu = platform.processor() or platform.machine()
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    uptime = _uptime_str()
    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        items = items + [f"Memory: {_fmt_bytes(vm.used)} / {_fmt_bytes(vm.total)}", f"Uptime: {uptime}"]
    else:
        items = items + [f"Uptime: {uptime}"]

    # aesthetic suggestion from first user message + lore snippets
    # read last conv if any thread is provided
    start_text = ""
    lore_snips = ""
    thread_id = request.args.get("thread_id")
    profile_id = request.args.get("profile_id", type=int)
    if thread_id and profile_id:
        conv = load_conv(thread_id)
        for m in conv.get("messages", []):
            if m.get("role") == "user":
                start_text = m.get("content","")[:1500]; break
        # Aggregate quick lore
        for f in LORE_DIR.glob("*.json"):
            db = load_json(f, {})
            for c in (db.get("chunks") or [])[:2]:
                lore_snips += " " + c.get("text","")[:400]

    tags = _suggest_aesthetic_from_model(start_text, lore_snips) or _heuristic_aesthetic(start_text, lore_snips)

    # build combined left/right ascii + info like neofetch
    left_lines = (ascii_art + f"\n{app_name}").splitlines()
    right_lines = [
        f"Host     : {host}",
        f"OS       : {os_name}",
        f"CPU      : {cpu}",
        f"Python   : {py}",
        f"Cols     : {cols}",
        f"Style    : {', '.join(tags)}",
    ] + [f"- {x}" for x in items[:8]]

    width_left = max(len(s) for s in left_lines) if left_lines else 0
    rows = max(len(left_lines), len(right_lines))
    out=[]
    for i in range(rows):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        out.append(l.ljust(width_left+2) + r)

    return jsonify({
        "ascii": "\n".join(out),
        "app_name": app_name,
        "theme": th.get("slug","default"),
        "color": th.get("colors",{}).get("accent","#62ff80"),
        "tags": tags
    })

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

# ---------- RPG STATUS (per profile_id + thread_id) ----------
def rpg_path(profile_id, thread_id):
    p = RPG_DIR / f"{profile_id}_{thread_id}.json"
    if not p.exists():
        # initialize a blank sheet
        blank = {
            "updated_at": int(time.time()),
            "health": [],  # wounds: {id, part, kind, severity(0-100), bleeding(0-3), pain(0-3)}
            "bowels": 20,  # 0-100
            "bladder": 20, # 0-100
            "hunger": 20,  # 0-100 (0 full, 100 starving)
            "thirst": 20,  # 0-100
            "calories": 2000,
            "hydration_ml": 2000,
            "inventory": [] # {id, name, qty, weight}
        }
        save_json(p, blank)
    return p

def load_rpg(profile_id, thread_id): return load_json(rpg_path(profile_id, thread_id), {})
def save_rpg(profile_id, thread_id, obj): save_json(rpg_path(profile_id, thread_id), obj)

@app.route("/api/rpg/status", methods=["GET"])
@login_required
def rpg_get():
    profile_id = int(request.args.get("profile_id"))
    thread_id = request.args.get("thread_id")
    sheet = load_rpg(profile_id, thread_id)
    return jsonify(sheet)

@app.route("/api/rpg/status", methods=["POST"])
@login_required
def rpg_set():
    profile_id = int(request.args.get("profile_id"))
    thread_id = request.args.get("thread_id")
    patch = request.get_json(force=True)
    sheet = load_rpg(profile_id, thread_id)

    # allow replacing sections or full fields
    for k in ("health","bowels","bladder","hunger","thirst","calories","hydration_ml","inventory"):
        if k in patch: sheet[k] = patch[k]
    sheet["updated_at"] = int(time.time())
    save_rpg(profile_id, thread_id, sheet)
    return jsonify({"ok": True})

def _auto_tick_rpg(profile_id, thread_id, sheet, minutes=5):
    """very simple metabolism tick per turn"""
    # baseline increases
    sheet["hunger"] = min(100, sheet.get("hunger",20) + 2)
    sheet["thirst"] = min(100, sheet.get("thirst",20) + 3)
    sheet["bowels"] = min(100, sheet.get("bowels",20) + 1)
    sheet["bladder"] = min(100, sheet.get("bladder",20) + 2)
    # bleeding escalates thirst and hunger slightly
    bleeding = sum(w.get("bleeding",0) for w in sheet.get("health",[]))
    if bleeding:
        sheet["thirst"] = min(100, sheet["thirst"] + bleeding)
        sheet["hunger"] = min(100, sheet["hunger"] + max(0, bleeding-1))
    sheet["updated_at"] = int(time.time())
    save_rpg(profile_id, thread_id, sheet)

# ---------- Telemetry ----------
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

    # Guard: thread belongs to current user
    con = db(); cur = con.cursor()
    cur.execute("""SELECT t.id, p.kind FROM threads t 
                   JOIN profiles p ON p.id=t.profile_id 
                   WHERE t.id=? AND p.user_id=? AND p.id=?""",
                (thread_id, current_user_id(), profile_id))
    row = cur.fetchone(); con.close()
    if not row: return ("Forbidden", 403)
    profile_kind = row["kind"]

    # Save incoming
    conv = load_conv(thread_id)
    conv["messages"].extend(user_msgs)
    recent_msgs = conv["messages"][-18:]

    # Parse structured actions
    try:
        last = user_msgs[-1]["content"]
        if last.lstrip().startswith("{"):
            obj = json.loads(last)
            if isinstance(obj, dict) and "actions" in obj:
                user_msgs[-1]["content"] = "USER_ACTIONS:\n" + json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    user_turn_text = recent_msgs[-1]["content"] if recent_msgs else ""
    rag_ctx = retrieve_lore(lore_names, user_turn_text, k=6)  # lore first
    docs_ctx = None
    # You can also add RAG-by-thread if you kept it earlier:
    if (RAG_DIR / f"{profile_id}_{thread_id}.json").exists():
        docs_ctx = load_json(RAG_DIR / f"{profile_id}_{thread_id}.json", {}).get("docs")
    rag_concat = None
    if docs_ctx:
        # quick keyword RAG
        rag_concat = "[DOCS ATTACHED]\n" + "\n\n".join(f"[{d.get('name')}]" for d in docs_ctx[:6])

    base_messages = inject_memory(recent_msgs)
    if rag_concat:
        base_messages = [{"role":"system", "content": rag_concat}] + base_messages
    if rag_ctx:
        base_messages = [{"role":"system", "content":"Lore Context (authoritative canon; prefer over guesses):\n"+rag_ctx}] + base_messages

    # Upstream
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

        # simple per-turn RPG tick
        if profile_kind == "rpg":
            sheet = load_rpg(profile_id, thread_id)
            _auto_tick_rpg(profile_id, thread_id, sheet)

        # update cross-chat memory
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