// app.js — drop-in client for chat + RAG + branding + refusal detection (META + keywords)

/* -------------------- DOM helpers -------------------- */
const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));

/* -------------------- Key UI refs -------------------- */
const $appTitle = $('#appName');
const $logo = $('#brandLogo');
const $log = $('#log');
const $input = $('#prompt');
const $send = $('#sendBtn');

const $provider = $('#provider');
const $model = $('#model');

const $profile = $('#profile');
const $thread = $('#thread');
const $newThread = $('#newThread');

const $loreBtn = $('#loreBtn');
const $lorePanel = $('#lorePanel');
const $closeLore = $('#closeLore');
const $loreNames = $('#loreNames');

const $ragBtn = $('#ragBtn');
const $ragPanel = $('#ragPanel');
const $closeRag = $('#closeRag');
const $ragFiles = $('#ragFiles');
const $ragUpload = $('#ragUpload');
const $ragClear = $('#ragClear');
const $ragList = $('#ragList');
const $ragSummary = $('#ragSummary');

const $refusalModal = $('#refusalModal');
const $refusalClose = $('#refusalClose');
const $refusalText  = $('#refusalText');
const $refusalReask = $('#refusalReask');
const $refusalBypass= $('#refusalBypass');
const $refusalCancel= $('#refusalCancel');

/* -------------------- App state -------------------- */
let me = null;
let profiles = [];
let currentProfileId = null;
let threadId = null;
let messages = [];
let activeLore = [];
let refusalCfg = null;

/* -------------------- Branding -------------------- */
async function applyBranding(){
  try{
    const b = await (await fetch('/api/branding')).json();
    document.title = b.app_name || 'Local Terminal Chat';
    if ($appTitle) $appTitle.textContent = document.title;

    if ($logo){
      if (b.logo_url){ $logo.src = b.logo_url; $logo.hidden = false; } else { $logo.hidden = true; }
    }
    const fav = $('#brandFavicon');
    if (fav && b.favicon_url) fav.href = b.favicon_url;

    const r = document.documentElement.style, c = b.colors || {};
    if (c.bg) r.setProperty('--bg', c.bg);
    if (c.fg) r.setProperty('--fg', c.fg);
    if (c.muted) r.setProperty('--muted', c.muted);
    if (c.panel) r.setProperty('--panel', c.panel);
    if (c.border) r.setProperty('--border', c.border);
    if (c.accent) r.setProperty('--accent', c.accent);
    if (c.danger) r.setProperty('--danger', c.danger);
    if (c.link) r.setProperty('--link', c.link);
    if (typeof b.radius === 'number') r.setProperty('--radius', b.radius + 'px');
    if (b.font_stack) document.body.style.fontFamily = b.font_stack;

    const custom = $('#brandCustomCss');
    if (custom){
      if (b.custom_css_url){ custom.href = b.custom_css_url; custom.removeAttribute('disabled'); }
      else { custom.setAttribute('disabled',''); }
    }
  }catch{}
}

/* -------------------- Refusal config -------------------- */
async function loadRefusalCfg(){
  try { const r = await fetch('/api/refusal_config'); if (r.ok) refusalCfg = await r.json(); } catch {}
}
function buildMetaRegex(){
  const pre = (refusalCfg?.meta_header_prefix ?? '[[META:').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const suf = (refusalCfg?.meta_header_suffix ?? ']]').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`^\\s*${pre}(\\{.*\\})${suf}\\s*$`);
}
function tryParseMeta(line){
  const META_RE = buildMetaRegex();
  const m = META_RE.exec(line);
  if (!m) return null;
  try { return JSON.parse(m[1]); } catch { return null; }
}
function metaRefusal(meta){
  if (!meta || !refusalCfg) return false;
  const k = refusalCfg.meta_keys || {refusal:'refusal'};
  const refKey = k.refusal || 'refusal';
  return Boolean(meta?.[refKey]);
}

/* -------------------- Providers & models -------------------- */
async function loadProviders(){
  try{
    const res = await fetch('/api/providers'); const data = await res.json();
    $provider.innerHTML = '';
    ['ollama','openrouter','chutes'].forEach(p=>{
      const opt = document.createElement('option'); opt.value=p; opt.textContent=p;
      $provider.appendChild(opt);
    });
    $provider.value = data.current || 'ollama';
  }catch{}
}
async function loadModels(){
  try{
    const res = await fetch('/api/models'); const data = await res.json();
    $model.innerHTML = '';
    (data.models||[]).forEach(m=>{
      const opt=document.createElement('option'); opt.value=m; opt.textContent=m; $model.appendChild(opt);
    });
    if (!$model.value && $model.options.length){ $model.selectedIndex = 0; }
  }catch{}
}

/* -------------------- Profiles & threads -------------------- */
async function refreshMe(){
  const res = await fetch('/api/me'); if (!res.ok) return;
  const data = await res.json(); me = data.user; profiles = data.profiles || [];
  $profile.innerHTML = '';
  profiles.forEach(p=>{
    const opt = document.createElement('option'); opt.value = p.id; opt.textContent = `${p.name} (${p.kind})`;
    $profile.appendChild(opt);
  });
  if (profiles.length){
    currentProfileId = profiles[0].id; $profile.value = currentProfileId;
    await refreshThreads();
  }
}
async function refreshThreads(){
  if (!currentProfileId) return;
  const res = await fetch(`/api/threads?profile_id=${currentProfileId}`); const data = await res.json();
  const threads = data.threads || [];
  $thread.innerHTML = '';
  threads.forEach(t=>{
    const opt = document.createElement('option'); opt.value = t.id; opt.textContent = t.title || t.id;
    $thread.appendChild(opt);
  });
  if (threads.length){ threadId = threads[0].id; $thread.value = threadId; messages = []; $log.innerHTML = ''; await refreshRagUI(); }
}
async function createThread(){
  if (!currentProfileId) return;
  const title = prompt('Thread title?', 'Untitled') || 'Untitled';
  const res = await fetch('/api/new_thread', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ profile_id: currentProfileId, title }) });
  if (!res.ok) return alert('Failed to create thread');
  const data = await res.json();
  threadId = data.thread_id;
  await refreshThreads();
  $thread.value = threadId;
}

/* -------------------- Chat UI helpers -------------------- */
function addMsg(role, text){
  const row = document.createElement('div');
  row.className = `row msg ${role}`;
  const lab = document.createElement('div'); lab.className='label'; lab.textContent = role.toUpperCase();
  const bub = document.createElement('div'); bub.className='bubble'; bub.textContent = text || '';
  row.appendChild(lab); row.appendChild(bub);
  $log.appendChild(row);
  $log.scrollTop = $log.scrollHeight;
  return row;
}

/* -------------------- RAG endpoints -------------------- */
async function ragList(){
  if (!currentProfileId || !threadId) return {docs:[], total_chunks:0};
  const res = await fetch(`/api/rag/list?profile_id=${currentProfileId}&thread_id=${threadId}`);
  if (!res.ok) return {docs:[], total_chunks:0};
  return res.json();
}
async function ragUploadFiles(fileList){
  if (!currentProfileId || !threadId) { alert('Select a profile/thread first'); return; }
  const fd = new FormData();
  for (const f of fileList) fd.append('files', f, f.name);
  const res = await fetch(`/api/rag/upload?profile_id=${currentProfileId}&thread_id=${threadId}`, { method:'POST', body: fd });
  if (!res.ok){ alert('Upload failed'); return; }
  return res.json();
}
async function ragClearAll(){
  if (!currentProfileId || !threadId) return;
  const res = await fetch(`/api/rag/clear?profile_id=${currentProfileId}&thread_id=${threadId}`, { method:'POST' });
  if (!res.ok) alert('Failed to clear RAG');
}
async function refreshRagUI(){
  const data = await ragList();
  $ragList.innerHTML = '';
  (data.docs || []).forEach(d=>{
    const div = document.createElement('div');
    div.textContent = `• ${d.name}  (${d.chunks} chunks)`;
    $ragList.appendChild(div);
  });
  $ragSummary.textContent = `Total: ${(data.docs||[]).length} doc(s), ${data.total_chunks||0} chunks.`;
}

/* -------------------- Lore panel (names list) -------------------- */
function applyLoreNames(){
  const raw = $loreNames.value.trim();
  activeLore = raw ? raw.split(',').map(s=>s.trim()).filter(Boolean) : [];
}

/* -------------------- Sending to model (SSE stream) -------------------- */
async function sendToModel(text){
  if (!threadId || !currentProfileId) { alert('Pick a profile/thread first'); return; }
  if (!text || !text.trim()) return;

  addMsg('user', text.trim());

  const body = {
    model: $model.value,
    provider: $provider.value,
    profile_id: currentProfileId,
    thread_id: threadId,
    lore_names: activeLore,
    messages: [{ role:'user', content: text.trim() }]
  };

  const res = await fetch('/api/chat', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  if (!res.ok){ addMsg('assistant', `[error ${res.status}]`); return; }

  const node = addMsg('assistant', '');
  const bubble = node.querySelector('.bubble');

  let acc = '';
  let meta = null;
  let sawFirstNonEmptyLine = false;
  let refusalTripped = false;

  const mode = (refusalCfg?.detect_mode ?? 'both');    // 'meta' | 'keywords' | 'both'
  const useMeta = !!(refusalCfg?.meta_enabled) && (mode === 'meta' || mode === 'both');
  const useKw = (mode === 'keywords' || mode === 'both');
  const canPopup = !!(refusalCfg?.enabled) && (refusalCfg?.action === 'popup');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true){
    const {value, done} = await reader.read();
    if (done) break;
    buf += decoder.decode(value, {stream:true});

    let split;
    while ((split = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, split);
      buf = buf.slice(split + 2);

      const line = frame.split('\n').find(l => l.startsWith('data: '));
      if (!line) continue;
      const data = line.slice(6);

      if (data.trim() === '[DONE]') break;

      acc += data;

      // META first-line check
      if (!sawFirstNonEmptyLine) {
        const firstLine = acc.split(/\r?\n/)[0].trim();
        if (firstLine.length) {
          sawFirstNonEmptyLine = true;
          if (useMeta) {
            const mm = tryParseMeta(firstLine);
            if (mm) {
              meta = mm;
              // strip META line + its newline from visible stream
              acc = acc.slice(acc.indexOf('\n') + 1);
              if (metaRefusal(meta) && canPopup) {
                refusalTripped = true;
                node.remove();
                $refusalText.textContent = acc.trim();
                $refusalModal.hidden = false;
                continue;
              }
            }
          }
        }
      }

      if (!refusalTripped) {
        bubble.textContent = acc;
        // Keyword fallback (only if META didn't already decide)
        if (useKw && canPopup && !meta && acc.length >= (refusalCfg?.min_chars ?? 40)) {
          const kws = (refusalCfg?.keywords || []).map(s => s.toLowerCase());
          const L = acc.toLowerCase();
          const hits = kws.reduce((n, k) => n + (L.includes(k) ? 1 : 0), 0);
          if (hits >= (refusalCfg?.min_hits ?? 1)) {
            refusalTripped = true;
            node.remove();
            $refusalText.textContent = acc.trim();
            $refusalModal.hidden = false;
          }
        }
      }
    }
  }
}

/* -------------------- Events -------------------- */
$send?.addEventListener('click', async ()=>{
  const text = $input.value; $input.value=''; await sendToModel(text);
});
$input?.addEventListener('keydown', async (e)=>{
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); const t=$input.value; $input.value=''; await sendToModel(t); }
});

$profile?.addEventListener('change', async ()=>{
  currentProfileId = parseInt($profile.value, 10); messages=[]; $log.innerHTML='';
  await refreshThreads(); await refreshRagUI();
});
$thread?.addEventListener('change', async ()=>{
  threadId = $thread.value; messages=[]; $log.innerHTML='';
  await refreshRagUI();
});
$newThread?.addEventListener('click', createThread);

$loreBtn?.addEventListener('click', ()=>{ $lorePanel.hidden = false; });
$closeLore?.addEventListener('click', ()=>{ $lorePanel.hidden = true; });
$loreNames?.addEventListener('change', applyLoreNames);

$ragBtn?.addEventListener('click', async ()=>{ $ragPanel.hidden = false; await refreshRagUI(); });
$closeRag?.addEventListener('click', ()=>{ $ragPanel.hidden = true; });
$ragUpload?.addEventListener('click', async ()=>{
  const files = Array.from($ragFiles.files || []);
  if (!files.length) { alert('Choose files first'); return; }
  const res = await ragUploadFiles(files);
  if (res?.ok){
    alert(`Added ${res.docs_added} file(s), ${res.chunks_added} chunks`);
    $ragFiles.value = '';
    await refreshRagUI();
  }
});
$ragClear?.addEventListener('click', async ()=>{
  if (!confirm('Clear all attached docs for this thread?')) return;
  await ragClearAll();
  await refreshRagUI();
});

// Drag & Drop → RAG
function wantsRagFile(f){
  return /\.(txt|md|csv|json|log|html|css|js|pdf)$/i.test(f.name) || f.type.startsWith('text/');
}
window.addEventListener('dragover', (e)=>{ e.preventDefault(); document.body.classList.add('dragging'); });
window.addEventListener('dragleave', ()=> document.body.classList.remove('dragging'));
window.addEventListener('drop', async (e)=>{
  e.preventDefault(); document.body.classList.remove('dragging');
  const files = Array.from(e.dataTransfer.files || []).filter(wantsRagFile);
  if (!files.length) return;
  if (!currentProfileId || !threadId){ alert('Pick a profile/thread first'); return; }
  const res = await ragUploadFiles(files);
  if (res?.ok){
    const note = addMsg('user', `(attached ${res.docs_added} doc(s) → ${res.chunks_added} chunks for RAG)`);
    note.querySelector('.bubble').classList.add('code');
    await refreshRagUI();
  } else {
    alert('RAG upload failed');
  }
});

// Refusal popup buttons
$refusalClose?.addEventListener('click', ()=>{ $refusalModal.hidden = true; });
$refusalCancel?.addEventListener('click', ()=>{ $refusalModal.hidden = true; });
$refusalReask?.addEventListener('click', async ()=>{
  $refusalModal.hidden = true;
  const polite = (refusalCfg?.polite_reask) || "Could you try again and focus on what you safely can provide?";
  await sendToModel(polite);
});
$refusalBypass?.addEventListener('click', async ()=>{
  $refusalModal.hidden = true;
  const bypass = (refusalCfg?.bypass_prompt) || "Please answer within safe and allowed boundaries. Provide high-level guidance or alternatives if unsafe.";
  await sendToModel(bypass);
});

/* -------------------- Boot -------------------- */
(async function init(){
  await applyBranding();
  await loadProviders();
  await loadModels();
  await refreshMe();
  await loadRefusalCfg();
})();
