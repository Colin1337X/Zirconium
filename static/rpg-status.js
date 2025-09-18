// rpg_status.js — RPG sidebar with nutrients, debuffs, wounds, inventory, manual edits, quick consume.

(function(){
  const css = `
.rpg-side { position: fixed; top: 64px; right: 12px; width: 320px;
  max-height: calc(100vh - 80px); overflow: auto; background: var(--panel); color: var(--fg);
  border: 1px solid var(--border); border-radius: var(--radius, 8px); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  z-index: 1000; padding: 8px 10px; }
.rpg-side h3 { margin: 0 0 6px 0; font-size: 14px; color: var(--accent); }
.rpg-sec { margin-bottom: 8px; }
.rpg-grid { display:grid; grid-template-columns: 1fr 1fr; gap:6px; }
.rpg-l { display:flex; gap:6px; align-items:center; margin: 2px 0; }
.rpg-l input[type="number"], .rpg-grid input[type="number"] { width:100%; }
.rpg-wound { border:1px dashed var(--border); padding:4px; margin:4px 0; }
.rpg-inv-row { display:grid; grid-template-columns: 1fr 44px 56px 26px; gap:6px; align-items:center; margin: 2px 0; }
.rpg-mini { font-size: 11px; color: var(--muted); }
.rpg-btns { display:flex; gap:6px; flex-wrap: wrap; }
.debuff { font-size: 12px; margin: 2px 0; padding: 2px 4px; border-left: 3px solid var(--danger); background: color-mix(in srgb, var(--danger) 12%, transparent); }
.bad { color: var(--danger); }
`;
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);

  function qs(name){ const u = new URL(location.href); return u.searchParams.get(name); }

  function mount(){
    const box = document.createElement('div');
    box.className = 'rpg-side panel';
    box.innerHTML = `
      <h3>RPG Status</h3>

      <div class="rpg-sec">
        <strong>Nutrients</strong>
        <div class="rpg-grid">
          <label>Calories <input id="n_cal" type="number" min="0" step="10"></label>
          <label>Protein (g) <input id="n_pro" type="number" min="0" step="1"></label>
          <label>Water (ml) <input id="n_h2o" type="number" min="0" step="10"></label>
          <label>Vitamins <input id="n_vit" type="number" min="0" max="240" step="1"></label>
          <label>Micros <input id="n_mic" type="number" min="0" max="240" step="1"></label>
        </div>
        <small class="rpg-mini">Targets are set in Character Creator.</small>
      </div>

      <div class="rpg-sec">
        <strong>UI Bars</strong>
        <div class="rpg-grid">
          <label>Hunger (0–100) <input id="ui_hunger" type="number" min="0" max="100"></label>
          <label>Thirst (0–100) <input id="ui_thirst" type="number" min="0" max="100"></label>
          <label>Bowels (0–100) <input id="ui_bowels" type="number" min="0" max="100"></label>
          <label>Bladder (0–100) <input id="ui_bladder" type="number" min="0" max="100"></label>
        </div>
      </div>

      <div class="rpg-sec">
        <strong>Quick Consume</strong>
        <div class="rpg-grid">
          <label>Item kcal <input id="c_kcal" type="number" step="10"></label>
          <label>Protein g <input id="c_pro" type="number" step="1"></label>
          <label>Water ml <input id="c_h2o" type="number" step="10"></label>
          <label>Vitamins <input id="c_vit" type="number" step="1"></label>
          <label>Micros <input id="c_mic" type="number" step="1"></label>
        </div>
        <div class="rpg-btns"><button id="btn_consume">Apply</button></div>
      </div>

      <div class="rpg-sec">
        <strong>Wounds</strong>
        <div id="rpg_wounds"></div>
        <div class="rpg-btns"><button id="rpg_add_wound">+ Wound</button></div>
      </div>

      <div class="rpg-sec">
        <strong>Inventory</strong>
        <div id="rpg_inv"></div>
        <div class="rpg-btns"><button id="rpg_add_item">+ Item</button></div>
      </div>

      <div class="rpg-sec">
        <strong>Debuffs</strong>
        <div id="rpg_debuffs"></div>
      </div>

      <div class="rpg-btns">
        <button id="rpg_save">Save</button>
        <button id="rpg_refresh">Refresh</button>
      </div>
      <div class="rpg-mini" id="rpg_msg"></div>
    `;
    document.body.appendChild(box);
    return box;
  }

  function woundRow(w){
    const id = w.id || Math.random().toString(36).slice(2,8);
    return `
      <div class="rpg-wound" data-id="${id}">
        <div class="rpg-l"><label>Part <input type="text" value="${w.part||''}"></label>
        <label>Kind <input type="text" value="${w.kind||''}"></label></div>
        <div class="rpg-l"><label>Severity <input type="number" min="0" max="100" value="${w.severity??10}"></label>
        <label>Bleed <input type="number" min="0" max="3" value="${w.bleeding??0}"></label>
        <label>Pain <input type="number" min="0" max="3" value="${w.pain??0}"></label></div>
        <div class="rpg-btns"><button class="w-del">Delete</button></div>
      </div>`;
  }

  function invRow(it){
    const id = it.id || Math.random().toString(36).slice(2,8);
    return `
      <div class="rpg-inv-row" data-id="${id}">
        <input type="text" placeholder="name" value="${it.name||''}">
        <input type="number" placeholder="qty" min="0" step="1" value="${it.qty??1}">
        <input type="number" placeholder="weight" min="0" step="0.1" value="${it.weight??0}">
        <button class="i-del">×</button>
      </div>`;
  }

  async function fetchJSON(url){ const r = await fetch(url); return r.json(); }

  function ids(){ return { pid: qs('profile_id'), tid: qs('thread_id') }; }

  async function loadState(box){
    const {pid, tid} = ids();
    if (!pid || !tid) { box.style.display='none'; return; }
    box.style.display='block';
    const j = await fetchJSON(`/api/rpg/status?profile_id=${encodeURIComponent(pid)}&thread_id=${encodeURIComponent(tid)}`);
    const N = j.nutrients || {};
    box.querySelector('#n_cal').value = Math.round(N.calories||0);
    box.querySelector('#n_pro').value = Math.round(N.protein_g||0);
    box.querySelector('#n_h2o').value = Math.round(N.water_ml||0);
    box.querySelector('#n_vit').value = Math.round(N.vitamins||0);
    box.querySelector('#n_mic').value = Math.round(N.micronutrients||0);

    box.querySelector('#ui_hunger').value = Math.round(j.hunger||0);
    box.querySelector('#ui_thirst').value = Math.round(j.thirst||0);
    box.querySelector('#ui_bowels').value = Math.round(j.bowels||0);
    box.querySelector('#ui_bladder').value = Math.round(j.bladder||0);

    const wounds = j.health || [];
    const inv = j.inventory || [];
    const wWrap = box.querySelector('#rpg_wounds'); wWrap.innerHTML = wounds.map(woundRow).join('');
    const iWrap = box.querySelector('#rpg_inv'); iWrap.innerHTML = inv.map(invRow).join('');
    wWrap.querySelectorAll('.w-del').forEach(btn => btn.addEventListener('click', ()=>btn.closest('.rpg-wound').remove()));
    iWrap.querySelectorAll('.i-del').forEach(btn => btn.addEventListener('click', ()=>btn.closest('.rpg-inv-row').remove()));

    const dWrap = box.querySelector('#rpg_debuffs');
    dWrap.innerHTML = (j.debuffs||[]).map(d=>`<div class="debuff"><strong>${d.name}</strong> (${d.severity}) — ${d.text}</div>`).join('') || '<span class="rpg-mini">None</span>';
  }

  function collect(box){
    const N = {
      calories: +box.querySelector('#n_cal').value,
      protein_g: +box.querySelector('#n_pro').value,
      water_ml: +box.querySelector('#n_h2o').value,
      vitamins: +box.querySelector('#n_vit').value,
      micronutrients: +box.querySelector('#n_mic').value
    };
    const sheet = {
      nutrients: N,
      hunger: +box.querySelector('#ui_hunger').value,
      thirst: +box.querySelector('#ui_thirst').value,
      bowels: +box.querySelector('#ui_bowels').value,
      bladder: +box.querySelector('#ui_bladder').value,
      health: [],
      inventory: []
    };
    box.querySelectorAll('#rpg_wounds .rpg-wound').forEach(div=>{
      const [part, kind] = div.querySelectorAll('input[type="text"]');
      const [sev, bleed, pain] = div.querySelectorAll('input[type="number"]');
      sheet.health.push({ id: div.dataset.id, part: part.value.trim(), kind: kind.value.trim(),
        severity: +sev.value, bleeding: +bleed.value, pain: +pain.value });
    });
    box.querySelectorAll('#rpg_inv .rpg-inv-row').forEach(div=>{
      const [name, qty, weight] = div.querySelectorAll('input');
      sheet.inventory.push({ id: div.dataset.id, name: name.value.trim(), qty: +qty.value, weight: +weight.value });
    });
    return sheet;
  }

  async function save(box){
    const {pid, tid} = ids(); if (!pid || !tid) return;
    const body = collect(box);
    const r = await fetch(`/api/rpg/status?profile_id=${encodeURIComponent(pid)}&thread_id=${encodeURIComponent(tid)}`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    box.querySelector('#rpg_msg').textContent = r.ok ? 'Saved.' : 'Save failed.';
    if (r.ok) setTimeout(()=>loadState(box), 200);
  }

  async function consume(box){
    const {pid, tid} = ids(); if (!pid || !tid) return;
    const item = {
      calories: +box.querySelector('#c_kcal').value||0,
      protein_g: +box.querySelector('#c_pro').value||0,
      water_ml: +box.querySelector('#c_h2o').value||0,
      vitamins: +box.querySelector('#c_vit').value||0,
      micronutrients: +box.querySelector('#c_mic').value||0
    };
    const r = await fetch(`/api/rpg/consume`, {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({profile_id: pid, thread_id: tid, item})});
    const j = await r.json();
    box.querySelector('#rpg_msg').textContent = r.ok ? 'Consumed.' : 'Failed.';
    loadState(box);
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    const box = mount();
    box.querySelector('#rpg_add_wound').addEventListener('click', ()=>{
      const wrap = box.querySelector('#rpg_wounds');
      const temp = document.createElement('div'); temp.innerHTML = woundRow({severity:10, bleeding:0, pain:0});
      wrap.appendChild(temp.firstElementChild);
      wrap.querySelectorAll('.w-del').forEach(btn => btn.onclick = ()=>btn.closest('.rpg-wound').remove());
    });
    box.querySelector('#rpg_add_item').addEventListener('click', ()=>{
      const wrap = box.querySelector('#rpg_inv');
      const temp = document.createElement('div'); temp.innerHTML = invRow({qty:1, weight:0});
      wrap.appendChild(temp.firstElementChild);
      wrap.querySelectorAll('.i-del').forEach(btn => btn.onclick = ()=>btn.closest('.rpg-inv-row').remove());
    });
    box.querySelector('#rpg_save').addEventListener('click', ()=>save(box));
    box.querySelector('#rpg_refresh').addEventListener('click', ()=>loadState(box));
    box.querySelector('#btn_consume').addEventListener('click', ()=>consume(box));
    loadState(box).catch(()=>{});
    setInterval(()=>loadState(box), 5000);
    window.rpgStatusRefresh = ()=>loadState(box);
  });
})();