(function(){
  function qs(name){ const u = new URL(location.href); return u.searchParams.get(name); }
  function el(id){ return document.getElementById(id); }

  document.addEventListener('DOMContentLoaded', ()=>{
    // prefill from URL if present
    if (qs('profile_id')) el('profile_id').value = qs('profile_id');
    if (qs('thread_id')) el('thread_id').value = qs('thread_id');

    el('mode').addEventListener('change', ()=>{
      document.getElementById('manualTargets').style.display = (el('mode').value === 'manual') ? 'block' : 'none';
    });

    el('create').addEventListener('click', async ()=>{
      const body = {
        profile_id: +el('profile_id').value,
        thread_id: el('thread_id').value.trim(),
        name: el('name').value.trim(),
        desc: el('desc').value.trim(),
        height_cm: +el('height_cm').value,
        weight_kg: +el('weight_kg').value,
        mode: el('mode').value
      };
      if (body.mode === 'manual'){
        body.targets = {
          calories: +el('t_cal').value,
          protein_g: +el('t_pro').value,
          water_ml: +el('t_h2o').value,
          vitamins: +el('t_vit').value,
          micronutrients: +el('t_mic').value
        };
      }
      const r = await fetch('/api/rpg/create_character', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
      });
      const ok = r.ok;
      el('msg').textContent = ok ? 'Character saved. Jump into the thread to begin.' : 'Failed to save.';
    });
  });
})();
