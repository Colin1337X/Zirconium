// motd.js — renders customizable ASCII + MOTD list; fetches aesthetic tags tied to thread/profile.

(function(){
  const ensureVT = () => {
    if (document.querySelector('link[href*="VT323"]')) return;
    const l1 = document.createElement('link');
    l1.rel = 'preconnect'; l1.href = 'https://fonts.gstatic.com'; l1.crossOrigin = 'anonymous';
    const l2 = document.createElement('link');
    l2.rel = 'stylesheet';
    l2.href = 'https://fonts.googleapis.com/css2?family=VT323&display=swap';
    document.head.appendChild(l1); document.head.appendChild(l2);
  };

  function mount() {
    const wrap = document.createElement('div');
    wrap.className = 'panel motd-panel';
    const pre = document.createElement('pre');
    pre.className = 'motd-ascii';
    const ul = document.createElement('ul');
    ul.className = 'motd-list';
    wrap.appendChild(pre);
    wrap.appendChild(ul);

    const anchor = document.querySelector('.topbar') || document.body.firstElementChild;
    if (anchor && anchor.parentNode) anchor.parentNode.insertBefore(wrap, anchor.nextSibling);
    else document.body.insertBefore(wrap, document.body.firstChild);

    return {pre, ul, wrap};
  }

  function qs(name){
    const url = new URL(location.href);
    return url.searchParams.get(name);
  }

  async function load(pre, ul){
    const q = new URLSearchParams();
    const pid = qs('profile_id'); const tid = qs('thread_id');
    if (pid) q.set('profile_id', pid);
    if (tid) q.set('thread_id', tid);
    const r = await fetch('/api/motd?'+q.toString());
    const j = await r.json();
    pre.textContent = j.ascii || 'Welcome.';
    pre.style.setProperty('--motd-accent', j.color || '#62ff80');
    ul.innerHTML = '';
    const items = (j.tags && j.tags.length) ? [`Style: ${j.tags.join(', ')}`] : [];
    (items).forEach(s => {
      const li = document.createElement('li'); li.textContent = s; ul.appendChild(li);
    });
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    ensureVT();
    const {pre, ul} = mount();
    load(pre, ul).catch(()=>{ pre.textContent = 'Welcome.'; });
  });
})();