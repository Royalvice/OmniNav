(function () {
  const KEY = 'omninav_doc_lang';
  const ZH = 'zh';
  const EN = 'en';

  function detectDefaultLang() {
    const navLang = (navigator.language || '').toLowerCase();
    return navLang.startsWith('zh') ? ZH : EN;
  }

  function getLang() {
    const saved = localStorage.getItem(KEY);
    if (saved === ZH || saved === EN) {
      return saved;
    }
    return detectDefaultLang();
  }

  function setLang(lang) {
    const normalized = lang === EN ? EN : ZH;
    localStorage.setItem(KEY, normalized);

    document.querySelectorAll('.lang-zh').forEach((el) => {
      el.style.display = normalized === ZH ? '' : 'none';
    });
    document.querySelectorAll('.lang-en').forEach((el) => {
      el.style.display = normalized === EN ? '' : 'none';
    });

    document.querySelectorAll('.lang-switch-btn').forEach((btn) => {
      const isActive = btn.dataset.lang === normalized;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    });
  }

  function createSwitch() {
    const article = document.querySelector('article.bd-article');
    if (!article) {
      return;
    }

    const existing = article.querySelector('.lang-switch');
    if (existing) {
      return;
    }

    const wrap = document.createElement('div');
    wrap.className = 'lang-switch';

    const zhBtn = document.createElement('button');
    zhBtn.className = 'lang-switch-btn';
    zhBtn.dataset.lang = ZH;
    zhBtn.type = 'button';
    zhBtn.textContent = '中文';

    const enBtn = document.createElement('button');
    enBtn.className = 'lang-switch-btn';
    enBtn.dataset.lang = EN;
    enBtn.type = 'button';
    enBtn.textContent = 'English';

    wrap.appendChild(zhBtn);
    wrap.appendChild(enBtn);

    article.insertBefore(wrap, article.firstChild);

    wrap.querySelectorAll('.lang-switch-btn').forEach((btn) => {
      btn.addEventListener('click', () => setLang(btn.dataset.lang));
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    createSwitch();
    setLang(getLang());
  });
})();
