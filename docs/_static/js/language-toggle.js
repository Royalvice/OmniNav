(function () {
  function normalize(path) {
    if (!path || path === '') return '/';
    return path;
  }

  function targetPaths() {
    const path = normalize(window.location.pathname);
    const isZh = path === '/zh' || path.startsWith('/zh/');

    if (isZh) {
      const stripped = path.replace(/^\/zh/, '') || '/';
      return {
        isZh: true,
        en: stripped,
        zh: path,
      };
    }

    const base = path === '/' ? '/' : path;
    const zh = base === '/' ? '/zh/' : '/zh' + base;
    return {
      isZh: false,
      en: base,
      zh,
    };
  }

  function createSwitch() {
    const article = document.querySelector('article.bd-article');
    if (!article) return;
    if (article.querySelector('.lang-switch')) return;

    const { isZh, en, zh } = targetPaths();

    const wrap = document.createElement('div');
    wrap.className = 'lang-switch';

    const enLink = document.createElement('a');
    enLink.href = en;
    enLink.textContent = 'English';
    if (!isZh) enLink.classList.add('active');

    const zhLink = document.createElement('a');
    zhLink.href = zh;
    zhLink.textContent = '中文';
    if (isZh) zhLink.classList.add('active');

    wrap.appendChild(enLink);
    wrap.appendChild(zhLink);
    article.insertBefore(wrap, article.firstChild);
  }

  document.addEventListener('DOMContentLoaded', createSwitch);
})();
