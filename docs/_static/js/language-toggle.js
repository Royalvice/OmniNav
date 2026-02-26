(function () {
  const CONTENT_ROOTS = new Set([
    "getting_started",
    "api_reference",
    "index.html",
  ]);

  function splitPath(pathname) {
    const raw = (pathname || "").split("/").filter(Boolean);
    return raw;
  }

  function joinPath(parts, trailingSlash) {
    const p = "/" + parts.join("/");
    if (trailingSlash && !p.endsWith("/")) {
      return p + "/";
    }
    return p === "" ? "/" : p;
  }

  function targetPaths() {
    const path = window.location.pathname || "/";
    const trailingSlash = path.endsWith("/");
    const parts = splitPath(path);
    const zhIdx = parts.indexOf("zh");
    const isZh = zhIdx >= 0;

    if (isZh) {
      const enParts = parts.slice(0, zhIdx).concat(parts.slice(zhIdx + 1));
      return {
        isZh: true,
        en: joinPath(enParts, trailingSlash),
        zh: path,
      };
    }

    // Insert `zh` right before first content root segment.
    // This keeps prefixes like /OmniNav/ or /docs/_build/html/ untouched.
    let insertIdx = parts.findIndex((x) => CONTENT_ROOTS.has(x));
    if (insertIdx < 0) {
      insertIdx = parts.length;
    }
    const zhParts = parts.slice();
    zhParts.splice(insertIdx, 0, "zh");
    return {
      isZh: false,
      en: path,
      zh: joinPath(zhParts, trailingSlash),
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
