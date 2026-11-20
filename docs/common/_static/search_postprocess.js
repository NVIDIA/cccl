"use strict";

(function () {
  const maxBreadcrumbResults = 10;

  const decodeEntities = (value) =>
    String(value || "")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&amp;/g, "&")
      .replace(/&quot;/g, '"')
      .replace(/&#39;|&apos;/g, "'")
      .replace(/&nbsp;/g, " ")
      .replace(/&hellip;/g, "...");

  const getDocLinkSuffix = () =>
    (typeof DOCUMENTATION_OPTIONS !== "undefined" &&
      DOCUMENTATION_OPTIONS.LINK_SUFFIX) ||
    ".html";

  const pageInfoCache = new Map();

  const getDocTitle = (docName) => {
    if (
      typeof Search === "undefined" ||
      !Search._index ||
      !Array.isArray(Search._index.docnames) ||
      !Array.isArray(Search._index.titles)
    ) {
      return null;
    }

    const index = Search._index.docnames.indexOf(docName);
    return index >= 0 ? Search._index.titles[index] : null;
  };

  const getResultDocNameFromHref = (href) => {
    if (!href) {
      return null;
    }

    const withoutAnchor = String(href).split("#", 1)[0];
    const linkSuffix = getDocLinkSuffix();
    const suffixIndex = withoutAnchor.lastIndexOf(linkSuffix);
    if (suffixIndex < 0) {
      return null;
    }

    let docName = withoutAnchor.slice(0, suffixIndex);
    if (docName.startsWith("./")) {
      docName = docName.slice(2);
    }

    const contentRoot =
      document?.documentElement?.dataset?.content_root || "";
    if (contentRoot && docName.startsWith(contentRoot)) {
      docName = docName.slice(contentRoot.length);
    }

    return docName.replace(/^\/+/, "");
  };

  const getPageInfo = async (docName) => {
    if (pageInfoCache.has(docName)) {
      return pageInfoCache.get(docName);
    }

    const infoPromise = (async () => {
      const pageUrl = `${docName}${getDocLinkSuffix()}`;
      const response = await fetch(pageUrl);
      const html = await response.text();
      const parsed = new DOMParser().parseFromString(html, "text/html");

      const pageHeading = parsed.querySelector("h1");
      const breadcrumbLinks = Array.from(
        parsed.querySelectorAll(".breadcrumb-item a.nav-link"),
      );
      const breadcrumbs = breadcrumbLinks
        .map((breadcrumbLink) => {
          const rawHref = breadcrumbLink.getAttribute("href");
          if (!rawHref) {
            return null;
          }

          return {
            href: new URL(rawHref, response.url).href,
            title: breadcrumbLink.textContent?.trim() || null,
          };
        })
        .filter((breadcrumb) => breadcrumb && breadcrumb.title);

      return {
        pageTitle:
          pageHeading?.textContent?.replace(/#\s*$/, "").trim() ||
          getDocTitle(docName) ||
          null,
        breadcrumbs,
      };
    })().catch(() => null);

    pageInfoCache.set(docName, infoPromise);
    return infoPromise;
  };

  const addBreadcrumbTrail = async (listItem) => {
    if (
      !listItem ||
      listItem.dataset.ccclBreadcrumbsAttached === "true" ||
      listItem.dataset.ccclBreadcrumbsPending === "true"
    ) {
      return;
    }
    listItem.dataset.ccclBreadcrumbsPending = "true";

    const primaryLink = listItem.querySelector("a");
    if (!primaryLink) {
      delete listItem.dataset.ccclBreadcrumbsPending;
      return;
    }

    const primaryTitle = primaryLink.textContent?.trim() || "";
    const href = primaryLink.getAttribute("href");
    const docName = getResultDocNameFromHref(href);
    if (!docName) {
      delete listItem.dataset.ccclBreadcrumbsPending;
      return;
    }

    const pageInfo = await getPageInfo(docName);
    if (!pageInfo) {
      delete listItem.dataset.ccclBreadcrumbsPending;
      return;
    }

    const pageTitle = pageInfo.pageTitle || getDocTitle(docName);
    const breadcrumbs = [...(pageInfo.breadcrumbs || [])];
    if (pageTitle && primaryTitle && pageTitle !== primaryTitle) {
      breadcrumbs.push({
        href: `${docName}${getDocLinkSuffix()}`,
        title: pageTitle,
      });
    }

    if (breadcrumbs.length === 0) {
      delete listItem.dataset.ccclBreadcrumbsPending;
      return;
    }

    const breadcrumbContainer = document.createElement("div");
    breadcrumbContainer.className = "cccl-search-breadcrumbs";

    breadcrumbs.forEach((breadcrumb) => {
      const breadcrumbItem = document.createElement("span");
      breadcrumbItem.className = "breadcrumb-item";
      const breadcrumbLink = document.createElement("a");
      breadcrumbLink.href = breadcrumb.href;
      breadcrumbLink.textContent = breadcrumb.title;
      breadcrumbItem.appendChild(breadcrumbLink);
      breadcrumbContainer.appendChild(breadcrumbItem);
    });

    listItem.insertBefore(breadcrumbContainer, primaryLink.nextSibling);
    listItem.dataset.ccclBreadcrumbsAttached = "true";
    delete listItem.dataset.ccclBreadcrumbsPending;
  };

  const installResultDecorator = () => {
    if (
      typeof Search === "undefined" ||
      Search.__ccclResultDecoratorInstalled ||
      typeof MutationObserver === "undefined"
    ) {
      return;
    }

    const originalPerformSearch = Search.performSearch;
    Search.performSearch = (...args) => {
      const result = originalPerformSearch(...args);
      const output = Search.output;
      if (!output) {
        return result;
      }

      const decorateTopResults = () => {
        Array.from(output.querySelectorAll("li"))
          .slice(0, maxBreadcrumbResults)
          .forEach(addBreadcrumbTrail);
      };

      if (Search.__ccclResultsObserver) {
        Search.__ccclResultsObserver.disconnect();
      }

      decorateTopResults();

      const observer = new MutationObserver((mutations) => {
        decorateTopResults();
      });

      observer.observe(output, { childList: true, subtree: true });
      Search.__ccclResultsObserver = observer;
      Search.__ccclResultDecoratorInstalled = true;
      return result;
    };
  };

  const installPostprocess = () => {
    if (typeof Search === "undefined" || Search.__ccclDedupInstalled) {
      return;
    }

    const originalPerformSearch = Search._performSearch;
    Search._performSearch = (...args) => {
      const results = originalPerformSearch(...args);

      // Sphinx keeps results in low->high score order and displays via pop().
      // Walk from the end so we see the best-ranked result first, but prefer
      // canonical page links without anchors when collapsing duplicates.
      const chosen = new Map();
      for (let i = results.length - 1; i >= 0; --i) {
        const result = results[i];
        const title = String(result[1] || "").toLowerCase();
        const filename = String(result[5] || "");
        const key = `${filename}\0${title}`;
        const anchor = String(result[2] || "");
        const existing = chosen.get(key);
        if (!existing) {
          chosen.set(key, result);
          continue;
        }

        const existingAnchor = String(existing[2] || "");
        const prefersCurrent = existingAnchor && !anchor;
        if (prefersCurrent) {
          chosen.set(key, result);
        }
      }

      const deduped = [];
      const emitted = new Set();
      for (let i = 0; i < results.length; ++i) {
        const result = results[i];
        const title = String(result[1] || "").toLowerCase();
        const filename = String(result[5] || "");
        const key = `${filename}\0${title}`;
        if (emitted.has(key)) {
          continue;
        }
        const winner = chosen.get(key);
        if (winner) {
          winner[1] = decodeEntities(winner[1]);
          winner[3] = decodeEntities(winner[3]);
          deduped.push(winner);
          emitted.add(key);
        }
      }
      return deduped;
    };

    Search.__ccclDedupInstalled = true;
  };

  installPostprocess();
  installResultDecorator();
})();
