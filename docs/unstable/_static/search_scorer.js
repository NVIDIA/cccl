"use strict";

const _normalizeSearchSymbol = (value) =>
  (value || "")
    .toLowerCase()
    .replace(/&lt;|&gt;|&amp;|&quot;|&#39;|&apos;|&nbsp;|&hellip;/g, " ")
    .replace(/[^a-z0-9:]+/g, "");

const _splitSymbolWords = (value) =>
  (value || "")
    .replace(/&lt;|&gt;|&amp;|&quot;|&#39;|&apos;|&nbsp;|&hellip;/g, " ")
    .replace(/::/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .toLowerCase()
    .match(/[a-z0-9]+/g) || [];

const _getSearchQuery = () => {
  try {
    return new URLSearchParams(window.location.search).get("q") || "";
  } catch {
    return "";
  }
};

var Scorer = {
  // Keep strong object-name bias at the base layer.
  objNameMatch: 80,
  objPartialMatch: 35,
  objPrio: {
    0: 25, // highest-priority API objects
    1: 10,
    2: -10,
  },
  objPrioDefault: 0,
  title: 15,
  partialTitle: 7,
  term: 5,
  partialTerm: 2,

  score: (result) => {
    const [docName, title, anchor, descr, baseScore, filename] = result;
    let score = baseScore;

    const trimmedTitle = (title || "").trim();
    const trimmedAnchor = (anchor || "").trim();
    const trimmedDescription = (descr || "").trim();
    const trimmedFilename = (filename || "").trim();
    const trimmedDocName = (docName || "").trim();

    const lowerDescription = trimmedDescription.toLowerCase();
    const lowerFilename = trimmedFilename.toLowerCase();
    const lowerDocName = trimmedDocName.toLowerCase();
    const query = _getSearchQuery().trim();
    const lowerQuery = query.toLowerCase();
    const normalizedQuery = _normalizeSearchSymbol(query);

    const titleParts = trimmedTitle.split("::").filter(Boolean);
    const symbolDepth = titleParts.length;
    const leaf = titleParts.length
      ? titleParts[titleParts.length - 1]
      : trimmedTitle;
    const parentTitle =
      titleParts.length > 1 ? titleParts.slice(0, -1).join("::") : "";
    const normalizedLeaf = _normalizeSearchSymbol(leaf);
    const normalizedParent = _normalizeSearchSymbol(parentTitle);
    const normalizedLeafOnlyQuery = _normalizeSearchSymbol(
      query.includes("::") ? query.split("::").pop() : query,
    );
    const leafWords = _splitSymbolWords(leaf);
    const queryWords = _splitSymbolWords(query);
    const simpleQuery =
      lowerQuery &&
      !/[.:/_]/.test(lowerQuery) &&
      /^[a-z0-9]+$/.test(lowerQuery);

    const looksLikeNamespaceQualified =
      /^[a-zA-Z_]\w*(::[a-zA-Z_]\w*)+/.test(trimmedTitle);

    const isExactFunctionishTitle =
      /::[A-Za-z_]\w*$/.test(trimmedTitle); // e.g. thrust::transform

    const isClassishTitle =
      /::[A-Z]\w*$/.test(trimmedTitle); // e.g. cub::DeviceRadixSort

    const isParameterLike =
      /(template parameter|function parameter)/i.test(trimmedDescription);
    const isMemberLike =
      /(C\+\+ (member|type|property))/i.test(trimmedDescription);
    const isCallableLike =
      /(C\+\+ function\b|C\+\+ class\b|C\+\+ struct\b)/i.test(trimmedDescription);
    const isTopLevelSymbol = looksLikeNamespaceQualified && symbolDepth <= 2;
    const isNestedSymbol = symbolDepth >= 3;
    const isConstructorLike =
      isNestedSymbol &&
      _normalizeSearchSymbol(titleParts[symbolDepth - 2]) === normalizedLeaf;
    const hasQueryInFilename =
      lowerQuery && lowerFilename.includes(lowerQuery);
    const hasQueryInDocName = lowerQuery && lowerDocName.includes(lowerQuery);
    const isEnumeratorLike =
      /(C\+\+ enumerator\b)/i.test(trimmedDescription) || /^[A-Z0-9_]+$/.test(leaf);
    const isInternalHelperLike =
      /(policy|dispatch|state|status|callback|preference|layout|runningprefixop|emptycallback|op)/i.test(
        trimmedTitle,
      ) ||
      /(TileState|Policy|Dispatch|Callback|Preference|Layout|RunningPrefixOp|Status|EmptyCallback)/.test(
        trimmedTitle,
      );
    const isPythonModuleLike = /(Python module\b)/i.test(trimmedDescription);
    const leafStartsWithQueryWord =
      queryWords.length === 1 && leafWords[0] === queryWords[0];
    const leafEndsWithQueryWord =
      queryWords.length === 1 &&
      leafWords.length > 0 &&
      leafWords[leafWords.length - 1] === queryWords[0];
    const leafQueryRemainderWords = queryWords.length === 1
      ? leafWords.filter((word) => word !== queryWords[0])
      : [];
    const hasCompactQueryWordRemainder =
      leafStartsWithQueryWord &&
      leafQueryRemainderWords.length > 0 &&
      leafQueryRemainderWords.length <= 2;
    const hasHelperSuffix =
      /(Strategy|Policy|State|Status|Callback|Preference|Layout|Type|Op|Match|Functor|Tag|Traits|Descriptor|Counts)$/.test(
        leaf,
      );
    // Strong bias toward actual API symbols.
    if (isExactFunctionishTitle) score += 35;
    if (isClassishTitle) score += 20;

    // Small boost for anchored entries; these are often object targets.
    if (trimmedAnchor) score += 5;

    // Penalize taxonomy/concept pages that match lots of body text.
    if (
      lowerDescription.includes("thrust::") ||
      lowerDescription.includes("cub::") ||
      lowerDescription.includes("cuda::")
    ) {
      score += 8;
    }

    // Query-aware ranking: prefer canonical symbol pages over nested members.
    if (normalizedQuery) {
      if (normalizedLeaf === normalizedLeafOnlyQuery) {
        score += isTopLevelSymbol ? 180 : 35;
      } else if (
        normalizedLeafOnlyQuery &&
        normalizedLeaf.includes(normalizedLeafOnlyQuery)
      ) {
        score += 15;
      }
    }

    if (isNestedSymbol) score -= 25;
    if (isParameterLike) score -= 80;
    if (isMemberLike) score -= 35;
    if (isConstructorLike) score -= 30;
    if (isCallableLike && isTopLevelSymbol) score += 20;

    // Prefer libcudacxx/cuda symbols over thrust equivalents on ties.
    if (
      normalizedLeaf === normalizedLeafOnlyQuery &&
      /^cuda::/.test(trimmedTitle)
    ) {
      score += 12;
    }

    // For plain keyword queries, prefer pages that match in title/path metadata.
    if (simpleQuery) {
      if (hasQueryInFilename || hasQueryInDocName) score += 45;
      if (isInternalHelperLike) score -= 100;
      if (isPythonModuleLike) score -= 30;
      if (hasHelperSuffix) score -= 80;
      if (isTopLevelSymbol && isCallableLike && !hasHelperSuffix) score += 40;
      if (
        leafStartsWithQueryWord &&
        isTopLevelSymbol &&
        !isEnumeratorLike &&
        !isInternalHelperLike
      ) {
        score += 75;
      }
      if (
        leafEndsWithQueryWord &&
        isTopLevelSymbol &&
        !isEnumeratorLike &&
        !isInternalHelperLike
      ) {
        score += 45;
      }

      // For broad prefix-style queries like "block", prefer compact public API
      // names over longer compound variants or helper-like extensions.
      if (
        hasCompactQueryWordRemainder &&
        isTopLevelSymbol &&
        !isEnumeratorLike &&
        !isInternalHelperLike &&
        !hasHelperSuffix
      ) {
        score += 70 - 15 * (leafQueryRemainderWords.length - 1);
      }
      // If a nested member matches the query but its parent symbol also does,
      // prefer the parent page/class over the member overload.
      if (
        isNestedSymbol &&
        normalizedLeaf === normalizedLeafOnlyQuery &&
        normalizedParent.includes(normalizedLeafOnlyQuery)
      ) {
        score -= 70;
      }
      if (
        isTopLevelSymbol &&
        normalizedLeaf.includes(normalizedLeafOnlyQuery) &&
        normalizedLeaf !== normalizedLeafOnlyQuery
      ) {
        score += 70;
      }
    }

    return score;
  },
};
