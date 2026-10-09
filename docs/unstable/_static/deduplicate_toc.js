// Clean up the "On this page" sidebar for C++ API pages.
//
// Two problems caused by Breathe's per-overload anchor generation:
// 1. toc-h4 entries: Breathe adds a bare redundant "transform()" child anchor under each
//    overload's section heading. Always remove them.
// 2. Duplicate toc-h3 entries: when all overloads share the same display name
//    (e.g. "ExclusiveSum()"), keep only the first occurrence.
document.addEventListener('DOMContentLoaded', function() {
  var tocNav = document.getElementById('pst-page-toc-nav');
  if (!tocNav)
    return;

  tocNav.querySelectorAll('li.toc-h4').forEach(function(li) {
    var label = li.textContent.trim();
    if (label.endsWith('()')) {
      li.remove();
    }
  });

  var seen = new Set();
  tocNav.querySelectorAll('li.toc-h3').forEach(function(li) {
    var label = li.textContent.trim();
    if (!label.endsWith(')')) {
      return;
    }
    if (seen.has(label)) {
      li.remove();
    } else {
      seen.add(label);
    }
  });
});
