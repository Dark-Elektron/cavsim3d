// Two kinds of page carry maths, and they arrive in different shapes:
//
//   *.md pages     pymdownx.arithmatex (generic: true) rewrites $...$ and
//                  $$...$$ into \(...\) / \[...\] wrapped in .arithmatex
//                  elements before MathJax ever sees them.
//
//   *.ipynb pages  mkdocs-jupyter renders notebooks straight to HTML and never
//                  runs them through arithmatex, so markdown cells still hold
//                  literal $...$ / $$...$$ in plain <p> elements nested a few
//                  levels below .jp-MarkdownCell.
//
// Hence both delimiter styles below. The ignore/process classes matter just as
// much: the usual Material pairing is ignoreHtmlClass ".*|", which matches every
// class AND the empty string, so every classless <p> inside a notebook cell gets
// re-ignored and its maths is never reached. That works for arithmatex only
// because its maths sits directly inside the .arithmatex element. So ignore just
// the notebook code/output areas instead -- <pre> and <code> are skipped by
// MathJax's own skipHtmlTags default anyway.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: "tex2jax_ignore|jp-CodeCell|jp-OutputArea",
    processHtmlClass: "arithmatex|tex2jax_process"
  }
};

function typesetMath() {
  if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
    MathJax.typesetPromise();
  }
}

// Material exposes document$, which also fires on instant-navigation page
// swaps; fall back to DOMContentLoaded when it is unavailable.
if (typeof document$ !== "undefined") {
  document$.subscribe(typesetMath);
} else {
  document.addEventListener("DOMContentLoaded", typesetMath);
}
