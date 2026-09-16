// Load this configuration before MathJax. Notebook HTML uses dollar delimiters,
// while Arithmatex emits \(...\) and \[...\] for ordinary Markdown pages.
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Scan notebook prose as well as Arithmatex wrappers, but leave code alone.
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
  }
};
