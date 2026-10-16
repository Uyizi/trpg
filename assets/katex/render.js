document.querySelectorAll('.math').forEach(node => {
  const tex = node.dataset.tex;
  try {
    katex.render(tex, node, {
      displayMode: node.dataset.display === 'true',
      throwOnError: false,
      trust: false,
      strict: 'warn',
      maxExpand: 1000,
      maxSize: 20
    });
  } catch (_) {
    node.textContent = tex;
  }
});
