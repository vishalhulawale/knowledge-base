# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Shared Docs Theme

This site and the NeuroTrade docs (`vishalhulawale/neuro-trade`) must always share the same style and UI behaviour. Details are in `README.md` → Look and feel.

- Any change to `docs/stylesheets/extra.css`, `docs/javascripts/extra.js`, `docs/javascripts/build-info.js`, `scripts/check_docs_theme_sync.py`, the `theme:` block of `mkdocs.yml`, the Zensical pin or the deployment-time stamp step is made in **both** repos in the same session, and committed and pushed in both. If the other repo isn't in the session, add it with `add_repo`.
- The shared files stay byte-identical; class names use the neutral `doc-` prefix. Only the logo icon differs in `theme:`.
- Before committing such a change, run `python scripts/check_docs_theme_sync.py` and make sure it reports `Docs theme in sync`.
- In NeuroTrade, the same change also updates `docs/DEVELOPER.md` → Shared Theme with the Knowledge Base, and its commit follows that repo's style (past tense, title case).
