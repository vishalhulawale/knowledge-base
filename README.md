# Interview Prep Knowledge Base

Senior/Lead engineer interview preparation site built with [Zensical](https://zensical.org) (MkDocs-compatible `mkdocs.yml`) and hosted on Cloudflare Pages.

## Local preview
```bash
pip install -r requirements.txt
zensical serve        # http://localhost:8000
```

## Deploy (Cloudflare Pages, Git integration)
Cloudflare dashboard → Workers & Pages → Create → Pages → Connect to Git → `vishalhulawale/knowledge-base`:

| Setting | Value |
|---|---|
| Production branch | `main` |
| Framework preset | None |
| Build command | `pip install -r requirements.txt && zensical build` |
| Build output directory | `site` |
| Environment variable | `PYTHON_VERSION` = `3.12` (also pinned in `.python-version`) |

Every push to `main` deploys; other branches get preview URLs.

## Structure
- `docs/<topic>/index.md`: topic overview and subtopic status
- `docs/<topic>/<nn-subtopic>.md`: subtopic pages (written by the `research-subtopic` skill)
- `planning/`: Phase 1 topics, Phase 2 subtopics, `topics.json` manifest
- `.claude/skills/`: research and learning skills
