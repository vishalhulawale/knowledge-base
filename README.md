# Interview Prep Knowledge Base

Senior/Lead engineer interview preparation site built with [Zensical](https://zensical.org) (MkDocs-compatible `mkdocs.yml`) and hosted on Cloudflare Workers.

## Local preview
```bash
pip install -r requirements.txt
zensical serve        # http://localhost:8000
```

## Deploy (GitHub Actions → Cloudflare Worker with static assets)
`.github/workflows/deploy.yml` builds the site with Zensical and runs `wrangler deploy`, which uploads `site/` to the Worker defined in `wrangler.jsonc` (`knowledge-base`).

| Trigger | Result |
|---|---|
| Push to `main` / manual run | Deploy → https://knowledge-base.hulawale-vishal.workers.dev |
| Pull request | Build only (catches broken builds before merge) |

One-time setup:
1. **API token**: My Profile → API Tokens → Create Token → template **Edit Cloudflare Workers** (Account › Workers Scripts › Edit), scoped to your account.
2. **Account ID**: Workers & Pages overview (right sidebar).
3. **GitHub secrets**: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`.
4. If the Worker is also connected to Git in Cloudflare (Settings → Build → Git repository / Workers Builds), **disconnect it**, so only GitHub Actions deploys.

The site is protected by Cloudflare Access (Zero Trust), so only allowed identities can view it.

## Structure
- `docs/<topic>/index.md`: topic overview and subtopic status
- `docs/<topic>/<nn-subtopic>.md`: subtopic pages (written by the `research-subtopic` skill)
- `planning/`: Phase 1 topics, Phase 2 subtopics, `topics.json` manifest
- `.claude/skills/`: research and learning skills

## Look and feel
The site shares one theme with the NeuroTrade docs ([vishalhulawale/neuro-trade](https://github.com/vishalhulawale/neuro-trade)): Zensical's `modern` variant with Figtree (body) / JetBrains Mono (code), Lucide icons and a light / dark / system switch. The two sites must stay identical in style and UI behaviour, so every theme change is made in both repos (see `CLAUDE.md`).

| Shared with NeuroTrade | Rule |
|---|---|
| `docs/stylesheets/extra.css`, `docs/javascripts/extra.js`, `docs/javascripts/build-info.js`, `scripts/check_docs_theme_sync.py` | Byte-identical. Class names use the neutral `doc-` prefix (`doc-hero`, `doc-tagline`, `doc-progress`, `doc-reading`) |
| `theme:` block of `mkdocs.yml` | Identical except `icon.logo` (`lucide/graduation-cap` here) |
| Zensical pin | Same range (`requirements.txt` here, `requirements-docs.txt` there) |
| Deployment-time stamp | Same `DOCS_BUILD_TIME` command (`deploy.yml` here, `docs.yml` there) |

What the shared files do:
- `extra.css`: indigo/teal palette, home hero, cards, tables, admonitions, `P0`–`P3` badges and reading mode.
- `extra.js`: the reading-mode button in the header (hides tabs, sidebars and breadcrumbs; remembered per device under `docs.readingMode`), badges for inline `` `P0` ``–`` `P3` ``, an optional `doc-progress` checklist bar, and the "Last updated on" footer.
- `build-info.js`: holds `null`; the deploy workflow overwrites it with the deployment time. Don't commit a stamped copy.

Check the two repos are in sync (with neuro-trade checked out next to this repo):
```bash
python scripts/check_docs_theme_sync.py               # or pass the path to neuro-trade
```
It prints `Docs theme in sync` or a diff per drifted item (exit 1). Zensical is pre-1.0 and pinned to a minor range; upgrade both sites together and check them in light and dark mode.
