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
The site uses the same theme as the NeuroTrade docs: Zensical's `modern` variant with Inter / JetBrains Mono, Lucide icons and a light / dark / system switch (`mkdocs.yml`).
- `docs/stylesheets/extra.css`: indigo/teal palette, home hero, cards, tables, admonitions, `P0`–`P3` badges and reading mode.
- `docs/javascripts/extra.js`: the reading-mode button in the header (hides tabs, sidebars and breadcrumbs; remembered per device), priority badges for inline `` `P0` ``–`` `P3` `` and an optional `kb-progress` checklist bar.
- `docs/assets/last-updated.js` + `build-info.js`: the "Last updated on" footer stamped by the deploy workflow.

Zensical is pre-1.0 and pinned to a minor range in `requirements.txt`; after upgrading it, check the site in light and dark mode.
