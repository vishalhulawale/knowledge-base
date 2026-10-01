# Interview Prep Knowledge Base

Senior/Lead engineer interview preparation site built with [Zensical](https://zensical.org) (MkDocs-compatible `mkdocs.yml`) and hosted on Cloudflare Pages.

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
