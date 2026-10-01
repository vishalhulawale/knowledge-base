# Interview Prep Knowledge Base

Senior/Lead engineer interview preparation site built with [Zensical](https://zensical.org) (MkDocs-compatible `mkdocs.yml`) and hosted on Cloudflare Pages.

## Local preview
```bash
pip install -r requirements.txt
zensical serve        # http://localhost:8000
```

## Deploy (GitHub Actions → Cloudflare Pages, Direct Upload)
`.github/workflows/deploy.yml` builds the site with Zensical and deploys `site/` with `wrangler pages deploy`:

| Trigger | Result |
|---|---|
| Push to `main` | Production deploy → `https://interview-prep-kb.pages.dev` |
| Pull request to `main` | Preview deploy → `https://<branch>.interview-prep-kb.pages.dev` |
| Manual (Actions → Run workflow) | Redeploy current branch |

One-time setup:
1. **Create the Pages project** (Direct Upload type): Cloudflare dashboard → Workers & Pages → Create → Pages → *Upload assets* → name `interview-prep-kb` (upload any placeholder file), **or** `npx wrangler pages project create interview-prep-kb --production-branch main`.
   Do **not** use "Connect to Git" for this project; it would build twice, and a Direct Upload project can't be switched to Git integration later.
2. **API token**: My Profile → API Tokens → Create Token → Custom → permission **Account › Cloudflare Pages › Edit**, scoped to your account.
3. **Account ID**: Workers & Pages overview page (right sidebar) or the dashboard URL.
4. **GitHub secrets**: repo → Settings → Secrets and variables → Actions → add `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`.
5. Push to `main` and watch the Actions tab.

Optional: custom domain (Pages project → Custom domains) and Cloudflare Access to keep the site private.

## Structure
- `docs/<topic>/index.md`: topic overview and subtopic status
- `docs/<topic>/<nn-subtopic>.md`: subtopic pages (written by the `research-subtopic` skill)
- `planning/`: Phase 1 topics, Phase 2 subtopics, `topics.json` manifest
- `.claude/skills/`: research and learning skills
