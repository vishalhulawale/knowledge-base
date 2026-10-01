#!/usr/bin/env bash
# Render every ```mermaid block in the given markdown files; prints FAIL lines for parse errors.
# Needs: npm i -g @mermaid-js/mermaid-cli (and a Chromium; set PUPPETEER_EXECUTABLE_PATH if needed)
set -u
tmp=$(mktemp -d); fails=0
[ -n "${PUPPETEER_EXECUTABLE_PATH:-}" ] && echo "{\"executablePath\":\"$PUPPETEER_EXECUTABLE_PATH\",\"args\":[\"--no-sandbox\"]}" > "$tmp/p.json" || echo '{"args":["--no-sandbox"]}' > "$tmp/p.json"
python3 - "$tmp" "$@" <<'PY'
import re,sys,os
out=sys.argv[1]
for f in sys.argv[2:]:
    for i,m in enumerate(re.findall(r'```mermaid\n(.*?)```',open(f,encoding='utf-8').read(),re.S)):
        open(os.path.join(out,f"{os.path.basename(f)}__{i}.mmd"),'w',encoding='utf-8').write(m)
PY
for m in "$tmp"/*.mmd; do
  [ -e "$m" ] || continue
  mmdc -p "$tmp/p.json" -i "$m" -o "${m%.mmd}.svg" >/dev/null 2>&1 || { echo "FAIL $(basename "$m")"; fails=$((fails+1)); }
done
echo "mermaid: $fails failure(s)"; exit $fails
