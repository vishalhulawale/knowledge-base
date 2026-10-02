#!/usr/bin/env bash
# commit_review.sh "<message>" : test-build in scratch, then commit everything
set -e
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
which zensical >/dev/null || pip install -q zensical
rm -rf $HOME/kbbuild && mkdir -p $HOME/kbbuild && cp -r docs mkdocs.yml $HOME/kbbuild/ && (cd $HOME/kbbuild && zensical build 2>&1 | tail -2)
git add -A
git -c user.name="Vishal Hulawale" -c user.email="hulawale.vishal@gmail.com" commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Wh7bDFRx8xbnxH5KbLab6R"
git log --oneline -1; ls .git/*.lock 2>/dev/null | wc -l
