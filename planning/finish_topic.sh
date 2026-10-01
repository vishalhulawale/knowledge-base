#!/usr/bin/env bash
# finish_topic.sh <topic-slug> : sync nav/status, test-build in scratch, commit
set -e
cd "$(dirname "$0")/.."
python3 .claude/skills/research-subtopic/mark_done.py "$1"
export PATH=$HOME/.local/bin:$PATH
rm -rf $HOME/kbbuild && mkdir -p $HOME/kbbuild && cp -r docs mkdocs.yml $HOME/kbbuild/ && (cd $HOME/kbbuild && zensical build 2>&1 | tail -3)
git add -A
git -c user.name="Vishal Hulawale" -c user.email="hulawale.vishal@gmail.com" commit -q -m "docs($1): add subtopic pages

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Wh7bDFRx8xbnxH5KbLab6R"
git log --oneline -1; ls .git/*.lock 2>/dev/null | wc -l
