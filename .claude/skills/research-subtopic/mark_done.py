"""Mark subtopic pages as done and sync nav + topic index.
Usage: python3 .claude/skills/research-subtopic/mark_done.py <topic-slug> [<subtopic-slug> ...]
With no subtopic slugs, it just re-syncs the topic from files that exist on disk."""
import json, os, re, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(root)
topic = sys.argv[1]
d = json.load(open('planning/topics.json', encoding='utf-8'))
t = next(x for x in d['topics'] if x['slug'] == topic)
for s in t['subtopics']:
    if os.path.exists(f"docs/{topic}/{s['slug']}.md"):
        s['status'] = 'done'
json.dump(d, open('planning/topics.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

def page_title(path):
    m = re.search(r'^title:\s*"?(.+?)"?\s*$', open(path, encoding='utf-8').read(), re.M)
    return m.group(1) if m else os.path.basename(path)

# topic index table
idx = f'docs/{topic}/index.md'
lines = open(idx, encoding='utf-8').read().split('\n')
for i, s in enumerate(t['subtopics'], 1):
    star = '★' if s['resume'] else ''
    name = f"[{s['title']}]({s['slug']}.md)" if s['status'] == 'done' else s['title']
    status = ':material-check-circle: Done' if s['status'] == 'done' else ':material-progress-clock: To do'
    row = f"| {i} | {name} | {star} | {status} |"
    for j, l in enumerate(lines):
        if l.startswith(f"| {i} | "):
            lines[j] = row
open(idx, 'w', encoding='utf-8').write('\n'.join(lines))

# nav
m = open('mkdocs.yml', encoding='utf-8').read().split('\n')
start = next(i for i, l in enumerate(m) if l.strip() == f"- Overview: {topic}/index.md")
end = start + 1
while end < len(m) and m[end].startswith('          - '):
    end += 1
new = [m[start]]
for s in t['subtopics']:
    if s['status'] == 'done':
        path = 'docs/%s/%s.md' % (topic, s['slug'])
        new.append('          - %s: %s/%s.md' % (json.dumps(page_title(path)), topic, s['slug']))
m[start:end] = new
open('mkdocs.yml', 'w', encoding='utf-8').write('\n'.join(m))
print(f"{topic}: {sum(s['status']=='done' for s in t['subtopics'])}/{len(t['subtopics'])} done")
