import json
from graphify.detect import detect
from pathlib import Path

d = detect(Path('.'))
json.dump(d, open('.graphify_detect.json', 'w'))
print(f'Corpus: {d["total_files"]} files | ~{d["total_words"]} words')
ft = d.get('files', {})
print(f'  code:     {len(ft.get("code",[]))} files')
print(f'  docs:     {len(ft.get("document",[]))} files')
print(f'  papers:   {len(ft.get("paper",[]))} files')
print(f'  images:   {len(ft.get("image",[]))} files')
print(f'  archives: {len(d.get("archived_files",[]))} files')
w = d.get('total_words', 0)
f = d.get('total_files', 0)
if w > 2000000 or f > 200:
    print(f'WARNING: {w} words / {f} files exceeds limit')
else:
    print(f'Proceeding directly - within limits')
