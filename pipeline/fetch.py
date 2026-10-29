import csv
import os
import sys
import urllib.parse
import urllib.request

ITEM = 'https://archive.org/download/rdodneadsaudio/'
CATALOGUE = 'catalogue/neads/files.csv'

downloads, stems = sys.argv[1], sys.argv[2:]
with open(CATALOGUE, encoding='utf-8') as f:
    rows = {os.path.basename(r['path'])[:-4]: r for r in csv.DictReader(f)}
for stem in stems:
    row = rows.get(stem)
    if not row:
        raise SystemExit(f'{stem}: not in the catalogue')
    folder = os.path.join(downloads, stem)
    path = os.path.join(folder, stem + '.mp3')
    if os.path.exists(path) and os.path.getsize(path) == int(row['bytes']):
        print(stem, 'already here')
        continue
    os.makedirs(folder, exist_ok=True)
    urllib.request.urlretrieve(ITEM + urllib.parse.quote(row['path']), path + '.part')
    if os.path.getsize(path + '.part') != int(row['bytes']):
        raise SystemExit(f'{stem}: {os.path.getsize(path + ".part")} bytes, the catalogue says {row["bytes"]}')
    os.replace(path + '.part', path)
    print(stem, row['bytes'], 'bytes')
