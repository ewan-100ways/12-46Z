import csv
import gzip
import json
import re
import sys

NEADS = re.compile(r'(C\d{4})_(\d\d)_(\d\d)_(\d{4})_(\d{6})\.wav$', re.I)


def seconds(length):
    return sum(float(p) * 60 ** k for k, p in enumerate(reversed(str(length or 0).split(':'))))


def neads(files):
    yield ['path', 'channel', 'date', 'start', 'seconds', 'bytes']
    for item in files['files']:
        m = NEADS.search(item['name'])
        if item.get('source') == 'original' and m:
            yield [item['name'], m[1], f'{m[4]}-{m[3]}-{m[2]}', m[5], round(seconds(item.get('length')), 1), item.get('size', '')]


def faa(records):
    yield ['path', 'naid', 'unit', 'title', 'bytes', 'use', 'access']
    for r in records:
        unit = next((u['title'] for u in r.get('ancestors', []) if u.get('levelOfDescription') == 'fileUnit'), '')
        for o in r.get('digitalObjects', []):
            yield [o['objectUrl'], r['naId'], unit, r['title'], o.get('objectFileSize', ''),
                   r.get('useRestriction', {}).get('status', ''), r.get('accessRestriction', {}).get('status', '')]


kind, source, out = sys.argv[1], sys.argv[2], sys.argv[3]
with (gzip.open if source.endswith('.gz') else open)(source, 'rt', encoding='utf-8') as f:
    data = json.load(f)
with open(out, 'w', newline='', encoding='utf-8') as f:
    rows = csv.writer(f, lineterminator='\n')
    count = -1
    for row in {'neads': neads, 'faa': faa}[kind](data):
        rows.writerow(row)
        count += 1
print(count, 'recordings')
