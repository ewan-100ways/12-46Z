import csv
import gzip
import json
import re
import sys

NAME = re.compile(r'(C\d{4})_(\d\d)_(\d\d)_(\d{4})_(\d{6})\.mp3$')


def seconds(length):
    return sum(float(p) * 60 ** k for k, p in enumerate(reversed(str(length or 0).split(':'))))


source, out = sys.argv[1], sys.argv[2]
with (gzip.open if source.endswith('.gz') else open)(source, 'rt', encoding='utf-8') as f:
    files = json.load(f)['files']
count = total = 0
with open(out, 'w', newline='', encoding='utf-8') as f:
    rows = csv.writer(f, lineterminator='\n')
    rows.writerow(['path', 'channel', 'date', 'start_utc', 'seconds', 'bytes'])
    for item in files:
        if not item['name'].lower().endswith('.mp3'):
            continue
        m = NAME.search(item['name'])
        channel, date, start = (m[1], f'{m[4]}-{m[3]}-{m[2]}', m[5]) if m else ('', '', '')
        s = seconds(item.get('length'))
        rows.writerow([item['name'], channel, date, start, round(s), item.get('size', '')])
        count += 1
        total += s
print(count, 'mp3s,', round(total / 3600), 'hours')
