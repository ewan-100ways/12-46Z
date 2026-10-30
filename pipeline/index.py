import csv, json, os, re, sys
neads, faa, faacat, out = sys.argv[1:5]
OFFSET = 1190
POS = {'02': 'MCC', '03': 'MCC/T', '04': 'ID', '05': 'ID', '07': 'ID2', '08': 'ID2', '09': 'TT', '10': 'TT', '13': 'TT', '14': 'TT',
       '17': 'TT', '18': 'TT', '21': 'TT', '11': 'ACWO', '12': 'ACWO', '15': 'AST', '16': 'AST', '19': 'SD', '24': 'AICC'}
rows = []
for release in os.listdir(neads):
    for name in os.listdir(os.path.join(neads, release)):
        m = re.match(r'C0(\d)(\d\d)_11_09_2001_(\d{6})\.jsonl$', name)
        if not m or m[2] == '01':
            continue
        st = m[3]; t0 = int(st[:2]) * 3600 + int(st[2:4]) * 60 + int(st[4:]) + OFFSET
        ch = f'C0{m[1]}{m[2]}'
        for line in open(os.path.join(neads, release, name), encoding='utf-8'):
            s = json.loads(line)
            if not s['words']:
                continue
            p = sum(w[3] for w in s['words']) / len(s['words'])
            rows.append([t0 + s['start'], t0 + s['end'], ch, POS.get(m[2], '') if m[1] == '1' else '', p, s['text'], f'{release}/{name[:-6]}@{s["start"]:.1f}'])
titles = {r['naid']: r['title'] for r in csv.DictReader(open(faacat, encoding='utf-8'))}
for name in os.listdir(faa):
    naid = name[:-6]
    title = titles.get(naid, '')
    unit = re.sub(r'^.* - (\d+ )?', '', title)[:40]
    m = re.search(r'(\d{4}) ?-+ ?(\d{4}) UTC', title)
    base = (int(m[1][:2]) * 3600 + int(m[1][2:]) * 60) if m else 0
    shift = None
    for line in open(os.path.join(faa, name), encoding='utf-8'):
        s = json.loads(line)
        k = re.search(r'period (\d{4})', s['text'])
        if k:
            base = int(k[1][:2]) * 3600 + int(k[1][2:]) * 60; shift = s['end']
        if not s['words'] or re.search(r're-?recording|coordinated universal|universal coordinated|certif', s['text'], re.I):
            continue
        t = base + s['start'] - (shift or 0)
        p = sum(w[3] for w in s['words']) / len(s['words'])
        rows.append([t, t + s['end'] - s['start'], 'FAA ' + naid, unit, p, s['text'], f'faa/{naid}@{s["start"]:.1f}'])
rows.sort()
keep, last = [], {}
for r in rows:
    key = re.sub(r'[^a-z0-9]', '', r[5].lower())
    prev = last.get(key)
    if prev and abs(prev[0] - r[0]) < 4 and len(key) > 8:
        if r[2] not in prev[2]:
            prev[2] += '+' + r[2]
        continue
    last[key] = r
    keep.append(r)
hms = lambda t: f'{int(t) // 3600:02d}:{int(t) % 3600 // 60:02d}:{int(t) % 60:02d}'
with open(out, 'w', encoding='utf-8', newline='\n') as f:
    for r in keep:
        f.write('\t'.join([hms(r[0]), f'{r[1] - r[0]:.0f}', r[2], r[3], f'{r[4]:.2f}', r[5], r[6]]) + '\n')
print(len(rows), 'segments,', len(keep), 'after merging copies')
