import csv, json, os, re, shutil, sys

timeline, mix, faacat, out = sys.argv[1:5]
FACILITY = {'ZBW': 'Boston Center', 'ZNY': 'New York Center', 'ZID': 'Indianapolis Center', 'ZDC': 'Washington Center',
            'ZOB': 'Cleveland Center', 'IAD': 'Dulles', 'DCC': 'Command Center', 'K90': 'Cape Approach', 'N90': 'New York Approach',
            'A90': 'Boston Approach'}
titles = {r['naid']: r['title'] for r in csv.DictReader(open(faacat, encoding='utf-8'))}


def who(item):
    if not item['ch'].startswith('FAA'):
        return ' '.join(['NEADS', item['pos']]).strip()
    codes = re.findall(r'\b([A-Z]{3}|[A-Z]\d{2})\b', titles.get(item['ch'][4:], ''))
    return next((FACILITY[c] for c in codes if c in FACILITY), 'FAA')


d = json.load(open(timeline, encoding='utf-8'))
moments = []
for item in sorted((i for i in d['items'] if i['layer'] == 'fore'), key=lambda i: i['start']):
    name = who(item)
    if moments and moments[-1][2] == name and item['start'] - moments[-1][1] < 2:
        moments[-1][1] = item['end']
    else:
        moments.append([item['start'], item['end'], name])
os.makedirs(out, exist_ok=True)
shutil.copyfile(mix, os.path.join(out, 'mix.mp3'))
with open(os.path.join(out, 'moments.json'), 'w', encoding='utf-8', newline='\n') as f:
    json.dump({'lo': d['lo'], 'hi': d['hi'], 'silence': d['silence'], 'moments': moments}, f, separators=(',', ':'))
print(len(moments), 'moments')
