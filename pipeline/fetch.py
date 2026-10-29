import argparse
import concurrent.futures
import csv
import os
import re
import time
import urllib.parse
import urllib.request


def neads(row):
    name = os.path.basename(row['path'])
    release = re.search(r'RDOD_(\d{8})', row['path'])[1]
    url = 'https://archive.org/download/rdodneadsaudio/' + urllib.parse.quote(row['path'])
    return url, os.path.join(release, os.path.splitext(name)[0], name)


def faa(row):
    return row['path'], os.path.join('7419198', row['naid'], row['naid'] + '.mp3')


PLACES = {'neads': neads, 'faa': faa}

args = argparse.ArgumentParser()
args.add_argument('source', choices=PLACES)
args.add_argument('downloads')
args.add_argument('--match', default='')
args.add_argument('--jobs', type=int, default=6)
a = args.parse_args()
with open(f'catalogue/{a.source}/files.csv', encoding='utf-8') as f:
    rows = [r for r in csv.DictReader(f) if re.search(a.match, ' '.join(r.values()))]


SLACK = {'neads': 0, 'faa': 1024}


def fetch(row):
    url, place = PLACES[a.source](row)
    path = os.path.join(a.downloads, place)
    size = int(row['bytes'])
    whole = lambda p: abs(os.path.getsize(p) - size) <= SLACK[a.source]
    if os.path.exists(path) and whole(path):
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = f'{path}.{os.getpid()}.part'
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, part)
            if whole(part):
                os.replace(part, path)
                return size
        except OSError:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f'{url}: failed')


done = got = failed = 0
with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
    for future in concurrent.futures.as_completed([pool.submit(fetch, r) for r in rows]):
        done += 1
        try:
            got += future.result()
        except RuntimeError as e:
            failed += 1
            print(e, flush=True)
        if done % 500 == 0 or done == len(rows):
            print(f'{done}/{len(rows)} files, {got / 1e6:.0f} MB fetched, {failed} failed', flush=True)
