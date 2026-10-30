import json, os, re, subprocess, sys
import numpy as np

index, transcripts, downloads, score_path, out = sys.argv[1:6]
beds_dir = sys.argv[6] if len(sys.argv) > 6 else os.path.dirname(score_path)
RATE, OFFSET = 8000, 1190
FORE, FOCUS, MIDDLE, ROOM = 0.10, 0.05, 0.022, 0.03
POS = {'02': 0.0, '03': 0.15, '04': -0.4, '05': -0.45, '07': -0.6, '08': -0.6, '09': 0.5, '10': 0.55, '13': 0.45, '14': 0.6,
       '17': 0.4, '18': 0.5, '21': 0.65, '11': -0.2, '12': -0.25, '15': 0.3, '16': 0.35, '19': 0.1, '24': -0.7}
FAA_PAN = {'7597523': -0.8, '7597535': -0.6, '7597536': -0.6, '7598879': -0.5, '7597540': -0.35, '7598114': -0.3, '7598115': -0.3,
           '7598742': 0.5, '7598795': 0.5, '7595567': 0.6, '7595534': 0.8, '7595546': 0.7, '7595548': 0.7, '7595549': 0.7,
           '7596660': 0.75, '7596661': 0.75, '7596662': 0.75, '7596667': 0.75, '7596673': 0.75}
CERT = re.compile(r're-?record|coordinated universal|universal coordinated|certif|position being|my name is|employed as|'
                  r'facility involved|subject concerns|following sequence', re.I)
score = json.load(open(score_path, encoding='utf-8'))
sec = lambda v: sum(int(p) * k for p, k in zip(v.split(':'), (3600, 60, 1)))
LO, HI = sec(score['from']), sec(score['to'])
N = (HI - LO) * RATE
os.makedirs(out, exist_ok=True)

rows = [l.rstrip('\n').split('\t') for l in open(index, encoding='utf-8')]
rows = [r for r in rows if LO - 120 <= sec(r[0]) < HI]


def paths(ref):
    where, at = ref.rsplit('@', 1)
    if where.startswith('faa/'):
        naid = where[4:]
        return (os.path.join(transcripts, 'faa', '7419198', naid + '.jsonl'),
                os.path.join(downloads, 'faa', '7419198', naid, naid + '.mp3'), float(at))
    release, stem = where.split('/')
    return (os.path.join(transcripts, 'neads', release, stem + '.jsonl'),
            os.path.join(downloads, 'neads', release, stem, stem + '.wav'), float(at))


def true0(ref, row_time, at):
    m = re.search(r'_(\d{6})@', ref)
    if m:
        s = m[1]
        return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:]) + OFFSET
    return sec(row_time) - at


words_cache, audio_cache, line_cache = {}, {}, {}


def segments(path):
    if path not in words_cache:
        words_cache[path] = [s for s in map(json.loads, open(path, encoding='utf-8')) if s['words']]
    return words_cache[path]


def floor_db(x):
    hop = RATE // 20
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1)) + 1
    return int(np.clip(round(20 * np.log10(np.percentile(rms, 15) / 32768)) + 3, -80, -20))


def audio(path):
    if path in audio_cache:
        return audio_cache[path]
    channels = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels',
                                   '-of', 'csv=p=0', path], capture_output=True, text=True, check=True).stdout.strip() or 1)
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-ac', str(min(channels, 2)), '-ar', str(RATE),
                          '-f', 's16le', '-'], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32).reshape(-1, min(channels, 2))
    if x.shape[1] == 2:
        frames = x[: len(x) // 2048 * 2048].reshape(-1, 2048, 2)
        power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(0)[8:512]
        x = x[:, int(np.argmin(power.max(0) / (np.median(power, 0) + 1e-12)))]
    else:
        x = x[:, 0]
    if path.endswith('.mp3'):
        clean = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 's16le', '-ar', str(RATE), '-ac', '1', '-i', '-', '-af',
                                f'highpass=f=150,afftdn=nr=30:nf={floor_db(x)}', '-f', 's16le', '-'],
                               input=np.clip(x, -32768, 32767).astype(np.int16).tobytes(), capture_output=True, check=True).stdout
        x = np.frombuffer(clean, np.int16).astype(np.float32)
    audio_cache[path] = (x / 32768).astype(np.float32)
    while sum(v.nbytes for v in audio_cache.values()) > 700_000_000 and len(audio_cache) > 1:
        audio_cache.pop(next(iter(audio_cache)))
    return audio_cache[path]


def clean(s):
    w = [list(v) for v in s['words']]
    if len(w) > 1 and w[1][0] - w[0][1] > 3:
        length = min(w[0][1] - w[0][0], 0.6)
        w[0][0], w[0][1] = w[1][0] - length - 0.05, w[1][0] - 0.05
    if len(w) > 1 and w[-1][0] - w[-2][1] > 3:
        length = min(w[-1][1] - w[-1][0], 0.6)
        w[-1][0], w[-1][1] = w[-2][1] + 0.05, w[-2][1] + 0.05 + length
    return dict(s, words=w, start=w[0][0], end=w[-1][1])


def level(x, target):
    hop = RATE // 20
    if len(x) < hop:
        return x
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    loud = rms[rms > max(rms.max() / 8, 1e-5)]
    return x * (target / loud.mean()) if len(loud) else x


def find(t, ch):
    for r in rows:
        if r[0] == t and ch in r[2]:
            return r
    raise SystemExit(f'{t} {ch}: not in index')


def fade(x, a=0.3, b=0.3):
    x = x.copy()
    ka, kb = min(len(x) // 2, int(a * RATE)), min(len(x) // 2, int(b * RATE))
    if ka:
        x[:ka] *= np.linspace(0, 1, ka)
    if kb:
        x[len(x) - kb:] *= np.linspace(1, 0, kb)
    return x


def put(layer, x, t, pan=0.0):
    i = int((t - LO) * RATE)
    if i < 0:
        x, i = x[-i:], 0
    x = x[: max(0, N - i)]
    layer[i:i + len(x), 0] += x * np.sqrt((1 - pan) / 2)
    layer[i:i + len(x), 1] += x * np.sqrt((1 + pan) / 2)


def pan_of(ch):
    if ch.startswith('FAA'):
        return FAA_PAN.get(ch[4:], 0.0)
    if ch[3] == '1':
        return POS.get(ch[4:], 0.0)
    return ((int(ch[3:]) * 37) % 100) / 100 * 1.2 - 0.6


META = re.compile(r're-?record|re-?report|recording (?:is being|concerns|concerning|continues|will continue)|subject concerns|'
                  r'specific concerns|facility involved|positions? of operation|position being|hereby certify|true (?:re-?)?recording|'
                  r'my name is|(?:employed|applied) as|this portion|during the period|time frame|coordinat\w* universal|'
                  r'universal coordinat\w*|archive project|operations service|program analyst|quality assurance|end of (?:this|the re)|'
                  r'continuation of the', re.I)


def meta_spans(wpath):
    if '/faa/' not in wpath.replace('\\', '/'):
        return []
    marks = []
    for s in segments(wpath):
        s = clean(s)
        groups = [[s['words'][0]]]
        for w in s['words'][1:]:
            if w[0] - groups[-1][-1][1] > 1.5:
                groups.append([w])
            else:
                groups[-1].append(w)
        for g in groups:
            text = ' '.join(w[2].strip() for w in g)
            if not META.search(text):
                continue
            end = g[-1][1]
            for k, w in enumerate(g):
                if '2001' in w[2] and len(g) - k > 4:
                    end = w[1]
                    break
            marks.append([g[0][0], end, text])
    marks.sort()
    spans = []
    for a, b, text in marks:
        if spans and a - spans[-1][1] <= 15:
            spans[-1][1] = max(spans[-1][1], b)
            spans[-1][2] += ' ' + text
        else:
            spans.append([a, b, text])
    out = []
    for k, (a, b, text) in enumerate(spans):
        a = 0.0 if a < 90 else a - 1.0
        if re.search(r'end of (?:this|the re)|will continue', text, re.I):
            b = spans[k + 1][0] - 1.0 if k + 1 < len(spans) else float('inf')
        else:
            b += 1.0
        out.append((a, b))
    return out


def lines_of(wpath, apath):
    if apath in line_cache:
        return line_cache[apath]
    x = audio(apath)
    spans = meta_spans(wpath)
    hop = RATE // 20
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    found = []
    if len(rms):
        on = rms > max(float(np.percentile(rms, 10)) * 3, 1e-4)
        for seg in segments(wpath):
            raw = seg['start']
            seg = clean(seg)
            groups = [[seg['words'][0]]]
            for w in seg['words'][1:]:
                if w[0] - groups[-1][-1][1] > 1.5:
                    groups.append([w])
                else:
                    groups[-1].append(w)
            for g in groups:
                i0, i1 = max(0, int((g[0][0] - 0.7) * RATE) // hop), min(len(on), int((g[-1][1] + 0.7) * RATE) // hop + 1)
                hit = np.flatnonzero(on[i0:i1])
                if len(hit) < 3 or any(a <= g[-1][1] and g[0][0] <= b for a, b in spans):
                    continue
                start, end = (i0 + hit[0]) * hop / RATE - 0.15, (i0 + hit[-1] + 1) * hop / RATE + 0.25
                found.append({'seg': raw, 'start': max(0.0, start), 'end': min(len(x) / RATE, end),
                              'words': [[min(max(w[0], start), end), w[2].strip()] for w in g],
                              'text': ' '.join(w[2].strip() for w in g)})
    found.sort(key=lambda l: l['start'])
    line_cache[apath] = found
    return found


recordings = {}
for r in rows:
    ch = r[2].split('+')[0]
    wpath, apath, at = paths(r[6])
    recordings.setdefault(apath, {'wpath': wpath, 'ch': ch, 'pos': r[3], 't0': []})['t0'].append(true0(r[6], r[0], at))
for v in recordings.values():
    v['t0'] = float(np.median(v['t0']))

fore = np.zeros((N, 2), np.float32)
middle = np.zeros((N, 2), np.float32)
busy = np.zeros((HI - LO) * 4 + 1, np.int8)
duck = np.zeros(N, np.float32)
used, timeline, said = set(), [], []
norm = lambda text: re.sub(r'[^a-z0-9]', '', text.lower())

picked = []
for b in score['beats']:
    row = find(b['at'], b['ch'])
    wpath, apath, at = paths(row[6])
    rec = recordings[apath]
    L = lines_of(wpath, apath)
    anchor = [k for k, l in enumerate(L) if abs(l['seg'] - at) < 0.1]
    if not anchor:
        print(b['at'], b['ch'], 'no speech under the line')
        continue
    first = anchor[0]
    if 'to' in b:
        ks = [k for k in range(first, len(L)) if L[k]['start'] <= sec(b['to']) - rec['t0']]
    else:
        ks = [first]
        while ks[-1] + 1 < len(L) and L[ks[-1] + 1]['start'] - L[ks[-1]]['end'] <= 2.5 and L[ks[-1] + 1]['end'] - L[first]['start'] <= 20:
            ks.append(ks[-1] + 1)
    picked.append({'b': b, 'row': row, 'apath': apath, 'ks': sorted(set(ks) | {first})})

by_rec = {}
for p in picked:
    by_rec.setdefault(p['apath'], []).append(p)
for apath, ps in by_rec.items():
    ps.sort(key=lambda p: p['ks'][0])
    L = line_cache[apath]
    for p, q in zip(ps, ps[1:]):
        t0 = recordings[apath]['t0']
        gap = (t0 + L[p['ks'][-1]]['end'], t0 + L[q['ks'][0]]['start'])
        if any(gap[0] <= sec(s) <= gap[1] for s, _ in score['silence']) or q['b'].get('hush', 0) >= 1 or                 p['b'].get('focus') != q['b'].get('focus'):
            continue
        if gap[1] - gap[0] <= 40:
            p['ks'] = sorted(set(p['ks']) | set(range(p['ks'][-1] + 1, q['ks'][0])))


def first_time(p):
    if 'place' in p['b']:
        return sec(p['b']['place'])
    return recordings[p['apath']]['t0'] + line_cache[p['apath']][p['ks'][0]]['start']


last_end, played = 0, {}
for p in sorted(picked, key=first_time):
    b, row, apath = p['b'], p['row'], p['apath']
    rec, L, x = recordings[apath], line_cache[apath], audio(apath)
    chosen = [L[k] for k in p['ks']]
    pieces = [[chosen[0]]]
    for l in chosen[1:]:
        if l['start'] - pieces[-1][-1]['end'] < 3:
            pieces[-1].append(l)
        else:
            pieces.append([l])
    shift = (sec(b['place']) - (rec['t0'] + chosen[0]['start'])) if 'place' in b else 0
    spoken = np.concatenate([x[int(l['start'] * RATE):int(l['end'] * RATE)] for l in chosen])
    focus = b.get('focus', False)
    gain = (FOCUS if focus else FORE) / (np.sqrt((spoken ** 2).mean()) + 1e-6)
    for piece_lines in pieces:
        a, e = piece_lines[0]['start'] - 0.3, piece_lines[-1]['end'] + 0.3
        seam, rise = None, 0.3
        if apath in played and a < played[apath][0]:
            print('   trimmed', round(played[apath][0] - a, 2), 's already played from', piece_lines[0]['text'][:40])
            if played[apath][1] == last_end and not focus:
                a, seam = played[apath][0] - 0.3, last_end - 0.3
            else:
                a, rise = played[apath][0], 0.05
        if e - a < 0.6:
            continue
        piece = fade(x[int(max(0, a) * RATE):int(e * RATE)] * gain, rise, 0.3)
        if focus:
            when = rec['t0'] + a + shift
            put(middle, piece, when, pan_of(rec['ch']))
            played[apath] = (e, None)
            busy[int((when - LO) * 4):int((when + len(piece) / RATE - LO) * 4) + 1] += 1
            timeline.append({'start': round(when, 2), 'end': round(when + len(piece) / RATE, 2), 'ch': rec['ch'], 'pos': rec['pos'],
                             'layer': 'middle', 'focus': True, 'text': ' '.join(l['text'] for l in piece_lines)})
            continue
        when = seam if seam else max(rec['t0'] + a + shift, last_end - 0.2)
        last_end = when + len(piece) / RATE
        played[apath] = (e, last_end)
        put(fore, piece, when, pan_of(rec['ch']) * 0.3)
        i, j = int((when - LO) * RATE), int((last_end - LO) * RATE)
        duck[max(0, i):max(0, j)] = np.maximum(duck[max(0, i):max(0, j)], b.get('hush', 0.7))
        timeline.append({'start': round(when, 2), 'end': round(last_end, 2), 'ch': rec['ch'], 'pos': rec['pos'], 'layer': 'fore',
                         'words': [[round(when + max(t, a) - a, 2), w] for l in piece_lines for t, w in l['words']]})
    for k in p['ks']:
        used.add((apath, k))
        said.append((rec['t0'] + L[k]['start'], norm(L[k]['text'])))
    print(b['at'], rec['ch'], len(pieces), 'pieces |', ' / '.join(l['text'][:40] for l in chosen[:4]))

room = np.zeros(N, np.float32)
FRAME = RATE // 20
heard = []
for apath, rec in sorted(recordings.items()):
    L = lines_of(rec['wpath'], apath)
    x = audio(apath)
    for k, l in enumerate(L):
        t = rec['t0'] + l['start']
        if not (LO <= t < HI):
            continue
        piece = x[int(l['start'] * RATE):int(l['end'] * RATE)]
        n = len(piece) // FRAME
        if n < 4:
            continue
        env = np.log(np.sqrt((piece[: n * FRAME].reshape(n, FRAME) ** 2).mean(1)) + 1e-5)
        weight = 3.0 if (apath, k) in used else (0.4 if not rec['ch'].startswith('FAA') else 0) + min(len(l['words']), 20) / 20
        weight += 0.2 if '03012647' in apath else 0
        heard.append({'t': t, 'end': rec['t0'] + l['end'], 'env': env, 'apath': apath, 'k': k, 'weight': weight})


def same(a, b):
    d = int(round((b['t'] - a['t']) * 20))
    best = 0.0
    for shift in range(-20, 21):
        lag = d + shift
        i0, i1 = max(0, lag), min(len(a['env']), len(b['env']) + lag)
        if i1 - i0 < 24:
            continue
        u, v = a['env'][i0:i1], b['env'][i0 - lag:i1 - lag]
        if u.std() < 1e-3 or v.std() < 1e-3:
            continue
        best = max(best, float(np.corrcoef(u, v)[0, 1]))
    return best >= 0.75


kept, echoes = {}, set()
for h in sorted(heard, key=lambda h: -h['weight']):
    near = [o for b in range(int(h['t'] // 10) - 4, int(h['end'] // 10) + 2) for o in kept.get(b, [])]
    if any(o['apath'] != h['apath'] and o['t'] - 3 < h['end'] and h['t'] < o['end'] + 3 and same(o, h) for o in near):
        echoes.add((h['apath'], h['k']))
    else:
        kept.setdefault(int(h['t'] // 10), []).append(h)
print(len(echoes), 'lines dropped as the same speech heard on another recording')

candidates = []
for apath, rec in sorted(recordings.items()):
    L = line_cache[apath]
    x = audio(apath)
    for k, l in enumerate(L):
        t = rec['t0'] + l['start']
        if not (LO <= t < HI) or (apath, k) in used or (apath, k) in echoes:
            continue
        piece = x[int(l['start'] * RATE):int(l['end'] * RATE)]
        if not rec['ch'].startswith('FAA'):
            r = level(piece, ROOM)
            i = int((t - LO) * RATE)
            room[i:i + len(r)] += r[: N - i]
        if len(l['words']) < 3 or CERT.search(l['text']) or any(abs(s - t) < 6 and n == norm(l['text']) for s, n in said):
            continue
        weight = (0.4 if not rec['ch'].startswith('FAA') else 0) + min(len(l['words']), 20) / 20
        candidates.append((weight, t, l['end'] - l['start'], apath, k))

taken = {}
tokens = lambda text: set(re.sub(r'[^a-z0-9 ]', '', text.lower()).split())
placed = []
for apath, k in used:
    l = line_cache[apath][k]
    placed.append((recordings[apath]['t0'] + l['start'], tokens(l['text']), recordings[apath]['ch'].startswith('FAA')))
for weight, t, length, apath, k in sorted(candidates, reverse=True):
    i, j = int((t - LO) * 4), int((t + length - LO) * 4) + 1
    if busy[i:j].max(initial=0) >= 3:
        continue
    words, faa = tokens(line_cache[apath][k]['text']), recordings[apath]['ch'].startswith('FAA')
    if len(words) >= 3 and any(abs(pt - t) <= (60 if faa and pf else 3) and (len(words & pw) / len(words | pw) >= 0.5 or
                               min(len(words), len(pw)) >= 6 and len(words & pw) / min(len(words), len(pw)) >= 0.6)
                               for pt, pw, pf in placed):
        continue
    placed.append((t, words, faa))
    busy[i:j] += 1
    taken.setdefault(apath, []).append((t, length, k))
for apath, items in sorted(taken.items()):
    rec, x = recordings[apath], audio(apath)
    for t, length, k in items:
        l = line_cache[apath][k]
        piece = fade(level(x[int(l['start'] * RATE):int(l['end'] * RATE)], MIDDLE * (0.7 if rec['ch'].startswith('FAA') else 1.0)), 0.1, 0.2)
        put(middle, piece, t, pan_of(rec['ch']))
        timeline.append({'start': round(t, 2), 'end': round(t + length, 2), 'ch': rec['ch'], 'pos': rec['pos'], 'layer': 'middle',
                         'text': l['text']})

tmp = os.path.join(out, 'room.raw')
room.tofile(tmp)
blurred = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'f32le', '-ar', str(RATE), '-ac', '1', '-i', tmp, '-af',
                          'highpass=f=200,lowpass=f=700,lowpass=f=700,aecho=0.8:0.7:60|110|170:0.35|0.25|0.15', '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
os.remove(tmp)
room = np.frombuffer(blurred, np.float32)[:N].copy()
room = np.pad(room, (0, N - len(room)))

clock = np.zeros((N, 2), np.float32)
for start, end, gain in score['clock']:
    for name in sorted(os.listdir(os.path.join(downloads, 'neads', '03012647'))):
        m = re.match(r'C0201_11_09_2001_(\d{6})$', name)
        if not m:
            continue
        t0 = int(m[1][:2]) * 3600 + int(m[1][2:4]) * 60 + int(m[1][4:]) + OFFSET
        x = audio(os.path.join(downloads, 'neads', '03012647', name, name + '.wav'))
        a, e = sec(start) - t0, sec(end) - t0
        if 0 <= a < len(x) / RATE:
            piece = fade(level(x[int(a * RATE):int(e * RATE)], FORE * gain), 1.0, 0.05)
            put(clock, piece, sec(start))

hop = RATE // 100
env = duck[: N // hop * hop].reshape(-1, hop).max(1)
smooth = np.zeros_like(env)
for i in range(len(env)):
    prev = smooth[i - 1] if i else 0
    smooth[i] = prev + (env[i] - prev) * (0.25 if env[i] > prev else 0.012)
gain = 1 - np.interp(np.arange(N), np.arange(len(smooth)) * hop, smooth)
quiet = np.ones(N, np.float32)
for start, length in score['silence']:
    i, j = int((sec(start) - LO) * RATE), int((sec(start) + length - LO) * RATE)
    quiet[i:j] = 0
    k = min(N - j, 4 * RATE)
    quiet[j:j + k] = np.linspace(0, 1, k)
cut = quiet.copy()
for start, length in score['silence']:
    j = int((sec(start) + length - LO) * RATE)
    cut[j:] = 1
under = gain * quiet
stems = {'fore': fore * cut[:, None], 'clock': clock[:, 0] * cut * np.sqrt(2), 'middle': middle * under[:, None],
         'room': room * 0.7 * under}
beds = []
for k, bed in enumerate(score.get('beds', [])):
    loop = np.fromfile(os.path.join(beds_dir, bed['file']), np.float32)
    at = int(bed.get('start', 0) * RATE)
    stems[f'bed{k}'] = np.resize(np.roll(loop, -at), N) * bed['gain'] * (1 - 0.6 * (1 - gain)) * quiet
    beds.append({'name': bed['name'], 'stem': f'bed{k}'})
wide = lambda x: x if x.ndim == 2 else np.repeat(x[:, None] / np.sqrt(2), 2, 1)
mix = sum(wide(stems[k]) for k in ('fore', 'clock', 'middle', 'room')) + (wide(stems['bed0']) if beds else 0)
scale = 0.7 / np.percentile(np.abs(mix[np.abs(mix) > 1e-4]), 99.9)


def write(x, name, rate='64k'):
    tmp = os.path.join(out, name + '.raw')
    x.astype(np.float32).tofile(tmp)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(RATE), '-ac', str(x.shape[1] if x.ndim == 2 else 1),
                    '-i', tmp, '-ar', '16000', '-c:a', 'libmp3lame', '-b:a', rate, os.path.join(out, name + '.mp3')], check=True)
    os.remove(tmp)


write(np.tanh(mix * scale) * 0.95, 'mix')
for name, x in stems.items():
    write(x * scale, name, '64k' if x.ndim == 2 else '40k')
timeline.sort(key=lambda t: t['start'])
json.dump({'lo': LO, 'hi': HI, 'silence': [[sec(s[0]), s[1]] for s in score['silence']], 'beds': beds, 'items': timeline},
          open(os.path.join(out, 'timeline.json'), 'w', encoding='utf-8'), ensure_ascii=False)
print(sum(1 for t in timeline if t['layer'] == 'fore'), 'pieces,', sum(1 for t in timeline if t['layer'] == 'middle'), 'middle lines')
