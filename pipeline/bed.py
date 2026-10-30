import csv, os, re, subprocess, sys
import numpy as np

catalogue, downloads, out, day, first, last = sys.argv[1:7]
MINUTES = int(sys.argv[7]) if len(sys.argv) > 7 else 60
LAYERS, RATE, OFFSET, TAIL = 6, 8000, 1190, 30
sec = lambda v: sum(int(p) * k for p, k in zip(v.split(':'), (3600, 60, 1)))
LO, HI = sec(first), sec(last)
LENGTH = (MINUTES * 60 + TAIL) * RATE
os.makedirs(out, exist_ok=True)

files = []
for r in csv.DictReader(open(catalogue, encoding='utf-8')):
    if r['date'] != day or r['channel'][3:] == '01' or 'RDOD_04020857' in r['path']:
        continue
    s = r['start']
    t = int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:]) + OFFSET
    if LO <= t < HI:
        release = re.search(r'RDOD_(\d{8})', r['path'])[1]
        stem = r['path'].rsplit('/', 1)[1][:-4]
        files.append((t, r['channel'], os.path.join(downloads, release, stem, stem + '.wav')))
files.sort()


def decode(path):
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(RATE), '-f', 's16le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def dial_tone(x):
    hop = RATE // 10
    n = len(x) // hop
    if n < 8:
        return x
    spec = np.abs(np.fft.rfft(x[: n * hop].reshape(n, hop) * np.hanning(hop), axis=1)) ** 2
    hz = np.fft.rfftfreq(hop, 1 / RATE)
    band = spec[:, (hz >= 200) & (hz <= 1000)].sum(1) + 1e-12
    held = (spec[:, (hz >= 340) & (hz <= 360)].sum(1) / band > 0.08) & (spec[:, (hz >= 430) & (hz <= 450)].sum(1) / band > 0.08)
    mute = np.ones(len(x), np.float32)
    i = 0
    while i < n:
        if held[i]:
            j = i
            while j < n and held[j]:
                j += 1
            if j - i >= 8:
                mute[max(0, i * hop - RATE // 10):min(len(x), j * hop + RATE // 10)] = 0
            i = j
        else:
            i += 1
    return x * mute


def active(x):
    hop = RATE // 20
    n = len(x) // hop
    if n < 10:
        return []
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1))
    on = rms > max(np.percentile(rms, 10) * 3, 1e-4)
    runs, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j < n and (on[j] or (j + 6 < n and on[j:j + 6].any())):
                j += 1
            if j - i >= 10:
                runs.append(x[max(0, i * hop - RATE * 3 // 10):min(len(x), j * hop + RATE * 3 // 10)])
            i = j
        else:
            i += 1
    return runs


def level(x, target=0.06):
    hop = RATE // 20
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1))
    loud = rms[rms > rms.max() / 8] if len(rms) else rms
    return x * (target / loud.mean()) if len(loud) and loud.mean() > 1e-5 else x


layers = [np.zeros(LENGTH, np.float32) for _ in range(LAYERS)]
filled = [0] * LAYERS
used = 0
for t, ch, path in files:
    if min(filled) >= LENGTH:
        break
    if not os.path.exists(path):
        continue
    for run in active(dial_tone(decode(path))):
        k = int(np.argmin(filled))
        if filled[k] >= LENGTH:
            break
        run = level(run)
        cross = min(len(run) // 4, 3 * RATE if len(run) > 12 * RATE else RATE // 2)
        start = max(0, filled[k] - cross)
        run = run[: LENGTH - start]
        if cross and start:
            run[:cross] *= np.sin(np.linspace(0, np.pi / 2, cross)) ** 2
            layers[k][start:start + cross] *= np.cos(np.linspace(0, np.pi / 2, cross)) ** 2
        layers[k][start:start + len(run)] += run
        filled[k] = start + len(run)
    used += 1
print(used, 'files used, layers filled to', [round(f / RATE / 60, 1) for f in filled], 'min')

mix = sum(l * g for l, g in zip(layers, np.linspace(1.0, 0.6, LAYERS)))
tmp = os.path.join(out, 'bed.raw')
mix.astype(np.float32).tofile(tmp)
shaped = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'f32le', '-ar', str(RATE), '-ac', '1', '-i', tmp, '-af',
                         'highpass=f=250,lowpass=f=800,lowpass=f=800,aecho=0.8:0.6:45|80|130:0.3|0.2|0.12', '-f', 'f32le', '-'],
                        capture_output=True, check=True).stdout
os.remove(tmp)
bed = np.frombuffer(shaped, np.float32)[:LENGTH].copy()
k = TAIL * RATE
loop = bed[: MINUTES * 60 * RATE].copy()
ramp = np.linspace(0, np.pi / 2, k)
loop[:k] = loop[:k] * np.sin(ramp) + bed[MINUTES * 60 * RATE:MINUTES * 60 * RATE + k] * np.cos(ramp)
loop = dial_tone(loop)
hop = RATE // 2
rms = np.sqrt((loop[: len(loop) // hop * hop].reshape(-1, hop) ** 2).mean(1))
loop *= 0.05 / np.median(rms)
loop.astype(np.float32).tofile(os.path.join(out, f'bed-{day}-{MINUTES}.f32'))
subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(RATE), '-ac', '1', '-i', os.path.join(out, f'bed-{day}-{MINUTES}.f32'),
                '-ar', '16000', '-c:a', 'libmp3lame', '-b:a', '48k', os.path.join(out, f'bed-{day}-{MINUTES}.mp3')], check=True)
print('half-second level, median and p10 to p90:', np.round(20 * np.log10(np.percentile(rms * 0.05 / np.median(rms), [10, 50, 90]) + 1e-9), 1))
