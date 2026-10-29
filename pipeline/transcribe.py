import argparse
import json
import os
import re
import subprocess
import time

import numpy as np

try:
    import nvidia
    for part in ('cublas', 'cudnn', 'cuda_nvrtc'):
        for root in nvidia.__path__:
            if os.path.isdir(os.path.join(root, part, 'bin')):
                os.environ['PATH'] = os.path.join(root, part, 'bin') + os.pathsep + os.environ['PATH']
except ImportError:
    pass

from faster_whisper import WhisperModel

RATE = 16000
AUDIO = ('.wav', '.mp3')

args = argparse.ArgumentParser()
args.add_argument('models')
args.add_argument('downloads')
args.add_argument('out')
args.add_argument('--match', default='')
args.add_argument('--list')
args.add_argument('--model', default='large-v3')
args.add_argument('--device', default='cuda')
args.add_argument('--compute', default='float16')
a = args.parse_args()
listed = set(open(a.list, encoding='utf-8').read().split()) if a.list else None

todo = []
for folder, _, names in os.walk(a.downloads):
    for name in names:
        stem = os.path.splitext(name)[0]
        if name.lower().endswith(AUDIO) and re.search(a.match, name) and (listed is None or stem in listed):
            path = os.path.join(folder, name)
            release = os.path.relpath(os.path.dirname(folder), a.downloads)
            done = os.path.join(a.out, release, os.path.splitext(name)[0] + '.jsonl')
            if not os.path.exists(done):
                todo.append((path, done))
todo.sort(key=lambda t: os.path.basename(t[0]))
print(len(todo), 'to transcribe', flush=True)
if not todo:
    raise SystemExit

def speech(path):
    channels = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels',
                                   '-of', 'csv=p=0', path], capture_output=True, text=True, check=True).stdout.strip() or 1)
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-ac', str(min(channels, 2)), '-ar', str(RATE),
                          '-f', 's16le', '-'], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32).reshape(-1, min(channels, 2)) / 32768
    if x.shape[1] == 1:
        return x[:, 0].copy()
    frames = x[: len(x) // 4096 * 4096].reshape(-1, 4096, 2)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(0)[16:1024]
    peak = power.max(0) / (np.median(power, 0) + 1e-12)
    return x[:, int(np.argmin(peak))].copy()


model = WhisperModel(a.model, device=a.device, compute_type=a.compute, cpu_threads=4, download_root=a.models)
heard = spent = 0.0
for n, (path, done) in enumerate(todo, 1):
    t0 = time.time()
    try:
        audio = speech(path)
        segments, _ = model.transcribe(audio, vad_filter=True, vad_parameters={'threshold': 0.25, 'min_silence_duration_ms': 500},
                                      word_timestamps=True, condition_on_previous_text=False)
        os.makedirs(os.path.dirname(done), exist_ok=True)
        with open(done + '.part', 'w', encoding='utf-8', newline='\n') as f:
            for s in segments:
                f.write(json.dumps({'start': s.start, 'end': s.end, 'text': s.text.strip(), 'p': s.avg_logprob,
                                    'nospeech': s.no_speech_prob,
                                    'words': [[w.start, w.end, w.word, round(w.probability, 2)] for w in s.words]}) + '\n')
        os.replace(done + '.part', done)
    except Exception as e:
        print(f'{path}: {type(e).__name__} {e}', flush=True)
        continue
    heard += len(audio) / RATE
    spent += time.time() - t0
    if n % 100 == 0 or n == len(todo):
        print(f'{n}/{len(todo)}: {heard / 3600:.1f} h of audio in {spent / 3600:.2f} h', flush=True)
