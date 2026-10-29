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
args.add_argument('--model', default='large-v3')
args.add_argument('--device', default='cuda')
args.add_argument('--compute', default='float16')
a = args.parse_args()

todo = []
for folder, _, names in os.walk(a.downloads):
    for name in names:
        if name.lower().endswith(AUDIO) and re.search(a.match, name):
            path = os.path.join(folder, name)
            release = os.path.relpath(os.path.dirname(folder), a.downloads)
            done = os.path.join(a.out, release, os.path.splitext(name)[0] + '.jsonl')
            if not os.path.exists(done):
                todo.append((path, done))
todo.sort(key=lambda t: os.path.basename(t[0]))
print(len(todo), 'to transcribe', flush=True)

model = WhisperModel(a.model, device=a.device, compute_type=a.compute, cpu_threads=4, download_root=a.models)
heard = spent = 0.0
for n, (path, done) in enumerate(todo, 1):
    t0 = time.time()
    try:
        raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(RATE), '-f', 's16le', '-'],
                             capture_output=True, check=True).stdout
        audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
        segments, _ = model.transcribe(audio, vad_filter=True, word_timestamps=True, condition_on_previous_text=False)
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
