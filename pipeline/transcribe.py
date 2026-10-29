import argparse
import json
import os
import subprocess
import time

import numpy as np
from faster_whisper import WhisperModel

RATE = 16000

args = argparse.ArgumentParser()
args.add_argument('models')
args.add_argument('out')
args.add_argument('recordings', nargs='+')
args.add_argument('--model', default='small.en')
args.add_argument('--device', default='cpu')
args.add_argument('--compute', default='int8')
a = args.parse_args()
model = WhisperModel(a.model, device=a.device, compute_type=a.compute, cpu_threads=4, download_root=a.models)
for path in a.recordings:
    name, t0 = os.path.splitext(os.path.basename(path))[0], time.time()
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(RATE), '-f', 's16le', '-'],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    segments, _ = model.transcribe(audio, vad_filter=True, word_timestamps=True, condition_on_previous_text=False)
    done = os.path.join(a.out, name + '.jsonl')
    with open(done + '.part', 'w', encoding='utf-8', newline='\n') as f:
        for s in segments:
            f.write(json.dumps({'start': s.start, 'end': s.end, 'text': s.text.strip(), 'p': s.avg_logprob, 'nospeech': s.no_speech_prob,
                                'words': [[w.start, w.end, w.word, round(w.probability, 2)] for w in s.words]}) + '\n')
    os.replace(done + '.part', done)
    print(f'{name}: {len(audio) / RATE / 60:.1f} min in {(time.time() - t0) / 60:.1f} min', flush=True)
