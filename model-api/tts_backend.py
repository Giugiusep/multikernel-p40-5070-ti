#!/usr/bin/env python3
"""Independent CPU speech service; never changes the resident language model."""
import io
import json
import math
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import onnxruntime as ort
from piper import PiperVoice, SynthesisConfig
from piper.config import PiperConfig

MODEL = '/home/shiba/models/piper-lessac/en_US-lessac-medium.onnx'
options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
voice = PiperVoice(config=PiperConfig.from_dict(json.load(open(MODEL + '.json'))),
                   session=ort.InferenceSession(MODEL, sess_options=options,
                                               providers=['CPUExecutionProvider']))
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data, content_type='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != '/v1/audio/voices':
            self.respond(404, b'{}')
            return
        self.respond(200, json.dumps({'model': 'piper-tts', 'voices': [
            {'id': 'lessac', 'language': 'en-US', 'format': 'wav', 'sample_rate': 22050}
        ]}).encode())

    def do_POST(self):
        if self.path != '/v1/audio/speech':
            self.respond(404, b'{}')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 32768:
                raise ValueError('invalid request size')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('JSON body must be an object')
            text = payload.get('input')
            if not isinstance(text, str) or not text.strip() or len(text) > 4096:
                raise ValueError('input must contain 1–4096 characters')
            if payload.get('model', 'piper-tts') != 'piper-tts':
                raise ValueError('speech model must be piper-tts')
            if payload.get('voice', 'lessac') != 'lessac':
                raise ValueError('voice must be lessac')
            if payload.get('response_format', 'wav') != 'wav':
                raise ValueError('only WAV output is supported; specify response_format=wav')
            speed = payload.get('speed', 1.0)
            if isinstance(speed, bool) or not isinstance(speed, (float, int)) or not math.isfinite(speed) or not 0.25 <= speed <= 4:
                raise ValueError('speed must be between 0.25 and 4')
            if payload.get('stream'):
                raise ValueError('streaming speech is not supported')
            output = io.BytesIO()
            with lock, wave.open(output, 'wb') as wav:
                voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=1 / speed))
            self.respond(200, output.getvalue(), 'audio/wav')
        except (ValueError, TypeError) as exc:
            self.respond(400, json.dumps({'error': {'message': str(exc)}}).encode())
        except Exception:
            self.log_error('speech synthesis failed')
            self.respond(500, b'{"error":{"message":"speech synthesis failed"}}')


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 19083), Handler).serve_forever()
