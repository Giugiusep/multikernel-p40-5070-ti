#!/usr/bin/env python3
"""CPU Whisper sidecar: multipart transcription and internal JSON audio input."""
import base64
import io
import json
import threading
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from faster_whisper import WhisperModel
import av
import numpy as np

MODEL = '/home/shiba/models/whisper-tiny'
MAX_BODY = 24 * 1024 * 1024
lock = threading.Lock()


def parse_request(body: bytes, content_type: str) -> tuple[dict, bytes]:
    if content_type.split(';')[0] == 'application/json':
        fields = json.loads(body)
        if not isinstance(fields, dict):
            raise ValueError('JSON body must be an object')
        audio = fields.get('audio')
        if not isinstance(audio, str):
            raise ValueError('audio must contain base64 encoded audio')
        return fields, base64.b64decode(audio, validate=True)
    if not content_type.startswith('multipart/form-data'):
        raise ValueError('send multipart/form-data with a file field')
    message = BytesParser(policy=policy.default).parsebytes(
        b'Content-Type: ' + content_type.encode() + b'\r\nMIME-Version: 1.0\r\n\r\n' + body)
    if not message.is_multipart() or message.defects:
        raise ValueError('invalid multipart body')
    fields, audio = {}, None
    for part in message.iter_parts():
        name = part.get_param('name', header='content-disposition')
        if name == 'file':
            if audio is not None:
                raise ValueError('send one audio file')
            audio = part.get_payload(decode=True)
        elif name:
            fields[name] = part.get_payload(decode=True).decode('utf-8')
    if not audio:
        raise ValueError('missing audio file')
    return fields, audio


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, value):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/health':
            self.respond(200, {'status':'ok','model':'whisper-tiny','device':'cpu'})
        else:
            self.respond(404, {'error':{'message':'not found'}})

    def do_POST(self):
        if self.path != '/v1/audio/transcriptions':
            self.respond(404, {'error': {'message': 'not found'}})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                raise ValueError('audio request limit is 24 MiB')
            fields, audio = parse_request(self.rfile.read(length), self.headers.get('Content-Type', ''))
            if fields.get('model', 'whisper-tiny') not in ('whisper-tiny', 'whisper-1'):
                raise ValueError('transcription model must be whisper-tiny')
            if fields.get('response_format', 'json') != 'json':
                raise ValueError('only JSON transcription output is supported')
            language = fields.get('language') or None
            if language is not None and not isinstance(language, str):
                raise ValueError('language must be a language code such as en or it')
            try:
                chunks, count = [], 0
                with av.open(io.BytesIO(audio)) as container:
                    resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
                    for frame in container.decode(audio=0):
                        frame.pts = None
                        for converted in resampler.resample(frame):
                            chunk = converted.to_ndarray().reshape(-1)
                            count += len(chunk)
                            if count > 120 * 16000:
                                raise ValueError('audio exceeds 120 seconds')
                            chunks.append(chunk)
                    for converted in resampler.resample(None):
                        chunks.append(converted.to_ndarray().reshape(-1))
                samples = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError('invalid or unsupported audio file') from exc
            if not 0 < len(samples) <= 120 * 16000:
                raise ValueError('audio duration must be 0–120 seconds')
            with lock:
                segments, info = model.transcribe(samples, language=language, beam_size=5,
                    temperature=0, condition_on_previous_text=False, vad_filter=False)
                segments = list(segments)
            self.respond(200, {'text': ''.join(s.text for s in segments).strip(),
                'language': info.language, 'duration': len(samples)/16000,
                'segments': [{'start':s.start,'end':s.end,'text':s.text} for s in segments]})
        except (ValueError, TypeError) as exc:
            self.respond(400, {'error': {'message': str(exc)}})
        except Exception:
            self.log_error('transcription failed')
            self.respond(500, {'error': {'message': 'transcription failed'}})


if __name__ == '__main__':
    model = WhisperModel(MODEL, device='cpu', compute_type='int8', cpu_threads=2,
                         num_workers=1, local_files_only=True)
    ThreadingHTTPServer(('127.0.0.1', 19084), Handler).serve_forever()
