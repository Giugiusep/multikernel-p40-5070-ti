"""Local-only Transformers completion backend for GPT and GPT-2 checkpoints."""
from __future__ import annotations
import argparse
import json
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def render_prompt(payload: dict, chat: bool) -> str:
    if payload.get('tools') or payload.get('functions'):
        raise ValueError('These base models do not support tool calling')
    if payload.get('n', 1) != 1:
        raise ValueError('Only n=1 is supported')
    if not chat:
        prompt = payload.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('prompt must be a nonempty string')
        return prompt
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError('Each message must be an object')
        role, content = message.get('role'), message.get('content')
        if role not in {'system', 'user', 'assistant'} or not isinstance(content, str):
            raise ValueError('Only system/user/assistant text messages are supported')
        parts.append(f'{role.capitalize()}: {content}')
    return '\n'.join(parts) + '\nAssistant:'


def request_options(payload: dict, prompt_tokens: int, context: int) -> dict:
    limit = payload.get('max_tokens', payload.get('max_completion_tokens', 64))
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('max_tokens must be a positive integer')
    if prompt_tokens + limit > context:
        raise ValueError(f'Context limit is {context} tokens: prompt={prompt_tokens}, requested output={limit}')
    temperature = payload.get('temperature', 1.0)
    top_p = payload.get('top_p', 1.0)
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative')
    if not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')
    for name in ['presence_penalty', 'frequency_penalty']:
        if payload.get(name, 0) != 0:
            raise ValueError(f'{name} is not supported by this backend')
    options = {'max_new_tokens': limit, 'do_sample': temperature > 0}
    if temperature > 0:
        options.update(temperature=temperature, top_p=top_p, top_k=0)
    return options


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def reply(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == '/health':
            self.reply(200, {'status': 'ok'})
        elif self.path == '/v1/models':
            self.reply(200, {'object': 'list', 'data': [{'id': ALIAS, 'object': 'model', 'owned_by': 'shiba-local'}]})
        else:
            self.reply(404, {'error': {'message': 'not found'}})

    def do_POST(self) -> None:
        if self.path not in {'/v1/completions', '/v1/chat/completions'}:
            self.reply(404, {'error': {'message': 'not found'}})
            return
        started_stream = False
        cancel = threading.Event()
        worker = None
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4 * 1024 * 1024:
                raise ValueError('Invalid request size')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('JSON body must be an object')
            if payload.get('model', ALIAS) != ALIAS:
                raise ValueError('Incorrect model for loaded backend')
            chat = self.path == '/v1/chat/completions'
            prompt = render_prompt(payload, chat)
            inputs = TOKENIZER(prompt, return_tensors='pt', add_special_tokens=False)
            count = inputs['input_ids'].shape[1]
            if count == 0:
                raise ValueError('Prompt contains no tokens')
            options = request_options(payload, count, CONTEXT)
            stops = payload.get('stop', [])
            if isinstance(stops, str):
                stops = [stops]
            if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
                raise ValueError('stop must contain nonempty strings')
            seed = payload.get('seed')
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
                raise ValueError('seed must be an integer')
            with LOCK:
                if seed is not None:
                    torch.manual_seed(seed)
                inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
                state = {}

                class Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        if cancel.is_set():
                            return True
                        if stops:
                            text = TOKENIZER.decode(input_ids[0, count:], skip_special_tokens=True)
                            return any(s in text for s in stops)
                        return False

                streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True, timeout=1)
                def generate():
                    try:
                        with torch.inference_mode():
                            state['output'] = MODEL.generate(**inputs, **options, pad_token_id=PAD,
                                stopping_criteria=StoppingCriteriaList([Stop()]), streamer=streamer)
                    except Exception as exc:
                        state['error'] = exc
                        streamer.end()
                worker = threading.Thread(target=generate, daemon=True)
                begin = time.monotonic()
                worker.start()
                ident = ('chatcmpl-' if chat else 'cmpl-') + uuid.uuid4().hex
                created = int(time.time())
                common = {'id': ident, 'created': created, 'model': ALIAS}
                stream = bool(payload.get('stream'))
                def emit(text='', reason=None, usage=None):
                    if chat:
                        choice = {'index': 0, 'delta': {'content': text}, 'finish_reason': reason}
                    else:
                        choice = {'index': 0, 'text': text, 'finish_reason': reason}
                    item = {**common, 'object': 'chat.completion.chunk' if chat else 'text_completion', 'choices': [choice]}
                    if usage is not None:
                        item['usage'] = usage
                    self.wfile.write(('data: ' + json.dumps(item) + '\n\n').encode())
                    self.wfile.flush()
                if stream:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    started_stream = True
                text, pending = '', ''
                keep = max((len(s) for s in stops), default=0)
                stopped = False
                iterator = iter(streamer)
                while True:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        break
                    except queue.Empty:
                        if not worker.is_alive():
                            break
                        continue
                    text += chunk
                    if stream and not stopped:
                        pending += chunk
                        positions = [pending.find(s) for s in stops if s in pending]
                        if positions:
                            emit(pending[:min(positions)])
                            pending = ''
                            stopped = True
                        elif len(pending) > keep:
                            ready = len(pending) - keep
                            emit(pending[:ready])
                            pending = pending[ready:]
                worker.join()
                if 'error' in state:
                    raise RuntimeError(str(state['error']))
                generated = state['output'].shape[1] - count
                positions = [text.find(s) for s in stops if s in text]
                if positions:
                    text = text[:min(positions)]
                reason = 'stop' if positions or generated < options['max_new_tokens'] else 'length'
                usage = {'prompt_tokens': count, 'completion_tokens': generated, 'total_tokens': count + generated}
                elapsed = time.monotonic() - begin
                if stream:
                    if not stopped and pending:
                        emit(pending)
                    emit(reason=reason, usage=usage if payload.get('stream_options', {}).get('include_usage') else None)
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                else:
                    choice = {'index': 0, 'finish_reason': reason}
                    choice.update({'message': {'role': 'assistant', 'content': text}} if chat else {'text': text})
                    self.reply(200, {**common, 'object': 'chat.completion' if chat else 'text_completion',
                        'choices': [choice], 'usage': usage,
                        'timings': {'generation_wall_seconds': elapsed, 'generated_tokens_per_second': generated / elapsed}})
        except (BrokenPipeError, ConnectionResetError):
            cancel.set()
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            if started_stream:
                self.wfile.write(('data: ' + json.dumps({'error': {'message': str(exc)}}) + '\n\ndata: [DONE]\n\n').encode())
            else:
                self.reply(400 if isinstance(exc, (ValueError, TypeError, KeyError)) else 503,
                           {'error': {'message': str(exc)}})
        finally:
            cancel.set()
            if worker is not None:
                worker.join(timeout=30)


def main():
    global torch, queue, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    global MODEL, TOKENIZER, PAD, ALIAS, CONTEXT, LOCK
    import torch
    import queue
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--alias', required=True)
    parser.add_argument('--context', required=True, type=int)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=19081)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this RTX-only backend')
    ALIAS, LOCK = args.alias, threading.Lock()
    TOKENIZER = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    MODEL = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                              dtype=torch.float16, attn_implementation='eager').to('cuda:0').eval()
    CONTEXT = min(args.context, MODEL.config.n_positions)
    PAD = TOKENIZER.pad_token_id
    if PAD is None:
        PAD = TOKENIZER.eos_token_id if TOKENIZER.eos_token_id is not None else 0
    print(f'Ready: {ALIAS}, CUDA0, context={CONTEXT}, dtype=FP16', flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
