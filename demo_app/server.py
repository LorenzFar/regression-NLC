"""Run with python -m demo_app.server; sessions live only in this process."""
import argparse
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Lock
from uuid import uuid4

from retrieval_pipeline.agent import DEFAULT_CHECKPOINT, build_agent, load_nlu
from retrieval_pipeline.pipeline import Session

ACTS = ('propose', 'counter', 'accept', 'reject', 'inquire', 'vague')
SCENARIOS = [
    dict(id='bike', title='City road bike', category='Bicycles', description='Aluminium frame, recently serviced.', listing_price='300', target='240', walk_away='280'),
    dict(id='phone', title='iPhone 13 · 128 GB', category='Electronics', description='Unlocked, good battery, light cosmetic wear.', listing_price='450', target='360', walk_away='420'),
    dict(id='table', title='Oak coffee table', category='Furniture', description='Compact solid-wood table, collection this weekend.', listing_price='120', target='95', walk_away='110'),
    dict(id='couch', title='Apartment-sized couch', category='Furniture', description='Two-seater in charcoal fabric, clean and smoke-free.', listing_price='380', target='300', walk_away='350'),
]


class KeywordNLU:
    """Explicit demo stand-in, never a silent replacement for checkpoint failure."""
    provenance = 'demo:keyword-nlu'

    def predict(self, history, message):
        text = message.lower()
        if re.search(r'\b(deal|yes|accept|agreed|fine)\b', text):
            act = 'accept'
        elif re.search(r'\b(no|nope|reject)\b', text):
            act = 'reject'
        elif re.search(r'\d', text):
            act = 'counter' if history else 'propose'
        elif '?' in text:
            act = 'inquire'
        else:
            act = 'vague'
        return {label: .95 if label == act else .01 for label in ACTS}


class DemoApp:
    def __init__(self, nlu, *, bank=None, provenance=None):
        self.nlu, self.bank = nlu, bank
        self.provenance = provenance or getattr(nlu, 'provenance', type(nlu).__name__)
        self.sessions = {}
        self.lock = Lock()

    def create(self, data):
        config = {}
        for key in ('listing_price', 'target', 'walk_away'):
            try:
                value = Decimal(str(data[key]))
            except (KeyError, InvalidOperation):
                raise ValueError(f'{key} must be a nonnegative amount') from None
            if not value.is_finite() or value < 0 or value.as_tuple().exponent < -2:
                raise ValueError(f'{key} must be finite, nonnegative, with at most two decimals')
            config[key] = str(value)
        config.update(gate_mode=data.get('gate_mode', 'full'),
                      threshold=float(data.get('threshold', .7)), fraction=str(data.get('fraction', '.5')))
        pipeline = build_agent(self.nlu, bank=self.bank, **config)
        counter_config = {**config, 'gate_mode': 'off'}
        without_gate = build_agent(self.nlu, bank=self.bank, generator=pipeline.generator, **counter_config)
        session_id = uuid4().hex
        session = Session(session_id, state={'listing_price': config['listing_price'], 'offers': [], 'agreement': 'none'})
        self.sessions[session_id] = (session, pipeline, without_gate)
        # The buyer makes the first move, answering the listing itself.
        listing = ' '.join(str(data.get(key, '')) for key in ('title', 'description')).strip()
        result = pipeline.open(session, context=listing)
        generation = result.generation
        opening = {**asdict(generation), 'reply': generation.text, **self.wording(pipeline, generation),
                   'probabilities': {}, 'price_reference': {}, 'policy': result.trace['policy'],
                   'gate': {'mode': config['gate_mode'], 'reasons': [], 'threshold': config['threshold']},
                   'offers': result.snapshot.state['offers'], 'agreement': result.snapshot.state['agreement'],
                   'finished': False, 'opening': True,
                   'counterfactual': {'reply': generation.text, 'action': generation.action}}
        return {'session_id': session_id, 'nlu_provenance': self.provenance, 'config': config,
                'opening': opening}

    @staticmethod
    def wording(pipeline, generation):
        """The bank row behind a retrieved reply: its template and original sentence."""
        row = next((row for row in pipeline.generator.bank if row['template_id'] == generation.template_id), {})
        return {'template': row.get('template'), 'original_response': row.get('response'),
                'source_dialogue_id': generation.source_dialogue_id}

    def turn(self, data):
        session, pipeline, without_gate = self.lookup(data)
        message = data.get('message')
        if not isinstance(message, str) or not message.strip() or len(message) > 4000:
            raise ValueError('message must contain 1–4000 characters')
        if session.snapshot().finished:
            raise ValueError('finished session requires a new negotiation')
        before = session.snapshot()
        turn_id = uuid4().hex
        result = pipeline.turn(session, message, case_id=session.session_id, turn_id=turn_id)
        # A separate restored session: this branch never publishes to the live ledger.
        try:
            alternative = without_gate.turn(Session.from_snapshot('counterfactual', before), message,
                                             case_id=session.session_id, turn_id=turn_id)
            counterfactual = {'reply': alternative.generation.text, 'action': alternative.generation.action,
                              'price': alternative.generation.price, 'finished': alternative.snapshot.finished}
        except Exception as exc:
            counterfactual = {'error': str(exc), 'reply': None, 'action': None}
        generation, trace = result.generation, result.trace
        state = result.snapshot.state
        return {**asdict(generation), 'reply': generation.text, **self.wording(pipeline, generation),
                'probabilities': trace['probabilities'], 'price_reference': trace['price_reference'],
                'gate': trace['gate'], 'policy': trace['policy'], 'trace': trace,
                'offers': state.get('offers', []), 'agreement': state.get('agreement', 'none'),
                'finished': result.snapshot.finished, 'counterfactual': counterfactual}

    def lookup(self, data):
        session_id = data.get('session_id')
        if not isinstance(session_id, str) or session_id not in self.sessions:
            raise KeyError('unknown session_id')
        return self.sessions[session_id]

    def reset(self, data):
        self.lookup(data)
        del self.sessions[data['session_id']]
        return {'reset': True}


def make_server(app, host='127.0.0.1', port=8000):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            content = json.dumps(payload, default=str, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == '/api/scenarios':
                self.send_json(200, {'scenarios': SCENARIOS, 'nlu_provenance': app.provenance})
            elif self.path.split('?')[0] in ('/', '/index.html'):
                content = (Path(__file__).parent / 'static' / 'index.html').read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            routes = {'/api/session': app.create, '/api/turn': app.turn, '/api/reset': app.reset}
            if self.path not in routes:
                self.send_json(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16000:
                    raise ValueError('request body must contain 1–16000 bytes')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('JSON body must be an object')
                with app.lock:
                    payload = routes[self.path](data)
                self.send_json(200, payload)
            except KeyError:
                self.send_json(404, {'error': 'unknown session_id'})
            except (ValueError, TypeError, ArithmeticError) as exc:
                self.send_json(400, {'error': str(exc)})
            except Exception as exc:
                self.send_json(422, {'error': str(exc)})

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
    parser.add_argument('--bank')
    parser.add_argument('--scripted-nlu', action='store_true')
    args = parser.parse_args()
    nlu = KeywordNLU() if args.scripted_nlu else load_nlu(args.checkpoint)
    provenance = 'demo:keyword-nlu (NOT the real model)' if args.scripted_nlu else f'real checkpoint: {Path(args.checkpoint).name}'
    server = make_server(DemoApp(nlu, bank=args.bank, provenance=provenance), port=args.port)
    print(f'Buyer demo: http://127.0.0.1:{server.server_port} · {provenance}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
