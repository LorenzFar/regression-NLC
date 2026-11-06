"""HTTP integration tests: no checkpoint, torch import, or network download."""
from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
from threading import Thread

from demo_app.server import DemoApp, KeywordNLU, SCENARIOS, make_server

BANK = Path(__file__).resolve().parents[1] / 'examples' / 'approved_fixture_bank.jsonl'


@contextmanager
def running_app():
    app = DemoApp(KeywordNLU(), bank=BANK)
    server = make_server(app, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection('127.0.0.1', server.server_port, timeout=10)

    def request(path, data=None):
        body = None if data is None else json.dumps(data)
        connection.request('GET' if data is None else 'POST', path, body,
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read()
        return response.status, json.loads(raw)

    try:
        yield app, request
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def create(request, **overrides):
    status, payload = request('/api/session', {**SCENARIOS[0], **overrides})
    assert status == 200, payload
    return payload['session_id']


def test_create_turn_trace_and_isolated_counterfactual():
    with running_app() as (app, request):
        status, presets = request('/api/scenarios')
        assert status == 200 and len(presets['scenarios']) == 4
        session_id = create(request)
        assert app.sessions[session_id][0].snapshot().state['listing_price'] == '300'
        status, turn = request('/api/turn', {'session_id': session_id, 'message': "I'm asking 300 for it"})
        assert status == 200, turn
        assert turn['reply'] and turn['action'] == 'counter'
        assert set(turn['probabilities']) == {'propose', 'counter', 'accept', 'reject', 'inquire', 'vague'}
        assert {'price', 'status', 'referenced_offer_id', 'reason'} <= turn['price_reference'].keys()
        assert {'mode', 'reasons', 'threshold'} <= turn['gate'].keys()
        assert turn['trace']['policy'] == turn['policy']
        assert turn['counterfactual']['reply']
        snapshot = app.sessions[session_id][0].snapshot()
        assert len(snapshot.history) == 2
        assert snapshot.state['offers'] == turn['offers']
        assert len(snapshot.applied_turn_ids) == 1


def test_confirmation_finishes_and_rejects_later_input():
    with running_app() as (app, request):
        session_id = create(request)
        status, opened = request('/api/turn', {'session_id': session_id, 'message': "I'm asking 300"})
        assert status == 200
        status, turn = request('/api/turn', {'session_id': session_id, 'message': 'Deal!'})
        assert status == 200, turn
        assert turn['action'] == 'confirm' and turn['finished']
        assert turn['agreement'] == 'confirmed'
        before = app.sessions[session_id][0].snapshot()
        status, error = request('/api/turn', {'session_id': session_id, 'message': 'another offer'})
        assert status == 400 and 'finished' in error['error']
        assert app.sessions[session_id][0].snapshot() == before


def test_bad_session_and_reset():
    with running_app() as (app, request):
        status, error = request('/api/turn', {'session_id': 'missing', 'message': 'hi'})
        assert status == 404 and 'session_id' in error['error']
        session_id = create(request)
        assert request('/api/reset', {'session_id': session_id}) == (200, {'reset': True})
        assert session_id not in app.sessions
        assert request('/api/turn', {'session_id': session_id, 'message': 'hi'})[0] == 404


def test_bad_settings_and_message_do_not_commit():
    with running_app() as (app, request):
        for overrides in ({'target': '-1'}, {'walk_away': 'NaN'}, {'threshold': 2},
                          {'fraction': 0}, {'gate_mode': 'unknown'}, {'target': 999}):
            assert request('/api/session', {**SCENARIOS[0], **overrides})[0] == 400
        session_id = create(request)
        before = app.sessions[session_id][0].snapshot()
        assert request('/api/turn', {'session_id': session_id, 'message': ''})[0] == 400
        assert app.sessions[session_id][0].snapshot() == before


def test_clarification_counterfactual_never_mutates_live_state():
    with running_app() as (app, request):
        session_id = create(request, threshold=.99)
        before = app.sessions[session_id][0].snapshot()
        status, turn = request('/api/turn', {'session_id': session_id, 'message': "I'm asking 300"})
        assert status == 200, turn
        assert turn['action'] == 'clarify'
        assert 'low_confidence' in turn['gate']['reasons']
        assert turn['counterfactual']['action'] == 'counter'
        assert app.sessions[session_id][0].snapshot().state == before.state
