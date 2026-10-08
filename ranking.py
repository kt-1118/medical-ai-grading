"""Google Sheetsへの同時提出をまとめ、サーバー内でランキングを共有する。"""
from concurrent.futures import Future
import json
import math
import re
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class RankingStore:
    def __init__(self, url, api_token, batch_delay=1.0):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.netloc != 'script.google.com' or not re.fullmatch(r'/macros/s/[A-Za-z0-9_-]+/exec', parsed.path):
            raise ValueError('ランキングの接続先を確認してください。')
        if len(api_token) < 32:
            raise ValueError('ランキングの接続設定を確認してください。')
        self.url, self.api_token = url, api_token
        self.batch_delay = batch_delay
        self.lock = threading.Lock()
        self.network_lock = threading.Lock()
        self.pending = {}
        self.waiters = []
        self.running = False
        self.rows = []
        self.fetched_at = 0

    def _request(self, action, entries=None):
        payload = {'token': self.api_token, 'action': action, 'entries': entries or []}
        request = Request(self.url, data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(request, timeout=15) as response:
            result = json.loads(response.read())
        if not result.get('ok'):
            raise ValueError('ランキングとの通信に失敗しました。')
        rows = result['scores']
        if not isinstance(rows, list) or any(
            not isinstance(row, list) or len(row) != 2 or
            not re.fullmatch(r'[0-9a-f]{64}', str(row[0])) or
            not isinstance(row[1], (int, float)) or not math.isfinite(row[1]) or not 0 <= row[1] <= 1
            for row in rows
        ):
            raise ValueError('ランキングの応答を確認してください。')
        return rows

    def _cache(self, rows):
        with self.lock:
            self.rows = rows
            self.fetched_at = time.monotonic()

    def submit(self, key, auc):
        if not re.fullmatch(r'[0-9a-f]{64}', key) or not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError('登録するスコアが不正です。')
        ticket = Future()
        with self.lock:
            self.pending[key] = max(self.pending.get(key, 0), auc)
            self.waiters.append(ticket)
            if not self.running:
                self.running = True
                threading.Thread(target=self._flush, daemon=True).start()
        return ticket

    def _flush(self):
        while True:
            time.sleep(self.batch_delay)
            with self.lock:
                entries, waiters = list(self.pending.items()), self.waiters
                self.pending, self.waiters = {}, []
            try:
                with self.network_lock:
                    rows = self._request('submit', entries)
                    confirmed = dict(rows)
                    if any(confirmed.get(key, -1) < auc for key, auc in entries):
                        raise ValueError('スコアの保存を確認できませんでした。')
                    self._cache(rows)
                for ticket in waiters:
                    ticket.set_result(True)
            except Exception:
                for ticket in waiters:
                    ticket.set_exception(RuntimeError('ランキングへの保存に失敗しました。'))
            with self.lock:
                if not self.pending:
                    self.running = False
                    return

    def snapshot(self):
        with self.network_lock:
            if time.monotonic() - self.fetched_at >= 30:
                self._cache(self._request('ranking'))
            with self.lock:
                return [row[:] for row in self.rows]


def ranking_rows(scores, own_key=None):
    best = {}
    for key, auc in scores:
        best[key] = max(best.get(key, 0), auc)
    ordered = sorted(best.items(), key=lambda pair: (-pair[1], pair[0]))
    rows, rank, previous = [], 0, None
    for index, (key, auc) in enumerate(ordered, 1):
        if auc != previous:
            rank = index
        rows.append({'position': rank, 'auc': auc, 'is_self': key == own_key, 'participants': len(ordered)})
        previous = auc
    return rows
