"""SQLite-backed ingestion, retrieval, attribution, and telemetry.

No external dependency is required for this module, so the core can be tested offline.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


def tokens(text: str) -> list[str]:
    """Index Latin words and overlapping CJK bigrams without a language model."""
    result: list[str] = []
    for part in re.findall(r"[a-zA-Z0-9]+|[\u3400-\u9fff]+", text.lower()):
        if '\u3400' <= part[0] <= '\u9fff':
            result.extend(part[i:i + 2] for i in range(len(part) - 1))
            if len(part) == 1:
                result.append(part)
        else:
            result.append(part)
    return result


def split_chunks(content: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    if not content.strip():
        return []
    clean = content.replace('\r\n', '\n').strip()
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        if end < len(clean):
            boundary = max(clean.rfind('\n', start + max_chars // 2, end),
                           clean.rfind('。', start + max_chars // 2, end))
            if boundary > start:
                end = boundary + 1
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    document_id: str
    title: str
    content: str
    score: float


class KnowledgeBase:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA foreign_keys=ON')
        con.execute('PRAGMA busy_timeout=10000')
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute('PRAGMA journal_mode=WAL')
            con.executescript('''
                CREATE TABLE IF NOT EXISTS documents (
                  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, title TEXT NOT NULL,
                  content_hash TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  UNIQUE (tenant, title)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                  id INTEGER PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                  ordinal INTEGER NOT NULL, content TEXT NOT NULL, search_text TEXT NOT NULL,
                  UNIQUE (document_id, ordinal)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(search_text, content='chunks', content_rowid='id');
                CREATE TRIGGER IF NOT EXISTS chunk_ai AFTER INSERT ON chunks BEGIN
                  INSERT INTO chunk_fts(rowid, search_text) VALUES (new.id, new.search_text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunk_ad AFTER DELETE ON chunks BEGIN
                  INSERT INTO chunk_fts(chunk_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
                END;
                CREATE TABLE IF NOT EXISTS traces (
                  id TEXT PRIMARY KEY, tenant TEXT NOT NULL, operation TEXT NOT NULL,
                  status TEXT NOT NULL, latency_ms REAL NOT NULL,
                  hit_count INTEGER NOT NULL DEFAULT 0, error_type TEXT,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            ''')

    def ingest(self, tenant: str, title: str, content: str) -> dict:
        if not tenant.strip() or not title.strip() or not content.strip():
            raise ValueError('tenant, title, and content are required')
        if len(content.encode('utf-8')) > 1_000_000:
            raise ValueError('document exceeds 1 MB')
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        with self._connect() as con:
            existing = con.execute('SELECT id, content_hash FROM documents WHERE tenant=? AND title=?',
                                   (tenant, title)).fetchone()
            if existing and existing['content_hash'] == digest:
                return {'document_id': existing['id'], 'changed': False, 'chunk_count':
                        con.execute('SELECT count(*) FROM chunks WHERE document_id=?',
                                    (existing['id'],)).fetchone()[0]}
            doc_id = existing['id'] if existing else str(uuid.uuid4())
            if existing:
                con.execute('DELETE FROM chunks WHERE document_id=?', (doc_id,))
                con.execute('UPDATE documents SET content_hash=? WHERE id=?', (digest, doc_id))
            else:
                con.execute('INSERT INTO documents(id,tenant,title,content_hash) VALUES (?,?,?,?)',
                            (doc_id, tenant, title, digest))
            pieces = split_chunks(content)
            con.executemany('INSERT INTO chunks(document_id,ordinal,content,search_text) VALUES (?,?,?,?)',
                            [(doc_id, i, p, ' '.join(tokens(p))) for i, p in enumerate(pieces)])
            return {'document_id': doc_id, 'changed': True, 'chunk_count': len(pieces)}

    def delete(self, tenant: str, document_id: str) -> bool:
        with self._connect() as con:
            document = con.execute('SELECT id FROM documents WHERE tenant=? AND id=?',
                                   (tenant, document_id)).fetchone()
            if not document:
                return False
            con.execute('DELETE FROM chunks WHERE document_id=?', (document_id,))
            con.execute('DELETE FROM documents WHERE tenant=? AND id=?', (tenant, document_id))
            return True

    def search(self, tenant: str, query: str, limit: int = 5) -> list[Hit]:
        terms = list(dict.fromkeys(tokens(query)))[:24]
        if not terms:
            return []
        # Every term is quoted, so user input cannot become FTS syntax.
        expression = ' OR '.join('"' + t + '"' for t in terms)
        with self._connect() as con:
            rows = con.execute('''
                SELECT c.id, d.id AS document_id, d.title, c.content,
                       bm25(chunk_fts) AS rank
                FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.rowid
                JOIN documents d ON d.id=c.document_id
                WHERE chunk_fts MATCH ? AND d.tenant=?
                ORDER BY rank, c.id LIMIT ?
            ''', (expression, tenant, limit)).fetchall()
        return [Hit(r['id'], r['document_id'], r['title'], r['content'], -r['rank']) for r in rows]

    def trace(self, tenant: str, operation: str, status: str, latency_ms: float,
              hit_count: int = 0, error_type: str | None = None) -> str:
        trace_id = str(uuid.uuid4())
        with self._connect() as con:
            con.execute('INSERT INTO traces(id,tenant,operation,status,latency_ms,hit_count,error_type) '
                        'VALUES (?,?,?,?,?,?,?)',
                        (trace_id, tenant, operation, status, latency_ms, hit_count, error_type))
        return trace_id

    def metrics(self, tenant: str) -> dict:
        with self._connect() as con:
            rows = con.execute('SELECT operation,status,latency_ms,error_type FROM traces '
                               'WHERE tenant=? ORDER BY created_at DESC LIMIT 1000', (tenant,)).fetchall()
        durations = sorted(r['latency_ms'] for r in rows)
        return {'requests': len(rows), 'errors': sum(r['status'] != 'ok' for r in rows),
                'p50_ms': durations[len(durations) // 2] if durations else None,
                'p95_ms': durations[min(len(durations) - 1, int(len(durations) * .95))] if durations else None,
                'by_operation': {op: sum(r['operation'] == op for r in rows)
                                 for op in sorted({r['operation'] for r in rows})},
                'by_error': {e: sum(r['error_type'] == e for r in rows)
                             for e in sorted({r['error_type'] for r in rows if r['error_type']})}}


def extractive_answer(question: str, hits: list[Hit]) -> tuple[str, list[dict]]:
    if not hits:
        return '知识库中没有找到足够的相关资料，无法回答。', []
    qterms = set(tokens(question))
    candidates = []
    for hit in hits:
        sentences = re.split(r'(?<=[。！？.!?])\s*|\n+', hit.content)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                candidates.append((len(qterms & set(tokens(sentence))), hit, sentence))
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [(h, s) for score, h, s in candidates if score > 0][:3]
    if not selected:
        return '检索到了资料，但没有足够的证据回答这个问题。', []
    cited: list[dict] = []
    answer_parts: list[str] = []
    for hit, sentence in selected:
        index = next((i for i, item in enumerate(cited, 1) if item['chunk_id'] == hit.chunk_id), None)
        if index is None:
            cited.append({'chunk_id': hit.chunk_id, 'document_id': hit.document_id,
                          'title': hit.title, 'excerpt': hit.content[:240]})
            index = len(cited)
        answer_parts.append(f'{sentence} [{index}]')
    return '\n'.join(answer_parts), cited


def evaluate(kb: KnowledgeBase, tenant: str, cases: list[dict], k: int = 5) -> dict:
    """Golden-set retrieval and answer checks; no LLM judge is implied."""
    if not cases:
        raise ValueError('at least one evaluation case is required')
    outcomes = []
    for case in cases:
        began = time.perf_counter()
        hits = kb.search(tenant, case['question'], k)
        answer, citations = extractive_answer(case['question'], hits)
        ids = [h.document_id for h in hits]
        expected = case['expected_document_id']
        rank = ids.index(expected) + 1 if expected in ids else None
        required = case.get('answer_contains', [])
        outcomes.append({'question': case['question'], 'hit': rank is not None, 'rank': rank,
                         'answer_coverage': all(t.lower() in answer.lower() for t in required),
                         'has_citation': bool(citations),
                         'latency_ms': round((time.perf_counter() - began) * 1000, 2)})
    count = len(cases)
    return {'case_count': count, 'recall_at_k': sum(x['hit'] for x in outcomes) / count,
            'mrr_at_k': sum(1 / x['rank'] for x in outcomes if x['rank']) / count,
            'answer_coverage': sum(x['answer_coverage'] for x in outcomes) / count,
            'citation_rate': sum(x['has_citation'] for x in outcomes) / count,
            'cases': outcomes}
