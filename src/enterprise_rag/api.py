"""HTTP adapter. Run with uvicorn enterprise_rag.api:app --app-dir src."""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .core import KnowledgeBase, evaluate
from .providers import generate

app = FastAPI(title='Enterprise RAG Demo', version='0.1.0')


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=1_000_000)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class EvalCase(BaseModel):
    question: str = Field(min_length=1)
    expected_document_id: str
    answer_contains: list[str] = []


class EvalIn(BaseModel):
    cases: list[EvalCase] = Field(min_length=1, max_length=100)
    k: int = Field(default=5, ge=1, le=20)


@lru_cache
def db() -> KnowledgeBase:
    return KnowledgeBase(os.getenv('RAG_DB_PATH', './rag.db'))


def tenant_auth(x_tenant_id: str = Header(default=''), x_api_key: str = Header(default='')) -> str:
    expected = os.getenv('RAG_API_KEY')
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail='invalid API key')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', x_tenant_id):
        raise HTTPException(status_code=400, detail='invalid tenant ID')
    return x_tenant_id


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/documents')
def ingest(payload: DocumentIn, tenant: str = Depends(tenant_auth)):
    began = time.perf_counter()
    try:
        result = db().ingest(tenant, payload.title, payload.content)
        db().trace(tenant, 'ingest', 'ok', (time.perf_counter() - began) * 1000)
        return result
    except ValueError as exc:
        db().trace(tenant, 'ingest', 'error', (time.perf_counter() - began) * 1000,
                   error_type='validation')
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete('/documents/{document_id}')
def remove(document_id: str, tenant: str = Depends(tenant_auth)):
    if not db().delete(tenant, document_id):
        raise HTTPException(status_code=404, detail='document not found')
    return {'deleted': True}


@app.post('/ask')
def ask(payload: AskIn, tenant: str = Depends(tenant_auth)):
    began = time.perf_counter()
    try:
        hits = db().search(tenant, payload.question, payload.k)
        answer, citations = generate(payload.question, hits)
        duration = (time.perf_counter() - began) * 1000
        trace_id = db().trace(tenant, 'ask', 'ok', duration, len(hits))
        return {'answer': answer, 'citations': citations, 'retrieved': len(hits),
                'trace_id': trace_id, 'latency_ms': round(duration, 2)}
    except (RuntimeError, ValueError, KeyError) as exc:
        db().trace(tenant, 'ask', 'error', (time.perf_counter() - began) * 1000,
                   error_type=type(exc).__name__)
        raise HTTPException(status_code=503, detail='answer generation unavailable') from exc


@app.post('/evaluate')
def run_eval(payload: EvalIn, tenant: str = Depends(tenant_auth)):
    return evaluate(db(), tenant, [c.model_dump() for c in payload.cases], payload.k)


@app.get('/metrics')
def metrics(tenant: str = Depends(tenant_auth)):
    return db().metrics(tenant)
