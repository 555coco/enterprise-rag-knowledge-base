"""Offline interview demonstration: ingest, ask, evaluate and inspect metrics."""
import json
import tempfile
from pathlib import Path

from .core import KnowledgeBase, evaluate, extractive_answer


def main():
    dataset = Path(__file__).resolve().parents[2] / 'datasets'
    with tempfile.TemporaryDirectory() as directory:
        kb = KnowledgeBase(Path(directory) / 'demo.db')
        documents = {}
        for filename, title in [('returns.md', '退货政策'), ('shipping.md', '物流帮助')]:
            documents[title] = kb.ingest('demo', title, (dataset / filename).read_text())['document_id']
        question = '退货申请期限是多久？'
        hits = kb.search('demo', question)
        answer, citations = extractive_answer(question, hits)
        kb.trace('demo', 'ask', 'ok', 1.0, len(hits))
        cases = [json.loads(line) for line in (dataset / 'sample.jsonl').read_text().splitlines()]
        for case in cases:
            case['expected_document_id'] = documents[case.pop('expected_title')]
        print(json.dumps({'answer': answer, 'citations': citations,
                          'evaluation': evaluate(kb, 'demo', cases),
                          'metrics': kb.metrics('demo')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
