import tempfile
import unittest
from pathlib import Path

from enterprise_rag.core import KnowledgeBase, evaluate, extractive_answer, split_chunks


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.kb = KnowledgeBase(Path(self.directory.name) / 'test.db')

    def test_ingest_idempotency_update_and_delete_index(self):
        first = self.kb.ingest('a', '政策', '签收后30天内可以退货。')
        second = self.kb.ingest('a', '政策', '签收后30天内可以退货。')
        self.assertEqual(first['document_id'], second['document_id'])
        self.assertFalse(second['changed'])
        self.assertEqual(len(self.kb.search('a', '退货')), 1)
        updated = self.kb.ingest('a', '政策', '只接受换货申请。')
        self.assertEqual(updated['document_id'], first['document_id'])
        self.assertEqual(self.kb.search('a', '退货'), [])
        self.assertEqual(len(self.kb.search('a', '换货')), 1)
        self.assertTrue(self.kb.delete('a', first['document_id']))
        self.assertEqual(self.kb.search('a', '换货'), [])

    def test_tenant_isolation_and_query_syntax(self):
        doc = self.kb.ingest('a', 'secret', '管理员密码保存在保险箱。')
        self.assertEqual(self.kb.search('b', '管理员密码'), [])
        self.assertFalse(self.kb.delete('b', doc['document_id']))
        self.assertEqual(self.kb.search('a', '" OR NOT *'), [])
        self.assertEqual(len(self.kb.search('a', '管理员密码')), 1)

    def test_answer_and_golden_evaluation(self):
        document = self.kb.ingest('demo', '退货', '签收后的30天内可以申请退货。')
        hits = self.kb.search('demo', '退货期限30天')
        answer, citations = extractive_answer('退货期限30天', hits)
        self.assertIn('[1]', answer)
        self.assertEqual(citations[0]['document_id'], document['document_id'])
        report = evaluate(self.kb, 'demo', [{'question': '退货期限30天',
                          'expected_document_id': document['document_id'],
                          'answer_contains': ['30天']}])
        self.assertEqual(report['recall_at_k'], 1.0)
        self.assertEqual(report['mrr_at_k'], 1.0)
        self.assertEqual(report['citation_rate'], 1.0)

    def test_missing_evidence_and_size_limit(self):
        self.assertEqual(extractive_answer('不存在', [])[1], [])
        with self.assertRaises(ValueError):
            self.kb.ingest('a', 'big', 'x' * 1_000_001)
        self.assertEqual(self.kb.search('a', 'big'), [])

    def test_chunk_overlap_and_metrics(self):
        chunks = split_chunks('a' * 1000)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][-80:], chunks[1][:80])
        self.kb.trace('demo', 'ask', 'error', 12.5, error_type='timeout')
        self.assertEqual(self.kb.metrics('demo')['by_error']['timeout'], 1)


if __name__ == '__main__':
    unittest.main()
