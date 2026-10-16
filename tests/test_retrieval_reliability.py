"""Offline retrieval regressions: no checkpoint loading or external requests."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.retrieval import RetrievalStore
from models.web_retrieval import search_local, search_web, should_search


def cards(*items):
    return ''.join(f'<li class="b_algo"><h2><a href="{url}">{title}</a></h2>'
                   f'<p>{snippet}</p></li>' for url, title, snippet in items)


class WebEvidenceTests(unittest.TestCase):
    def test_recency_and_request_words_cannot_make_unrelated_hit_relevant(self):
        html = cards(('https://example.com/history', 'History archive',
                      'Latest search results 2026 about ancient empires'))
        with patch('models.web_retrieval.fetch', return_value=(html, 'https://example.com/history')):
            result = search_web('latest quantum error correction 2026')
        self.assertEqual(result['status'], 'empty')
        self.assertEqual(result['hits'], [])

    def test_version_prefix_is_not_exact_model_match(self):
        html = cards(('https://example.com/qwen', 'Qwen3.50 release', 'Qwen3.50 documentation'))
        with patch('models.web_retrieval.fetch', return_value=(html, 'https://example.com/qwen')):
            self.assertEqual(search_web('Qwen3.5 latest')['hits'], [])

    def test_narrowed_retry_must_still_match_original_subject(self):
        initial = '<html>No organic results</html>'
        narrowed = cards(('https://example.com/qwen', 'Qwen3.5', 'Qwen3.5 inference deployment'))
        with patch('models.web_retrieval.fetch', side_effect=[(initial, ''), (narrowed, ''),
                   ('<p>Qwen3.5 supports inference deployment.</p>', 'https://example.com/qwen')]):
            result = search_web('Qwen3.5 quantum error correction benchmark')
        self.assertEqual(result['hits'], [])

    def test_unrelated_redirect_body_cannot_be_promoted_to_page_evidence(self):
        html = cards(('https://example.com/qwen', 'Qwen3.5', 'Qwen3.5 technical report'))
        with patch('models.web_retrieval.fetch', side_effect=[(html, ''),
                   ('<p>This is an unrelated login page with a long account management notice.</p>',
                    'https://example.com/login')]):
            result = search_web('Qwen3.5 technical report')
        hit = result['hits'][0]
        self.assertEqual(hit['fetch_status'], 'search_snippet_only')
        self.assertEqual(hit['url'], 'https://example.com/qwen')
        self.assertNotIn('account management', hit['text'])
        self.assertEqual(result['status'], 'limited')

    def test_relevant_page_retains_provenance(self):
        html = cards(('https://example.com/report', 'Qwen3.5 report', 'Qwen3.5 technical report'))
        body = '<p>Qwen3.5 technical report describes the architecture and evaluation methodology.</p>'
        with patch('models.web_retrieval.fetch', side_effect=[(html, ''), (body, 'https://example.com/report')]):
            result = search_web('Qwen3.5 technical report')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['hits'][0]['fetch_status'], 'page_excerpt')
        self.assertIn('不是文章发布日期', result['hits'][0]['text'])

    def test_invalid_worker_payload_is_failure_not_a_chat_crash(self):
        for payload in ([], {'status': 'ok', 'hits': 'bad'}, {'status': 'ok', 'hits': [{}]}):
            with self.subTest(payload=payload), patch('models.web_retrieval.subprocess.run',
                    return_value=SimpleNamespace(stdout=json.dumps(payload))):
                result = search_local('quantum error correction')
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['hits'], [])

    def test_timeout_has_machine_readable_reason(self):
        with patch('models.web_retrieval.subprocess.run', side_effect=subprocess.TimeoutExpired('search', 35)):
            result = search_local('quantum error correction')
        self.assertEqual(result.get('error'), 'timeout')
        self.assertEqual(result['hits'], [])

    def test_worker_cannot_claim_full_evidence_for_snippets(self):
        hit = {'id': 'W1', 'kind': 'web', 'text': 'Quantum error correction research',
               'source': 'Report', 'url': 'https://example.com/report',
               'retrieved_at': '2026-10-15T00:00:00Z', 'fetch_status': 'search_snippet_only'}
        with patch('models.web_retrieval.subprocess.run', return_value=SimpleNamespace(
                stdout=json.dumps({'status': 'ok', 'hits': [hit]}))):
            self.assertEqual(search_local('quantum error correction')['status'], 'failed')

    def test_valid_limited_worker_result_is_preserved(self):
        hit = {'id': 'W1', 'kind': 'web', 'text': 'Quantum error correction research',
               'source': 'Report', 'url': 'https://example.com/report',
               'retrieved_at': '2026-10-15T00:00:00Z', 'fetch_status': 'search_snippet_only'}
        payload = {'status': 'limited', 'hits': [hit], 'notice': 'Only snippets'}
        with patch('models.web_retrieval.subprocess.run',
                   return_value=SimpleNamespace(stdout=json.dumps(payload))):
            self.assertEqual(search_local('quantum error correction'), payload)

    def test_existing_simple_question_routing_preserved(self):
        for question in ('你是谁', '解释勾股定理', '检索是什么意思', '不要联网，解释光合作用'):
            with self.subTest(question=question):
                self.assertFalse(should_search(question))
        self.assertTrue(should_search('搜索量子纠错最新进展'))


class LocalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.store = RetrievalStore(':memory:')
        self.addCleanup(self.store.close)

    def test_generic_request_overlap_does_not_retrieve_unrelated_memory(self):
        self.store.remember('请搜索历史论文，今天我想了解帝国。', 'demo')
        self.assertEqual(self.store.search('请搜索今天量子纠错论文', session='demo'), [])

    def test_relevant_memory_remains_and_other_session_is_isolated(self):
        self.store.remember('量子纠错通过冗余编码保护量子信息。', 'demo')
        self.store.remember('量子纠错资料不应跨会话。', 'other')
        hits = self.store.search('请搜索量子纠错论文', session='demo', include_knowledge=False)
        self.assertEqual(len(hits), 1)
        self.assertIn('冗余编码', hits[0]['text'])

    def test_version_identifiers_cannot_retrieve_nearby_versions(self):
        self.store.remember('Qwen3.50 technical report and architecture', 'demo')
        self.assertEqual(self.store.search('Qwen3.5 technical report', session='demo'), [])

    def test_requested_local_document_still_searchable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.md'
            path.write_text('量子纠错通过冗余编码保护量子信息。', encoding='utf-8')
            self.store.ingest(path)
            self.assertTrue(self.store.search('根据本地文档解释量子纠错'))
            self.assertEqual(self.store.search('根据本地文档解释量子纠错', include_knowledge=False), [])


class SemanticEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from models.semantic_retrieval import SemanticIndex
        cls.np = np
        cls.index_class = SemanticIndex

    def make_index(self, texts, scores):
        index = self.index_class.__new__(self.index_class)
        index.docs = [{'source': f'https://arxiv.org/abs/{i}', 'text': text} for i, text in enumerate(texts)]
        index.vectors = self.np.asarray(scores, dtype=self.np.float32).reshape(-1, 1)
        index.encoder = SimpleNamespace(encode=lambda *a, **kw: self.np.asarray([[1.0]]))
        return index

    def test_empty_semantic_index_is_an_empty_result(self):
        index = self.make_index([], [])
        index.encoder = SimpleNamespace(encode=lambda *a, **kw: self.fail('empty index encoded query'))
        self.assertEqual(index.search('量子纠错论文'), [])

    def test_nearby_version_is_not_semantic_evidence_for_exact_identifier(self):
        index = self.make_index(['Qwen3.50 report', 'Qwen3.5 report'], [0.95, 0.90])
        hits = index.search('Qwen3.5 report', limit=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['text'], 'Qwen3.5 report')

    def test_nonfinite_scores_never_become_evidence(self):
        index = self.make_index(['invalid score', 'quantum error correction'], [float('nan'), 0.8])
        self.assertEqual([hit['text'] for hit in index.search('量子纠错论文')], ['quantum error correction'])

    def test_cross_language_semantic_results_are_not_lexically_rejected(self):
        index = self.make_index(['Quantum error correction using redundant encoding'], [0.85])
        self.assertEqual(len(index.search('量子纠错论文')), 1)


class ChatPolicyIntegrationTests(unittest.TestCase):
    def test_turn_policy_does_not_persist_but_conversation_does(self):
        from contextlib import contextmanager
        from models import chat_pretrained

        prepared = []
        questions = iter(['请用三条说明今年的技术大会发布', '离线回答 1+1', '/exit'])

        class Backend:
            identity = {'fingerprint': 'offline-test', 'vocab_size': 64}
            description = {'model': 'offline-test-model'}
            last_response_truncated = False

            def __init__(self, *args, **kwargs):
                pass

            def prepare(self, system, history, question, hits, *args):
                prepared.append((system, list(history), question, list(hits)))
                return [], history, hits

            def answer(self, *args, **kwargs):
                return '测试回答'

            def encode(self, *args, **kwargs):
                return [1, 2]

        class UI:
            def __init__(self, *args, **kwargs):
                pass

            def read(self):
                return next(questions)

            @contextmanager
            def response(self):
                yield lambda text: None

        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / 'session.sqlite'
            argv = ['chat', '--db', str(db), '--session', 'policy-test', '--device', 'cpu',
                    '--web', 'on', '--context-length', '4096', '--max-new-tokens', '128']
            with patch('sys.argv', argv), patch.object(chat_pretrained, 'PretrainedChatModel', Backend), \
                    patch.object(chat_pretrained, 'ChatUI', UI), \
                    patch.object(chat_pretrained, 'configure_chat_diagnostics'), \
                    patch.object(chat_pretrained, 'search_local', return_value={
                        'status': 'failed', 'hits': [], 'notice': 'searchTimeout'}) as search, \
                    patch('builtins.print'):
                chat_pretrained.main()
            store = RetrievalStore(db)
            self.addCleanup(store.close)
            restored = store.load_messages('policy-test', Backend.identity, 4096)

        self.assertEqual(search.call_count, 1)
        self.assertIn('本轮未获得可用联网证据', prepared[0][0])
        self.assertIn('稳定知识', prepared[0][0])
        self.assertNotIn('searchTimeout', prepared[1][0])
        self.assertIn('本轮没有执行联网检索', prepared[1][0])
        self.assertIn('上一题的字数、条数和格式要求只适用于上一题', prepared[1][0])
        self.assertEqual(prepared[1][1], restored[:2])
        self.assertEqual([item['role'] for item in restored], ['user', 'assistant', 'user', 'assistant'])
        self.assertEqual(restored[0]['content'], '请用三条说明今年的技术大会发布')
        self.assertNotIn('searchTimeout', json.dumps(restored, ensure_ascii=False))

    def test_comparison_can_use_separate_versioned_sources(self):
        from models.retrieval_policy import relevant_text
        self.assertTrue(relevant_text('比较 Qwen3.5 和 Qwen3.6', 'Qwen3.5 technical report'))

    def test_model_family_accepts_named_variants(self):
        from models.retrieval_policy import identifiers_match, relevant_text
        for title in ('Qwen3.5-4B model card', 'Qwen3.5_4B model card'):
            with self.subTest(title=title):
                self.assertTrue(identifiers_match('Qwen3.5 最新模型', title))
                self.assertTrue(relevant_text('Qwen3.5 最新模型', title))
        self.assertTrue(relevant_text('Qwen3.5-4B model card', 'Qwen3.5-4B-Instruct model card'))

    def test_model_variant_does_not_match_sibling_or_version_prefix(self):
        from models.retrieval_policy import identifiers_match
        self.assertFalse(identifiers_match('Qwen3.5 最新模型', 'Qwen3.50 model card'))
        self.assertFalse(identifiers_match('Qwen3.5-4B', 'Qwen3.5-40B model card'))
        self.assertFalse(identifiers_match('Qwen3.5-4B', 'Qwen3.5-8B model card'))

    def test_annual_events_accept_news_review_aliases_for_requested_year(self):
        from models.retrieval_policy import relevant_text
        for title in ('2025年十大国际新闻：年度事件盘点', '2025年世界重大事件回顾',
                      'Year in review: 2025 top world news'):
            with self.subTest(title=title):
                self.assertTrue(relevant_text('2025年有哪些大事', title))

    def test_annual_event_question_framing_is_not_an_extra_subject(self):
        from models.retrieval_policy import relevant_text
        for query in ('2025年有什么大事', '2025年发生了哪些大事', 'What major events happened in 2025?'):
            with self.subTest(query=query):
                self.assertTrue(relevant_text(query, '2025年十大国际新闻：年度事件盘点'))

    def test_annual_events_reject_year_only_and_wrong_year_articles(self):
        from models.retrieval_policy import relevant_text
        for title in ('2025年量子纠错论文研究', '2024年十大国际新闻：年度事件盘点',
                      '2025年量子纠错十大论文回顾', '2025年新闻：订阅服务登录说明'):
            with self.subTest(title=title):
                self.assertFalse(relevant_text('2025年有哪些大事', title))

    def test_annual_events_keep_user_subject_constraints(self):
        from models.retrieval_policy import relevant_text
        self.assertFalse(relevant_text('2025年量子纠错有哪些大事', '2025年十大国际新闻：年度事件盘点'))
        self.assertTrue(relevant_text('2025年量子纠错有哪些大事', '2025年量子纠错重大事件回顾'))


if __name__ == '__main__':
    unittest.main()
