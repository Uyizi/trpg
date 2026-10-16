"""Transport budget regressions, using fake clocks and sockets only."""
import socket
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from models import direct_network, web_retrieval


class DeadlineTests(unittest.TestCase):
    def test_physical_interface_failure_is_observable_without_search_fallback(self):
        error = TimeoutError('adapter discovery timed out')
        error.stage = 'physical_interface'
        output = io.StringIO()
        with patch.object(web_retrieval, 'os', SimpleNamespace(name='nt')), \
                patch.object(direct_network, 'interface', side_effect=error), \
                patch.object(web_retrieval, 'search_web') as search, \
                patch('sys.argv', ['web', '--stdio']), \
                patch('sys.stdin', io.StringIO('{"query":"Qwen3.5"}')), \
                patch('sys.stdout', output):
            web_retrieval.main()
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failure_stage'], 'physical_interface')
        self.assertEqual(result['hits'], [])
        search.assert_not_called()

    def test_adapter_setup_is_subtracted_from_connect_budget(self):
        now = [0.0]
        sock = MagicMock()
        def adapter(*args, **kwargs):
            now[0] = 4.0
            return {'index': 1, 'ip': '192.0.2.10'}
        with patch.object(direct_network, 'interface', side_effect=adapter), \
                patch.object(direct_network.socket, 'socket', return_value=sock), \
                patch('time.monotonic', side_effect=lambda: now[0]):
            with self.assertRaises(TimeoutError):
                direct_network.connect('223.5.5.5', 443, 3)
        sock.connect.assert_not_called()

    def test_body_read_uses_remaining_deadline(self):
        now = [0.0]
        raw = MagicMock()
        response = MagicMock()
        response.status = 200
        response.getheader.return_value = 'text/plain'
        response.headers.get_content_charset.return_value = 'utf-8'
        def read(size):
            now[0] += 4
            return b'a' if now[0] == 4 else b''
        response.read1.side_effect = read
        connection = MagicMock()
        connection.getresponse.return_value = response
        target = (web_retrieval.urlsplit('http://example.com/'), 'example.com',
                  [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('93.184.215.14', 80))])
        with patch.object(web_retrieval, 'public_target', return_value=target), \
                patch.object(web_retrieval.socket, 'create_connection', return_value=raw), \
                patch.object(web_retrieval.http.client, 'HTTPConnection', return_value=connection), \
                patch('time.monotonic', side_effect=lambda: now[0]):
            web_retrieval.fetch('http://example.com/', timeout=7)
        self.assertIn(unittest.mock.call(3.0), connection.sock.settimeout.call_args_list)


if __name__ == '__main__':
    unittest.main()
