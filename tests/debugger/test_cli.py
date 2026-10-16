import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from qwen_debugger.cli import main
from qwen_debugger.storage import Store


class CliTests(unittest.TestCase):
    def test_dry_run_creates_no_job_and_queue_is_persistent(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            base=['--data-dir',d]
            self.assertEqual(main(base+['cases','import','examples/debugger/owner_transfer.json']),0)
            replay=['replay','--case','owner','--profile','examples/debugger/mock.json']
            self.assertEqual(main(base+replay+['--dry-run']),0)
            self.assertEqual(Store(Path(d)).rows('jobs'),[])
            self.assertEqual(main(base+replay+['--queue']),0)
            s=Store(Path(d)); self.assertEqual(len(s.rows('jobs')),1)
            self.assertEqual(s.rows('jobs')[0]['state'],'PENDING')
            self.assertEqual(s.rows('attempts'),[])
