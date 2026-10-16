import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class RangeDownloadTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('models.mtp_range_download'))
        from models import mtp_range_download
        return mtp_range_download

    def test_server_ignoring_range_is_rejected_before_any_body_is_needed(self):
        m=self.module()
        with self.assertRaises(ValueError):m.validate_range(200,{'content-length':'9320000000'},0,7)

    def test_exact_range_and_total_are_required(self):
        m=self.module()
        self.assertEqual(m.validate_range(206,{'content-range':'bytes 0-7/500','content-length':'8'},0,7),500)
        for headers in ({'content-range':'bytes 1-8/500','content-length':'8'},
                        {'content-range':'bytes 0-7/500','content-length':'9'},
                        {'content-range':'bytes 0-7/501','content-length':'8'}):
            with self.assertRaises(ValueError):m.validate_range(206,headers,0,7,500)

    def test_only_official_https_origin_and_cdn_redirects_are_allowed(self):
        m=self.module()
        m.allowed_url('https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-4B/repo')
        m.allowed_url('https://cdn-lfs.hf.co/a')
        for url in ('http://huggingface.co/a','https://huggingface.co.evil.test/a',
                    'https://127.0.0.1/a','https://user:pass@huggingface.co/a'):
            with self.assertRaises(ValueError):m.allowed_url(url)

    def test_resume_does_not_mix_changed_source_etag_or_index(self):
        m=self.module()
        name='mtp.fc.weight';shard='model-00001-of-00001.safetensors'
        header=json.dumps({name:{'dtype':'BF16','shape':[1],'data_offsets':[0,2]}}).encode()
        class Transfer:
            version=1
            def __init__(self):
                self.bytes=0;self.connections=[];self.dn=SimpleNamespace(DIAGNOSTICS=[])
            def fetch(self,url,*,begin=None,end=None,**kwargs):
                if begin is None:
                    index=json.dumps({'weight_map':{name:shard},'version':self.version}).encode()
                    return index,{'x-repo-commit':'a'*40},url
                body=struct.pack('<Q',len(header))+header+bytes([self.version,0])
                data=body[begin:end+1];self.bytes+=len(data)
                return data,{'content-range':f'bytes {begin}-{end}/{len(body)}',
                            'etag':f'"v{self.version}"'},url
        with tempfile.TemporaryDirectory() as folder,patch.object(m,'PhysicalTransfer',Transfer),redirect_stdout(io.StringIO()):
            job={'source':'huggingface','revision':'a'*40,'expected_shapes':{name:[1]},'output':folder}
            self.assertEqual(m.run(job),0)
            Transfer.version=2
            self.assertEqual(m.run(job),0)
            self.assertEqual(Path(folder,'parts/00.bin').read_bytes(),bytes([2,0]))

    def test_modelscope_requires_immutable_revision_before_network(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder,patch.object(m,'PhysicalTransfer') as transfer,redirect_stdout(io.StringIO()):
            self.assertEqual(m.run({'source':'modelscope','revision':'master','output':folder}),1)
            transfer.assert_not_called()

    def test_explicit_huggingface_revision_must_match_server_commit(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder,patch.object(m,'PhysicalTransfer') as transfer,redirect_stdout(io.StringIO()):
            transfer.return_value.fetch.return_value=(b'{"weight_map":{}}',{'x-repo-commit':'b'*40},'https://huggingface.co/a')
            transfer.return_value.bytes=0;transfer.return_value.connections=[]
            transfer.return_value.dn=SimpleNamespace(DIAGNOSTICS=[])
            result=m.run({'source':'huggingface','revision':'a'*40,'expected_shapes':{},'output':folder})
            self.assertEqual(result,1)
            saved=json.loads(Path(folder,'provenance.json').read_text())
            self.assertIn('revision',saved['message'].lower())


if __name__=='__main__':unittest.main()
