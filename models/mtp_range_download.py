"""Bounded official MTP extraction over Windows physical sockets only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import ipaddress
import json
from pathlib import Path
import re
import ssl
import struct
import sys
import threading
import time
from urllib.parse import urljoin,urlsplit,urlencode


def allowed_url(url):
    p=urlsplit(url);host=p.hostname or ''
    roots=('modelscope.cn','aliyuncs.com','alicdn.com','huggingface.co','hf.co')
    if (p.scheme!='https' or p.port not in (None,443) or p.username or p.password
        or not any(host==r or host.endswith('.'+r) for r in roots)):
        raise ValueError('Only verified official HTTPS origins/CDN redirects are allowed')
    return p


def validate_range(status,headers,begin,end,total=None):
    if status!=206:raise ValueError('Range request requires HTTP206; refusing entire shard')
    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',headers.get('content-range',''))
    if not m or (int(m[1]),int(m[2]))!=(begin,end) or int(m[3])<=end:
        raise ValueError('Content-Range mismatch')
    if total is not None and int(m[3])!=total:raise ValueError('Source total size changed')
    if headers.get('content-length') and int(headers['content-length'])!=end-begin+1:
        raise ValueError('Range Content-Length mismatch')
    if headers.get('content-encoding','identity')!='identity':raise ValueError('Compressed range refused')
    return int(m[3])


class PhysicalTransfer:
    def __init__(self,budget=280*2**20):
        if sys.platform!='win32':raise RuntimeError('Physical downloads must execute in Windows worker')
        from . import direct_network
        self.dn=direct_network;self.bytes=0;self.budget=budget;self.lock=threading.Lock();self.connections=[]
        self.preferred_ips={}
        self.dn.interface()

    def fetch(self,url,*,begin=None,end=None,total=None,etag=None,limit=2**20):
        deadline=time.monotonic()+180
        for _ in range(8):
            p=allowed_url(url)
            addresses=self.dn.resolve(p.hostname,443,timeout=self.dn.remaining(deadline,'dns'))
            if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
                raise ValueError('Public DNS destinations required')
            raw=None;last_error=None
            # A CDN can return several dead edges alongside a working one.
            # Reuse only an address that the current physical worker reached.
            preferred=self.preferred_ips.get(p.hostname)
            addresses=sorted(addresses,key=lambda a:a[4][0]!=preferred)
            for address in addresses[:8]:
                ip=address[4][0]
                try:
                    raw=self.dn.connect(ip,443,min(4,self.dn.remaining(deadline,'connect')))
                    self.connections.append({'host':p.hostname,'ip':ip,'status':'connected'});break
                except OSError as exc:
                    last_error=exc
                    self.connections.append({'host':p.hostname,'ip':ip,'status':'failed','error':type(exc).__name__})
            if raw is None:raise last_error or ConnectionError('No physical endpoint connected')
            conn=http.client.HTTPConnection(p.hostname,443)
            try:
                raw.settimeout(min(30,self.dn.remaining(deadline,'tls')))
                transport=ssl.create_default_context().wrap_socket(raw,server_hostname=p.hostname)
                self.preferred_ips[p.hostname]=ip
                conn.sock=transport
                headers={'Accept-Encoding':'identity','User-Agent':'ai-project-official-mtp-extractor/1.0'}
                if begin is not None:
                    headers['Range']=f'bytes={begin}-{end}'
                    if etag:headers['If-Match']=etag
                conn.request('GET',(p.path or '/')+('?'+p.query if p.query else ''),headers=headers)
                response=conn.getresponse();rh={k.lower():v for k,v in response.getheaders()}
                if response.status in (301,302,303,307,308):
                    url=urljoin(url,rh.get('location',''));continue
                if begin is not None:
                    validate_range(response.status,rh,begin,end,total)
                    if etag and rh.get('etag') and etag!=rh['etag']:raise ValueError('Source ETag changed')
                    limit=end-begin+1
                elif response.status!=200:raise ValueError(f'HTTP{response.status}')
                if rh.get('content-length') and int(rh['content-length'])>limit:
                    raise ValueError('Content exceeds allowed size; body not read')
                body=bytearray()
                while True:
                    transport.settimeout(min(30,self.dn.remaining(deadline,'body')))
                    block=response.read1(min(65536,limit+1-len(body)))
                    if not block:break
                    with self.lock:
                        self.bytes+=len(block)
                        if self.bytes>self.budget:raise ValueError('Cumulative download budget exceeded')
                    body.extend(block)
                    if len(body)>limit:raise ValueError('Response exceeds bound')
                if begin is not None and len(body)!=limit:raise ValueError('Truncated range')
                return bytes(body),rh,url
            finally:
                conn.close();raw.close()
        raise ValueError('Too many HTTPS redirects')


def run(job):
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    result={'status':'running','source':job['source'],'model':'Qwen/Qwen3.5-4B',
            'transport':'Windows physical IP_UNICAST_IF + source IPv4 + AliDNS HTTPS + verified TLS; no proxy fallback',
            'parts':[],'training':False}
    transfer=None
    def save():
        if transfer:
            result['received_bytes']=transfer.bytes;result['network_diagnostics']=transfer.dn.DIAGNOSTICS
            result['connections']=transfer.connections
        (out/'provenance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    try:
        revision=job.get('revision')
        if job['source'] not in ('modelscope','huggingface'):
            raise ValueError('Unsupported official model source')
        if job['source']=='modelscope' and not re.fullmatch(r'[0-9a-f]{40}',revision or ''):
            raise ValueError('ModelScope requires an immutable 40-hex revision')
        transfer=PhysicalTransfer()
        def url(filename):
            if job['source']=='modelscope':
                return 'https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-4B/repo?'+urlencode({'Revision':revision,'FilePath':filename})
            return f'https://huggingface.co/Qwen/Qwen3.5-4B/resolve/{revision or "main"}/{filename}'
        index_raw,index_headers,_=transfer.fetch(url('model.safetensors.index.json'))
        if job['source']=='huggingface':
            actual_revision=index_headers.get('x-repo-commit')
            if not actual_revision or not re.fullmatch(r'[0-9a-f]{40}',actual_revision):
                raise ValueError('Cannot pin HuggingFace revision')
            if re.fullmatch(r'[0-9a-f]{40}',revision or '') and revision!=actual_revision:
                raise ValueError('Pinned HuggingFace revision disagrees with source index')
            revision=actual_revision
        if job.get('index_sha256') and hashlib.sha256(index_raw).hexdigest()!=job['index_sha256']:
            raise ValueError('Pinned source index SHA256 mismatch')
        from .mtp_probe import _json
        index=_json(index_raw);mapping=index['weight_map'];expected=job['expected_shapes']
        names=sorted(expected)
        if any(name not in mapping for name in names):raise ValueError('Missing official MTP tensor index')
        result['revision']=revision;result['index_sha256']=hashlib.sha256(index_raw).hexdigest()
        (out/'source-index.json').write_bytes(index_raw)
        shard_info={}
        for shard in sorted({mapping[n] for n in names}):
            if Path(shard).name!=shard or not shard.endswith('.safetensors'):raise ValueError('Invalid shard name')
            first,rh,_=transfer.fetch(url(shard),begin=0,end=7)
            total=validate_range(206,rh,0,7)
            if job.get('sizes',{}).get(shard,total)!=total:raise ValueError('Pinned file size mismatch')
            header_length=struct.unpack('<Q',first)[0]
            if not 2<=header_length<=16*2**20:raise ValueError('Invalid header length')
            raw,rh,_=transfer.fetch(url(shard),begin=8,end=7+header_length,total=total,etag=rh.get('etag'))
            header=_json(raw)
            shard_info[shard]={'header':header,'header_length':header_length,'total':total,'etag':rh.get('etag'),
                               'header_sha256':hashlib.sha256(raw).hexdigest()}
            (out/(shard+'.header.json')).write_bytes(raw)
        parts=out/'parts';parts.mkdir(exist_ok=True)
        for index,name in enumerate(names):
            shard=mapping[name];info=shard_info[shard];desc=info['header'].get(name,{})
            shape=desc.get('shape');dtype=desc.get('dtype');offsets=desc.get('data_offsets')
            if shape!=expected[name] or dtype!='BF16' or not isinstance(offsets,list) or len(offsets)!=2:
                raise ValueError('Official tensor shape/dtype mismatch: '+name)
            size=2
            for dim in shape:size*=dim
            start,stop=offsets
            if type(start)is not int or type(stop)is not int or not 0<=start<stop<=info['total']-8-info['header_length'] or stop-start!=size:
                raise ValueError('Tensor source offset mismatch')
            part=parts/f'{index:02d}.bin';record_path=parts/f'{index:02d}.json'
            key={'name':name,'shard':shard,'revision':revision,'offsets':offsets,'size':size,
                 'source':job['source'],'index_sha256':result['index_sha256'],
                 'header_sha256':info['header_sha256'],'etag':info['etag']}
            if part.exists() and record_path.exists():
                old=json.loads(record_path.read_text())
                if all(old.get(k)==v for k,v in key.items()) and part.stat().st_size==size and hashlib.sha256(part.read_bytes()).hexdigest()==old['sha256']:
                    result['parts'].append(old);save();continue
            absolute=8+info['header_length']+start
            ranges=[(at,min(at+4*2**20,absolute+size)-1) for at in range(absolute,absolute+size,4*2**20)]
            def chunk(pair):
                for attempt in range(2):
                    try:return transfer.fetch(url(shard),begin=pair[0],end=pair[1],total=info['total'],etag=info['etag'])[0]
                    except (OSError,TimeoutError):
                        if attempt:raise
            with ThreadPoolExecutor(max_workers=3) as pool:
                pieces=list(pool.map(chunk,ranges))
            digest=hashlib.sha256()
            with part.with_suffix('.partial').open('wb') as handle:
                for piece in pieces:handle.write(piece);digest.update(piece)
            part.with_suffix('.partial').replace(part)
            record={**key,'sha256':digest.hexdigest()};record_path.write_text(json.dumps(record,indent=2)+'\n')
            result['parts'].append(record);save()
            print(json.dumps({'tensor':name,'bytes':size,'completed_tensors':len(result['parts']),'received_bytes':transfer.bytes}),flush=True)
        metadata={'__metadata__':{'format':'pt','source':'Qwen/Qwen3.5-4B','revision':revision}}
        offset=0
        for name,record in zip(names,result['parts']):
            metadata[name]={'dtype':'BF16','shape':expected[name],'data_offsets':[offset,offset+record['size']]};offset+=record['size']
        header=json.dumps(metadata,separators=(',',':')).encode();header+=b' '*((-len(header))%8)
        target=out/'model.safetensors';temporary=out/'model.safetensors.partial'
        with temporary.open('wb') as handle:
            handle.write(struct.pack('<Q',len(header)));handle.write(header)
            for i in range(len(names)):
                with (parts/f'{i:02d}.bin').open('rb') as source:
                    while block:=source.read(2**20):handle.write(block)
        temporary.replace(target)
        result.update(status='ok',tensor_count=len(names),payload_bytes=offset,file_bytes=target.stat().st_size,
                      sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    except Exception as exc:
        result.update(status='failed',error=type(exc).__name__,message=str(exc),stage=getattr(exc,'stage',None))
    save();print(json.dumps({k:v for k,v in result.items() if k not in ('parts','network_diagnostics')}),flush=True)
    return 0 if result['status']=='ok' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    args=p.parse_args();return run(json.loads(args.job.read_text(encoding='utf-8')))


if __name__=='__main__':raise SystemExit(main())
