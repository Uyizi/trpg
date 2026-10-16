"""Multilingual E5 semantic retrieval over public paper abstracts; CPU queries."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModel
from .runtime_paths import ROOT
from .retrieval_policy import identifiers_match
MODEL=ROOT/'base/multilingual-e5-small'
INDEX=ROOT/'data/semantic_arxiv'

class Encoder:
    def __init__(self,device='cpu'):
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        self.model=AutoModel.from_pretrained(MODEL,local_files_only=True).to(device).eval()
    @torch.inference_mode()
    def encode(self,texts,query=False):
        # E5 requires these prefixes even for non-English text.
        texts=[('query: ' if query else 'passage: ')+x for x in texts]
        batch=self.tokenizer(texts,padding=True,truncation=True,max_length=512,return_tensors='pt').to(self.device)
        h=self.model(**batch).last_hidden_state
        mask=batch['attention_mask'].unsqueeze(-1)
        pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled.float(),dim=-1).cpu().numpy()

class SemanticIndex:
    def __init__(self):
        self.meta=json.loads((INDEX/'manifest.json').read_text())
        self.vectors=np.load(INDEX/'vectors.npy',mmap_mode='r')
        self.docs=[json.loads(s) for s in (INDEX/'documents.jsonl').open()]
        if len(self.docs)!=len(self.vectors):raise ValueError('Semantic index size mismatch')
        self.encoder=Encoder()
    def search(self,query,limit=4):
        if type(limit) is not int or limit < 0:
            raise ValueError('limit 必须是非负整数')
        if not query.strip() or limit == 0 or not self.docs:return []
        q=self.encoder.encode([query],query=True)[0]
        scores=np.concatenate([np.asarray(self.vectors[i:i+4096],dtype=np.float32)@q for i in range(0,len(self.docs),4096)])
        valid=np.flatnonzero(np.isfinite(scores))
        # Exact model/version identifiers cannot be replaced by a similar name.
        # No arbitrary cosine cutoff: E5 scores need corpus/query calibration.
        valid=np.asarray([i for i in valid if identifiers_match(query,self.docs[i]['text'])],dtype=np.int64)
        k=min(limit,len(valid))
        if not k:return []
        indices=valid[np.argpartition(scores[valid],-k)[-k:]]
        indices=indices[np.argsort(scores[indices])[::-1]]
        return [dict(id=f'V{int(i)}',kind='knowledge',source=self.docs[i]['source'],text=self.docs[i]['text'],similarity=float(scores[i]),
                     evidence_role='paper_snapshot_candidate') for i in indices]


def build(device='cuda'):
    if (INDEX/'manifest.json').exists():print('SEMANTIC_READY',flush=True);return
    INDEX.mkdir(parents=True,exist_ok=True)
    rows=[]
    for split in ('train','validation'):
        for line in (ROOT/'data/arxiv_snapshot'/f'{split}.jsonl').open():
            row=json.loads(line)
            rows.append({'source':row.get('source_url') or 'https://arxiv.org/abs/'+str(row['paper_id']),
                         'text':row['text'],'paper_id':row['paper_id']})
    encoder=Encoder(device)
    temp=INDEX/'vectors.partial.npy'
    first=encoder.encode([rows[0]['text']])
    vectors=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float16,shape=(len(rows),first.shape[1]))
    for start in range(0,len(rows),32):
        vectors[start:start+32]=encoder.encode([r['text'] for r in rows[start:start+32]])
        if start%1024==0:print('INDEX_PROGRESS',start,len(rows),flush=True)
    vectors.flush();del vectors
    with (INDEX/'documents.jsonl').open('w') as out:
        for row in rows:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    temp.replace(INDEX/'vectors.npy')
    pending=INDEX/'manifest.tmp'
    pending.write_text(json.dumps({'state':'completed','documents':len(rows),'model':str(MODEL),'method':'E5 masked mean pooling, L2 normalized cosine','max_embedding_tokens':512}))
    pending.replace(INDEX/'manifest.json')
    print('SEMANTIC_READY',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--query')
    args=p.parse_args()
    if args.build:build(args.device)
    elif args.query:print(json.dumps(SemanticIndex().search(args.query),ensure_ascii=False))
