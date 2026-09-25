#!/usr/bin/env python3
"""Additive, resumable R2 vLLM production adapter (never imports legacy R2 runtime)."""
import argparse, csv, hashlib, json, os, socket, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from urllib.request import Request, urlopen

def sha(x): return hashlib.sha256(x).hexdigest()
def dump(p, obj):
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,sort_keys=True)+'\n'); os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--model',required=True); p.add_argument('--revision',required=True); p.add_argument('--url'); p.add_argument('--host-id',default=os.environ.get('BWB_HOST_ID')); p.add_argument('--instance-id',default=os.environ.get('BWB_INSTANCE_ID')); p.add_argument('--gpu-uuid',default=os.environ.get('BWB_GPU_UUID')); p.add_argument('--out',type=Path,default=Path('outputs/r2')); p.add_argument('--qualification-only',action='store_true'); p.add_argument('--limit',type=int,default=0); p.add_argument('--concurrency',type=int,default=8); p.add_argument('--rolling-qualification',action='store_true'); p.add_argument('--dry-run',action='store_true',help='validate frozen input keyspace without contacting a model endpoint'); a=p.parse_args()
 if a.qualification_only and not a.limit: raise SystemExit('qualification smoke needs --limit')
 if a.concurrency != 8: raise SystemExit('frozen qualified concurrency is exactly 8')
 if a.model not in {'google/gemma-4-12B-it','meta-llama/Llama-3.1-8B-Instruct','deepseek-ai/DeepSeek-R1-Distill-Llama-8B'}: raise SystemExit('model is not in the frozen R2 lock')
 expected_revisions={'google/gemma-4-12B-it':'707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7','meta-llama/Llama-3.1-8B-Instruct':'0e9e39f249a16976918f6564b8830bc894c89659','deepseek-ai/DeepSeek-R1-Distill-Llama-8B':'81cee02dd020268dced5fa1327e8555acce9c63c'}
 if a.revision != expected_revisions[a.model]: raise SystemExit('model revision differs from frozen R2 lock')
 if not a.dry_run and not all((a.url,a.host_id,a.instance_id,a.gpu_uuid)): raise SystemExit('live run requires --url plus BWB_HOST_ID, BWB_INSTANCE_ID, and BWB_GPU_UUID')
 if not a.root.is_dir(): raise SystemExit(f'prepared R2 package root is missing: {a.root}')
 if not a.dry_run: a.out.mkdir(parents=True,exist_ok=True)
 mp=a.root/'PREPARED_INPUTS_MANIFEST.json'
 if mp.exists():
  manifest=json.loads(mp.read_text())
  if manifest.get('status')!='PASS' or manifest.get('local_only') is not True: raise SystemExit('prepared input manifest is not a verified local-only package')
  for name,entry in manifest.get('files',{}).items():
   path=a.root/name
   if not path.is_file() or sha(path.read_bytes())!=entry.get('sha256'): raise SystemExit(f'prepared input hash mismatch: {name}')
 expected=list(csv.DictReader(open(a.root/'07_EXPERIMENTS/R2/EXPECTED_ROWS.csv')))
 rows=[r for r in expected if r['model_id']==a.model]
 if len(rows)!=5400: raise SystemExit(f'expected 5400 keys, got {len(rows)}')
 if len({(r['task_id'],r['reference_id'],r['treatment'],r['seed']) for r in rows})!=5400: raise SystemExit('duplicate scientific keys')
 frozen_seeds={2701,2702,2703}
 seed_cells=defaultdict(set)
 for r in rows:
  seed=int(r['seed'])
  if seed not in frozen_seeds: raise SystemExit('R2 row uses a seed outside the frozen seed set')
  seed_cells[(r['task_id'],r['reference_id'],r['treatment'])].add(seed)
 if len(seed_cells)!=1800 or any(values!=frozen_seeds for values in seed_cells.values()): raise SystemExit('R2 row seed assignment differs from the frozen three-seed design')
 gold_rows=[json.loads(line) for line in open(a.root/'R2_TASK_GOLD.jsonl')]
 if len(gold_rows)!=300 or len({r['task_id'] for r in gold_rows})!=300 or {r['task_id'] for r in gold_rows}!={r['task_id'] for r in rows}: raise SystemExit('R2 local gold/order mapping is incomplete')
 order_hashes=json.loads((Path(__file__).resolve().parents[1]/'data/input_panel_manifest.json').read_text())['panels']['r2_primary'].get('analysis_order_hashes',[])
 order_map={digest:i for i,digest in enumerate(order_hashes)}
 if len(order_map)!=300 or any(r.get('analysis_order')!=order_map.get(r.get('task_hash')) for r in gold_rows): raise SystemExit('R2 local gold/order mapping differs from the frozen anonymous analysis order')
 contract={}
 for line in open(a.root/'02_R2/R2_EXACT_RENDER_LEDGER.jsonl'):
  r=json.loads(line); contract[(r['model_id'],str(r['task_id']),r['reference_id'],r['treatment'])]=r
 if any((a.model,str(r['task_id']),r['reference_id'],r['treatment']) not in contract for r in rows): raise SystemExit('missing rendered contract key')
 if a.dry_run:
  print(json.dumps({'status':'PREFLIGHT_PASS','study':'R2_MMLUPRO300','model':a.model,'panel_tasks':len({r['task_id'] for r in rows}),'generation_rows':len(rows),'rendered_contract_rows':len(contract),'frozen_seed_rows_verified':len(rows),'frozen_row_seeds':[2701,2702,2703],'request_seed_field_sent':False,'new_network_requests':0,'new_gpu_generations':0},sort_keys=True))
  return
 binding={'model':a.model,'revision':a.revision,'instance_id':a.instance_id,'host_id':a.host_id,'gpu_uuid':a.gpu_uuid,'max_tokens':8192,'max_model_len':10496,'concurrency':8,'max_num_seqs':8,'temperature':0.0,'top_p':1.0,'top_k':-1,'row_seed_identity_only':True,'canonical_request_sends_seed':False,'qualification_only':a.qualification_only}
 bp=a.out/'binding.json'
 if bp.exists() and json.loads(bp.read_text())!=binding: raise SystemExit('same-host binding mismatch')
 dump(bp,binding)
 out=a.out/('qualification.jsonl' if a.qualification_only else 'official.jsonl')
 todo=rows[:a.limit] if a.limit else rows
 done=set()
 if out.exists():
  for line in open(out):
   key=json.loads(line)['key']
   if key in done: raise SystemExit('duplicate committed key')
   done.add(key)
 if not done.issubset({'|'.join((a.model,str(r['task_id']),r['reference_id'],r['treatment'],str(r['seed']))) for r in todo}): raise SystemExit('wrong committed key')
 def generate(r):
  key='|'.join((a.model,str(r['task_id']),r['reference_id'],r['treatment'],str(r['seed'])))
  c=contract[(a.model,str(r['task_id']),r['reference_id'],r['treatment'])]
  # The canonical R2 adapter does not transmit a seed; keep the frozen row seed as identity only.
  payload={'model':a.model,'messages':c['messages'],'temperature':0.0,'top_p':1.0,'top_k':-1,'max_tokens':8192}
  req=Request(a.url.rstrip('/')+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urlopen(req,timeout=7200) as h: ans=json.load(h)
  text=ans['choices'][0]['message']['content']; usage=ans.get('usage',{})
  rec={'key':key,'study_id':'R2_MMLUPRO300','model_id':a.model,'model_revision':a.revision,'tokenizer_revision':a.revision,'task_id':str(r['task_id']),'reference_id':r['reference_id'],'treatment':r['treatment'],'seed':int(r['seed']),'seed_sent_to_backend':False,'seed_semantics':'row identity only; canonical R2 request omits seed','raw_response':text,'response_hash':sha(text.encode()),'prompt_hash':c['prompt_hash'],'input_token_ids_sha256':c['token_ids_sha256'],'max_output_tokens':8192,'output_tokens':usage.get('completion_tokens'),'OFFICIAL_PRODUCTION':not a.qualification_only,'QUALIFICATION_ONLY':a.qualification_only,'provider_instance_id':a.instance_id,'physical_host_id':a.host_id,'gpu_uuid':a.gpu_uuid,'request_config':{'temperature':0.0,'top_p':1.0,'top_k':-1,'max_tokens':8192},'completed_at':time.time()}
  return rec
 pending=[r for r in todo if '|'.join((a.model,str(r['task_id']),r['reference_id'],r['treatment'],str(r['seed']))) not in done]
 if a.rolling_qualification:
  if not a.qualification_only: raise SystemExit('rolling scheduler is qualification-only')
  with ThreadPoolExecutor(max_workers=8) as pool:
   it=iter(pending); futures={pool.submit(generate,r) for r in [next(it,None) for _ in range(8)] if r is not None}
   with open(out,'a') as f:
    while futures:
     ready,_=wait(futures,return_when=FIRST_COMPLETED)
     for fut in ready:
      futures.remove(fut); f.write(json.dumps(fut.result(),sort_keys=True)+'\n')
      nxt=next(it,None)
      if nxt is not None: futures.add(pool.submit(generate,nxt))
     f.flush(); os.fsync(f.fileno())
  dump(a.out/'progress.json',{'model':a.model,'completed':sum(1 for _ in open(out)) if out.exists() else 0,'expected':len(todo),'updated_at':time.time(),'qualification_only':True,'scheduler':'rolling_concurrency_8'})
  return
 for i in range(0,len(pending),a.concurrency):
  batch=pending[i:i+a.concurrency]
  with ThreadPoolExecutor(max_workers=a.concurrency) as pool: finished=list(pool.map(generate,batch))
  with open(out,'a') as f:
   for rec in finished: f.write(json.dumps(rec,sort_keys=True)+'\n')
   f.flush(); os.fsync(f.fileno())
 dump(a.out/'progress.json',{'model':a.model,'completed':sum(1 for _ in open(out)) if out.exists() else 0,'expected':len(todo),'updated_at':time.time(),'qualification_only':a.qualification_only})
if __name__=='__main__': main()
