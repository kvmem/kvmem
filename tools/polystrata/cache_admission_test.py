"""Native accepted-history admission regression; run on a dedicated free GPU.

Uses valid token IDs and the native GEN protocol to control exact token lengths.
It measures cache correctness, not answer quality. Requires an existing Qwen
MoE/Gemma model supported by the pinned PolyStrata worker; downloads nothing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from qualify import Worker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--minimum',type=int,choices=(0,4096),default=4096)
    parser.add_argument('--mtp-drafts',type=int,choices=range(4),default=0)
    parser.add_argument('--long',action='store_true')
    parser.add_argument('--batch-only',action='store_true')
    parser.add_argument('--mtp-only',action='store_true',help='Require real MTP verification on longer outputs')
    args=parser.parse_args()
    if args.mtp_only and (not args.mtp_drafts or args.minimum!=4096 or args.batch_only):
        parser.error('--mtp-only requires --mtp-drafts > 0, --minimum 4096 and no --batch-only')
    args.out.mkdir(parents=True,exist_ok=False)
    config=argparse.Namespace(worker=args.worker.resolve(),model=args.model.resolve(),gpu=args.gpu,
        scenario='admission',context=131072,threads=8,batch=512,stage=512,swaps=64,
        whole=True,no_graphs=False,mode='kvmem',budget=4096,reserve=1024,host_mib=16384,
        sink=64,recent=64,aggregation='mean',verify=True,lrfu=False,history_reuse=True,
        sessions=2,session_metrics=True,fault=None,profile=False,mtp_drafts=args.mtp_drafts,
        cache_min_tokens=args.minimum,vram_reserve_mib=512,kv_type='q8_0')
    config.batch_slots=2 if args.batch_only else 0
    config.ignore_eos=args.mtp_only
    log=args.out/'native.log';worker=None;checks=[]
    report=dict(status='running',minimum=args.minimum,checks=checks,
        worker_sha256=hashlib.sha256(args.worker.read_bytes()).hexdigest())
    def save(): (args.out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def tokens(n,tag):return [100+tag]+[256+tag]*(n-2)+[32]
    def run(label,ids,count=1,stop_after=None):
        response=worker.request(ids,(max(0,len(ids)-16),len(ids)),count,stop_after=stop_after)
        text=log.read_text(encoding='utf-8',errors='replace')
        metrics={}
        for marker in ('KVMEM_SESSIONS','KVMEM_CACHE_ADMISSION'):
            lines=re.findall(r'^'+marker+r' (.*)$',text,re.M)
            if lines:metrics.update({k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',lines[-1])})
        row=dict(label=label,input_tokens=len(ids),response=response,metrics=metrics)
        checks.append(row);save();print(json.dumps(dict(label=label,input_tokens=len(ids),
            resume=response.get('resume'),finish=response.get('finish'),metrics=metrics)),flush=True)
        assert 'error' not in response,response
        return row
    try:
        worker=Worker(config,log);report.update(command=worker.command,info=worker.info);save()
        if args.mtp_only:
            # One-token fixtures do not exercise the native adaptive drafter.
            # Keep EOS out of these bounded output fixtures, and prove that
            # verification actually offered and accepted draft tokens.
            parent=tokens(6000,31);run('mtp_long_parent',parent,64)
            saved=run('mtp_short_decode',tokens(256,32),64)
            assert saved['metrics']['cache_candidate_tokens']<4096,saved
            run('mtp_another_short',tokens(64,33),32)
            restored=run('mtp_parent_after_short',parent+[41]*32)
            assert restored['response']['resume']>=5999,restored
            assert restored['metrics']['evictions']==saved['metrics']['evictions'],restored
            document=tokens(4080,34)
            crossing=run('mtp_output_crosses_boundary',document,64)
            assert len(crossing['response']['tokens'])==64,crossing
            frontier=crossing['metrics']['cache_candidate_tokens']
            assert 4096<=frontier<=4080+64,crossing
            run('mtp_park_crossing',tokens(64,35))
            restored=run('mtp_restore_crossing',document+[44]*32)
            assert restored['response']['resume']>=4079,restored
            totals={key:sum(row['response'].get(key,0) for row in checks)
                    for key in ('draft_offered','draft_accepted')}
            report['speculative_totals']=totals
            assert totals['draft_offered']>0 and totals['draft_accepted']>0,totals
            report['status']='PASS'
            return
        if args.batch_only:
            assert args.minimum==4096, 'batch regression requires --minimum 4096'
            parent=tokens(6000,1);run('batch_parent',parent)
            saved=run('park_batch_parent',tokens(64,2))
            def batch(requests):
                pending=set();events=[]
                for slot,ids,count in requests:
                    pending.add(slot)
                    worker.process.stdin.write(f"BGEN {slot} {count} query_begin={len(ids)-16} query_end={len(ids)} temperature=0 {','.join(map(str,ids))}\n")
                    worker.process.stdin.flush()
                admitted=set()
                while pending or len(admitted)<len(requests):
                    line=worker.take();events.append(line)
                    assert not line.startswith('ERR '),line
                    if line.startswith('BADM '):
                        fields=line.split();slot=int(fields[1]);admitted.add(slot)
                        if fields[2]=='0':pending.discard(slot)
                    elif line.startswith('BDONE '):
                        fields=line.split();assert fields[3]!='error',line;pending.discard(int(fields[1]))
                checks.append(dict(label='batch_protocol',events=events));save()
            batch([(0,tokens(64,3),32),(1,tokens(96,4),32)])
            restored=run('parent_after_two_short_slots',parent+[41]*32)
            assert restored['response']['resume']>=5999,restored
            assert restored['metrics']['evictions']==saved['metrics']['evictions'],restored
            document=tokens(4095,21)
            batch([(0,document,3)])
            run('switch_after_batch_crossing',tokens(64,22))
            restored=run('restore_batch_crossing',document+[44]*32)
            assert restored['response']['resume']>=4094,restored
            report['status']='PASS'
            return
        parent=tokens(6000,1);run('long_parent',parent)
        short=run('short_switch',tokens(64,2))
        retained=short['metrics']['parked']
        evictions=short['metrics']['evictions']
        run('another_short',tokens(96,3))
        restored=run('long_after_short',parent+[41]*32)
        if args.minimum:
            assert retained==1 and restored['response']['resume']>=5999,restored
            assert restored['metrics']['evictions']==evictions,restored
            # Below minimum live state still supports direct accepted continuation.
            small=tokens(256,4);run('short_live',small)
            live=run('short_live_continuation',small+[42]*32)
            assert live['response']['resume']>=256,live
            before=live['metrics']['evictions']
            run('below_boundary',tokens(4095,5))
            after=run('switch_below_boundary',tokens(64,6))
            assert after['metrics']['evictions']==before,after
            for size,tag in ((4096,7),(4097,8)):
                document=tokens(size,tag);run(f'boundary_{size}',document)
                run(f'park_{size}',tokens(64,tag+10))
                reused=run(f'reuse_{size}',document+[43]*32)
                assert reused['response']['resume']>=size-1,reused
            document=tokens(4095,21);run('output_crosses_boundary',document,3)
            run('park_crossing',tokens(64,22))
            reused=run('restore_crossing',document+[44]*32)
            assert reused['response']['resume']>0,reused
            run('short_large_maximum_cancel',tokens(64,23),4096,stop_after=1)
            again=run('parent_after_cancel',document+[45]*32)
            assert again['response']['resume']>0,again
            if args.long:
                history=tokens(100000,24);run('100k_parent',history)
                run('short_after_100k',tokens(64,25))
                follow=run('100k_plus_30',history+[46]*30)
                assert follow['response']['resume']>=99999,follow
            assert checks[-1]['metrics']['cache_below_minimum']>0
        else:
            assert restored['response']['resume']==0,restored
            assert restored['metrics']['evictions']>evictions,restored
        report['status']='PASS'
    except Exception as error:
        report.update(status='FAIL',error=str(error));raise
    finally:
        if worker:worker.close()
        save()


if __name__=='__main__':main()
