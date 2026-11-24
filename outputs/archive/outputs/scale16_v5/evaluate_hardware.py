"""Frozen 32-raw-shot comparisons; never select a mask using test accuracy.

One aggregate 4096-shot hardware batch is randomly partitioned into 128 virtual
32-shot selections. These are not independent hardware jobs. Each random baseline
uses 32 feasible QUBO proposals. All failed quantum selections are retained.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
import experiment as e

HERE=Path(__file__).resolve().parent

def evaluate(keep):
    folder=HERE/'hardware'; out=folder/f'keep_{keep}'; out.mkdir(exist_ok=True)
    if (out/'summary.json').exists(): return
    a=np.load(HERE/'seed_1001'/f'keep_{keep}'/'qaoa_p1.npz')
    ids=a['ids']; pred=a['predictions']; lookup=np.full(65536,-1,dtype=int); lookup[ids]=np.arange(len(ids))
    stats=[]; all_counts={}
    top=np.argsort(pred)[:int(np.ceil(len(ids)/10))]
    for layer in [0,1]:
        result=json.loads((folder/f'keep_{keep}_p{layer}'/'result.json').read_text())
        counts=result['counts']; all_counts[layer]=counts
        full=np.zeros(65536)
        for b,n in counts.items(): full[int(b,2)]=n/4096
        raw=full[ids]; acceptance=float(raw.sum()); conditional=raw/acceptance
        stats.append({'keep':keep,'layer':layer,'job_id':result['job_id'],
                      'qpu_seconds':result['reported_usage_seconds'],'shots':4096,
                      'feasible_fraction':acceptance,'mean_feasible_qubo':float(conditional@pred),
                      'uniform_mean_qubo':float(pred.mean()),'top10_probability_conditional':float(conditional[top].sum()),
                      'top10_probability_raw':float(raw[top].sum()),'uniform_top10_probability':len(top)/len(ids)})
    rng=np.random.default_rng(20261109+keep)
    bag=np.array([int(b,2) for b,n in all_counts[1].items() for _ in range(n)])
    rng.shuffle(bag); groups=bag.reshape(128,32)
    batches=[]
    for j,bits in enumerate(groups):
        valid=lookup[bits]; valid=valid[valid>=0]
        qstate=None if len(valid)==0 else int(ids[valid[np.argmin(pred[valid])]])
        random_rows=rng.integers(0,len(ids),32)
        random_state=int(ids[random_rows[np.argmin(pred[random_rows])]])
        batches.append({'batch':j,'quantum_valid_shots':len(valid),'quantum_state':qstate,'random_state':random_state})
    e.save(out/'selection_frozen.json',{'candidate_budget':32,'seed':20261109+keep,'batches':batches,
            'frozen_before_test':True,'scope':'128 random partitions of one aggregate shot batch; not independent device repeats.'})
    # All prospective winning states are frozen before looking at any test metric.
    states=sorted({b['random_state'] for b in batches}|{b['quantum_state'] for b in batches if b['quantum_state'] is not None})
    arrays=np.load(HERE.parent/'fashion_v2'/'dataset_subset.npz')
    data={s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val','test']}
    initial=torch.load(HERE/'seed_1001'/'original_initialization.pt',weights_only=True)
    previous=pd.read_csv(HERE/'seed_1001'/f'keep_{keep}'/'comparison.csv').drop_duplicates('state').set_index('state')
    cache_path=out/'winning_mask_metrics.csv'
    cached=pd.read_csv(cache_path).to_dict('records') if cache_path.exists() else []
    known={int(r['state']):r for r in cached}
    for j,state in enumerate(states):
        if state in known: continue
        if state in previous.index:
            record={k:float(previous.loc[state,k]) for k in ['test_accuracy','val_accuracy','val_loss','test_loss']}
        else:
            mask=(state>>np.arange(16))&1; net=e.rewind(initial,mask); e.f.train(net,data,1001)
            val=e.f.evaluate(net,data['val']); test=e.f.evaluate(net,data['test'])
            record={'test_accuracy':test['accuracy'],'test_loss':test['loss'],'val_accuracy':val['accuracy'],'val_loss':val['loss']}
        record['state']=state; known[state]=record; cached.append(record)
        if len(cached)%16==0 or j==len(states)-1:
            pd.DataFrame(cached).to_csv(cache_path,index=False)
            print('Hardware accuracy check, keep',keep,len(cached),'/',len(states),'distinct winners',flush=True)
    for b in batches:
        b['random_test_accuracy']=known[b['random_state']]['test_accuracy']
        b['random_validation_loss']=known[b['random_state']]['val_loss']
        if b['quantum_state'] is None:
            b['quantum_test_accuracy']=None; b['quantum_validation_loss']=None
        else:
            b['quantum_test_accuracy']=known[b['quantum_state']]['test_accuracy']; b['quantum_validation_loss']=known[b['quantum_state']]['val_loss']
    table=pd.DataFrame(batches); table.to_csv(out/'batch_comparison.csv',index=False)
    failed=int(table.quantum_state.isna().sum())
    summary={'seed':1001,'keep':keep,'pruning_percent':100*(1-keep/16),'parameters':keep*2*795+10,
             'reconstructed_batches':128,'batch_size':32,'quantum_failed_batches':failed,
             'quantum_mean_test_accuracy_successful_batches':float(table.quantum_test_accuracy.mean()),
             'random_mean_test_accuracy':float(table.random_test_accuracy.mean()),
             'quantum_accuracy_utility_zero_for_failed_batch':float(table.quantum_test_accuracy.fillna(0).mean()),
             'quantum_mean_validation_loss_successful_batches':float(table.quantum_validation_loss.mean()),
             'random_mean_validation_loss':float(table.random_validation_loss.mean()),
             'first_frozen_quantum_state':batches[0]['quantum_state'],'first_frozen_random_state':batches[0]['random_state'],
             'first_frozen_quantum_test_accuracy':batches[0]['quantum_test_accuracy'],'first_frozen_random_test_accuracy':batches[0]['random_test_accuracy'],
             'hardware_distribution':stats,'scope':'Exploratory, first preselected initialization, one measurement batch. Report failures and acceptance. No independent-device-repeat confidence interval or speedup claim.'}
    e.save(out/'summary.json',summary); print(json.dumps(summary),flush=True)

if __name__=='__main__':
    while True:
        complete=True
        for keep in [8,4]:
            if (HERE/'hardware'/f'skip_keep_{keep}.json').exists(): continue
            paths=[HERE/'hardware'/f'keep_{keep}_p{layer}'/'result.json' for layer in [0,1]]
            if all(p.exists() for p in paths): evaluate(keep)
            else: complete=False
        if complete: break
        time.sleep(20)
