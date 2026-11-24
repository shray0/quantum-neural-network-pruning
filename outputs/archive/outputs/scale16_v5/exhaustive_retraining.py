"""Actual exhaustive retraining oracle, not exhaustive evaluation of a QUBO.

One CPU thread per pruning budget, same initialization/epochs/split as Qismet.
Resume checkpoints preserve completed work and cumulative active elapsed time.
No QPU calls. Test evaluation occurs only after the exhaustive winner is frozen.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
import torch
import experiment as e

HERE=Path(__file__).resolve().parent

def run(keep):
    out=HERE/'exhaustive_retraining'/f'keep_{keep}'; out.mkdir(parents=True,exist_ok=True)
    complete=out/'complete.json'
    if complete.exists(): print('Already complete:',keep,flush=True); return
    arrays=np.load(HERE.parent/'fashion_v2'/'dataset_subset.npz')
    data={s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),
             torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val']}
    initial=torch.load(HERE/'seed_1001'/'original_initialization.pt',weights_only=True)
    masks,ids,_=e.design(keep)
    protocol={'seed':1001,'keep':keep,'masks':len(ids),'hidden':32,'groups':16,'group_size':2,
              'epochs':20,'training_samples':5000,'validation_samples':1000,'cpu_threads':1,
              'objective':'actual final retrained validation cross-entropy',
              'method':'cold rewind and full retraining of every valid mask; no QUBO or search heuristic',
              'timing_scope':'Active elapsed time for rewind, train, validation, selection and checkpoints. Data loading is outside this interval. No earlier mask metrics reused.'}
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Protocol mismatch')
    e.save(path,protocol)
    label_path=out/'all_validation_losses.csv'
    records=pd.read_csv(label_path).to_dict('records') if label_path.exists() else []
    seen={int(r['state']) for r in records}
    old=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    previous_seconds=float(old.get('active_elapsed_seconds',0)); started=time.perf_counter()
    def checkpoint():
        elapsed=previous_seconds+time.perf_counter()-started
        table=pd.DataFrame(records); temp=label_path.with_suffix('.tmp'); table.to_csv(temp,index=False); temp.replace(label_path)
        best=min(records,key=lambda r:(r['val_loss'],r['state']))
        progress={'complete':False,'keep':keep,'completed_masks':len(records),'total_masks':len(ids),
                  'active_elapsed_seconds':elapsed,'estimated_remaining_seconds':elapsed/max(1,len(records))*(len(ids)-len(records)),
                  'best_so_far_state':best['state'],'best_so_far_validation_loss':best['val_loss']}
        e.save(out/'progress.json',progress)
        print(json.dumps(progress),flush=True)
    for mask,state in zip(masks,ids):
        if int(state) in seen: continue
        start=time.perf_counter(); net=e.rewind(initial,mask); e.f.train(net,data,1001)
        val=e.f.evaluate(net,data['val'])
        records.append({'state':int(state),'val_loss':val['loss'],'val_accuracy':val['accuracy'],
                        'retraining_and_validation_seconds':time.perf_counter()-start})
        if len(records)%32==0: checkpoint()
    checkpoint()
    best=min(records,key=lambda r:(r['val_loss'],r['state']))
    e.save(out/'winner_frozen_before_test.json',best)
    # Test loss/accuracy never choose the exhaustive winner.
    test=(torch.tensor(arrays['test_images'].reshape(-1,784)/255.,dtype=torch.float32),torch.tensor(arrays['test_labels'],dtype=torch.long))
    net=e.rewind(initial,(int(best['state'])>>np.arange(16))&1); e.f.train(net,data,1001)
    final=e.f.evaluate(net,test)
    result={**protocol,'complete':True,'winner_state':int(best['state']),
            'winner_validation_loss':best['val_loss'],'winner_validation_accuracy':best['val_accuracy'],
            'winner_test_accuracy':final['accuracy'],'winner_test_loss':final['loss'],
            'active_elapsed_seconds':previous_seconds+time.perf_counter()-started,
            'reported_mask_training_seconds':float(sum(r['retraining_and_validation_seconds'] for r in records))}
    e.save(complete,result); print('EXHAUSTIVE RETRAINING COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--keep',type=int,choices=[8,4],required=True)
    run(parser.parse_args().keep)
