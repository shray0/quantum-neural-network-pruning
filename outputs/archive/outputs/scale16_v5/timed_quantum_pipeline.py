"""Cold, timed Qismet+QPU pipeline versus actual exhaustive retraining.

192 retraining labels; no audit retrains in the production timing; full regression,
CPU angle optimization, actual IBM submission/queue, 4096 shots, final retraining.
This is an approximate method for true retrained loss, not an optimality guarantee.
"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import experiment as e
import hardware as h
from ibm_backend import HardwareRunner,connect

HERE=Path(__file__).resolve().parent

def prior_accounting():
    accounted,reported=h.prior_usage()
    old=json.loads((HERE/'hardware'/'ledger.json').read_text())
    if any(j['state']!='complete' for j in old['jobs']): raise RuntimeError('Earlier experiment still has unresolved jobs')
    accounted+=sum(j['accounted_seconds'] for j in old['jobs'])
    reported+=sum(j.get('reported_usage_seconds',0) for j in old['jobs'])
    return accounted,reported

def run(keep):
    folder=HERE/'end_to_end'/f'quantum_keep_{keep}'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): print('Timed quantum pipeline already complete',keep,flush=True); return
    arrays=np.load(HERE.parent/'fashion_v2'/'dataset_subset.npz')
    data={s:(torch.tensor(arrays[s+'_images'].reshape(-1,784)/255.,dtype=torch.float32),torch.tensor(arrays[s+'_labels'],dtype=torch.long)) for s in ['train','val']}
    initial=torch.load(HERE/'seed_1001'/'original_initialization.pt',weights_only=True)
    masks,ids,x=e.design(keep); lookup={int(s):i for i,s in enumerate(ids)}
    old=json.loads((HERE/'seed_1001'/f'keep_{keep}'/'selection_frozen.json').read_text())
    fit_states=old['fit_states']; fit=np.array([lookup[int(s)] for s in fit_states])
    protocol={'seed':1001,'keep':keep,'fit_retrains':192,'audit_retrains_in_production':0,
              'epochs':20,'cpu_threads':1,'layers':1,'restarts':3,'max_iterations':100,
              'shots':4096,'backend':'ibm_fez','selection':'lowest predicted QUBO score among every feasible observed mask',
              'scope':'Fresh timed replay. Wall time includes label generation, regression, CPU simulation-based angle optimization, IBM connection/transpilation/submission/queue, decoding and final retraining/test. Common data loading and mask-feature enumeration are outside the timed interval.'}
    path=folder/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Protocol changed')
    e.save(path,protocol)
    state_path=folder/'progress.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'started_epoch':time.time(),'phase':'labels'}
    e.save(state_path,state)
    label_path=folder/'fit_labels.csv'
    records=pd.read_csv(label_path).to_dict('records') if label_path.exists() else []
    seen={int(r['state']) for r in records}
    for row in fit:
        s=int(ids[row])
        if s in seen: continue
        net=e.rewind(initial,masks[row]); start=time.perf_counter(); e.f.train(net,data,1001)
        val=e.f.evaluate(net,data['val'])
        records.append({'state':s,'val_loss':val['loss'],'training_seconds':time.perf_counter()-start})
        if len(records)%16==0:
            pd.DataFrame(records).to_csv(label_path,index=False)
            state.update(phase='labels',labels=len(records),wall_seconds_so_far=time.time()-state['started_epoch'])
            e.save(state_path,state); print('Timed quantum',keep,len(records),'/192 fit labels',flush=True)
    if 'label_stage_finished_epoch' not in state:
        state['label_stage_finished_epoch']=time.time(); e.save(state_path,state)
    model_path=folder/'model.npz'
    if not model_path.exists():
        labels=pd.DataFrame(records).set_index('state'); y=labels.loc[fit_states].val_loss.to_numpy()
        start=time.perf_counter(); model=RidgeCV(alphas=np.logspace(-4,3,15),cv=KFold(5,shuffle=True,random_state=1001),scoring='neg_mean_squared_error')
        model.fit(x[fit],y); predicted=model.predict(x); regression_seconds=time.perf_counter()-start
        space=e.Subspace(masks,ids,keep); scale=float(predicted.std()); normalized=(predicted-predicted.mean())/scale
        start=time.perf_counter(); results=[]
        for restart in range(3):
            theta=np.random.default_rng(1001+keep*100+10+restart).uniform(-1,1,2)
            results.append(minimize(lambda t:space.probabilities(t,normalized,1)@normalized,theta,method='BFGS',options={'maxiter':100,'gtol':1e-5}))
        best=min(results,key=lambda r:r.fun)
        np.savez(model_path,linear=model.coef_[:16],pairs=model.coef_[16:],constant=model.intercept_,theta=best.x,
                 scale=scale,ids=ids,predictions=predicted,probabilities=space.probabilities(best.x,normalized,1))
        state.update(regression_seconds=regression_seconds,angle_optimization_seconds=time.perf_counter()-start,
                     angle_evaluations=sum(r.nfev for r in results),phase='hardware',alpha=float(model.alpha_))
        e.save(state_path,state)
    assets=np.load(model_path)
    accounted,reported=prior_accounting()
    service,backend=connect(backend_name='ibm_fez')
    runner=HardwareRunner(service,backend,HERE/'end_to_end'/'hardware_ledger.json',allowance_seconds=600-accounted,
                          reserve_seconds=120,max_job_seconds=60,max_jobs=2,logical_qubits=16)
    tag=f'cold_keep_{keep}'; existing=runner.find_job(tag)
    if existing:
        if not existing.get('job_id'): raise RuntimeError('Ambiguous job submission; reconcile before continuing')
        result=runner.collect(existing['job_id'],wait=False)
    else:
        result=runner.sample(h.build(keep,1,assets=assets),shots=4096,output_dir=HERE/'end_to_end',tag=tag,wait=False)
    while result['state']=='pending':
        state.update(phase='hardware',job_id=result['job_id'],hardware_status=result.get('remote_status'),wall_seconds_so_far=time.time()-state['started_epoch'])
        e.save(state_path,state); print('Timed quantum',keep,result['job_id'],result.get('remote_status'),flush=True)
        time.sleep(20); result=runner.collect(result['job_id'],wait=False)
    if result['state']!='complete': raise RuntimeError('Hardware job failed; preserve accounting')
    b=np.zeros((16,16)); b[np.triu_indices(16,1)]=assets['pairs']
    candidates=[int(bits,2) for bits in result['counts'] if int(bits,2).bit_count()==keep]
    if not candidates: raise RuntimeError('No feasible measured mask; do not silently resample')
    def score(s):
        m=(s>>np.arange(16))&1
        return float(assets['constant']+m@assets['linear']+m@b@m)
    selected=min(candidates,key=score)
    e.save(folder/'selection_frozen_before_test.json',{'state':selected,'qubo_score':score(selected),'job_id':result['job_id']})
    net=e.rewind(initial,(selected>>np.arange(16))&1); e.f.train(net,data,1001)
    val=e.f.evaluate(net,data['val'])
    test=(torch.tensor(arrays['test_images'].reshape(-1,784)/255.,dtype=torch.float32),torch.tensor(arrays['test_labels'],dtype=torch.long))
    metrics=e.f.evaluate(net,test)
    result_summary={**protocol,'complete':True,'selected_state':selected,'validation_loss':val['loss'],'validation_accuracy':val['accuracy'],
                    'test_accuracy':metrics['accuracy'],'test_loss':metrics['loss'],
                    'end_to_end_wall_seconds':time.time()-state['started_epoch'],
                    'label_stage_wall_seconds':state['label_stage_finished_epoch']-state['started_epoch'],
                    'fit_label_training_seconds':float(sum(r['training_seconds'] for r in records)),
                    'regression_seconds':state['regression_seconds'],'angle_optimization_seconds':state['angle_optimization_seconds'],
                    'qpu_reported_seconds':result['reported_usage_seconds'],'job_id':result['job_id'],
                    'true_retraining_global_optimality_proven':False,'prior_reported_qpu_seconds':reported}
    e.save(folder/'complete.json',result_summary); print('TIMED QUANTUM PIPELINE COMPLETE',json.dumps(result_summary),flush=True)

if __name__=='__main__':
    for keep in [4,8]: run(keep)
