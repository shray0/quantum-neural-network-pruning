"""Build the complete scaling-study report after all fixed runs finish."""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'/'scale16_v5'
required=[OUT/f'seed_{s}'/f'keep_{k}'/name for s in [1001,2002,3003] for k in [8,4] for name in ['summary.json','direct_classical.json']]
required+=[OUT/'hardware'/f'keep_{k}'/'summary.json' for k in [8,4]]
if '--wait' in sys.argv:
    while not all(p.exists() for p in required): time.sleep(20)
if not all(p.exists() for p in required): raise RuntimeError('The full fixed experiment is not complete yet')

comparisons=[]; audits=[]; timings=[]; traces=[]; labels=[]; hardware=[]; distributions=[]
for seed in [1001,2002,3003]:
    dense=json.loads((OUT/f'seed_{seed}'/'dense.json').read_text())
    for keep in [8,4]:
        base=OUT/f'seed_{seed}'/f'keep_{keep}'
        summary=json.loads((base/'summary.json').read_text()); pruning=100*(1-keep/16)
        frame=pd.read_csv(base/'comparison.csv')
        frame=frame.loc[~frame.method.isin(['Greedy swaps','Simulated annealing'])].copy()
        for r in json.loads((base/'direct_classical.json').read_text()):
            if r['method'] in ['Greedy swaps','Simulated annealing']:
                frame=pd.concat([frame,pd.DataFrame([{'method':r['method'],'state':r['state'],'parameters':keep*2*795+10,
                    'test_accuracy':r['test_accuracy'],'val_accuracy':r['val_accuracy'],
                    'validation_drop_pp':100*(dense['val_accuracy']-r['val_accuracy']),'qubo_gap':r['qubo_gap']}])],ignore_index=True)
            timings.append({'seed':seed,'pruning':pruning,'method':r['method'],'seconds':r['seconds'],'clock':'CPU elapsed, no cached score table'})
        frame['seed']=seed; frame['pruning']=pruning; frame['dense_test_accuracy']=dense['test_accuracy']
        frame['test_gain_vs_dense_pp']=100*(frame.test_accuracy-dense['test_accuracy'])
        comparisons.append(frame)
        for r in summary['surrogate_audit']: audits.append({'seed':seed,'pruning':pruning,**r})
        for r in summary['qaoa_traces']: traces.append({'seed':seed,'pruning':pruning,**r})
        labels.append({'seed':seed,'pruning':pruning,'label_training_seconds':summary['label_training_seconds'],
                       'regression_seconds':summary['regression_seconds'],'qaoa_all_local_optimization_seconds':sum(r['seconds'] for r in summary['qaoa_traces'])})

comparison=pd.concat(comparisons,ignore_index=True); audit=pd.DataFrame(audits); timing=pd.DataFrame(timings); trace=pd.DataFrame(traces)
comparison.to_csv(OUT/'all_comparisons.csv',index=False); audit.to_csv(OUT/'all_surrogate_audits.csv',index=False)
timing.to_csv(OUT/'all_direct_search_times.csv',index=False); trace.to_csv(OUT/'all_qaoa_optimization.csv',index=False)
pd.DataFrame(labels).to_csv(OUT/'preparation_times.csv',index=False)
for keep in [8,4]:
    s=json.loads((OUT/'hardware'/f'keep_{keep}'/'summary.json').read_text()); hardware.append(s)
    for r in s['hardware_distribution']: distributions.append(r)
hardware_frame=pd.DataFrame([{k:v for k,v in s.items() if k!='hardware_distribution'} for s in hardware])
hardware_frame.to_csv(OUT/'hardware'/'all_accuracy_comparisons.csv',index=False)
pd.DataFrame(distributions).to_csv(OUT/'hardware'/'all_distribution_metrics.csv',index=False)

methods=['QAOA best32','Random best32','Exact linear','Greedy swaps','Simulated annealing','Exact QUBO']
fig,axes=plt.subplots(1,2,figsize=(12,4.3))
for ax,pruning in zip(axes,[50,75]):
    subset=comparison.loc[comparison.pruning==pruning]
    means=subset.groupby('method').test_gain_vs_dense_pp.mean().reindex(methods)
    ax.bar(np.arange(6),means,color=['#2563eb','#888888','#a78bfa','#f59e0b','#10b981','#059669'])
    for j,name in enumerate(methods):
        values=subset.loc[subset.method==name,'test_gain_vs_dense_pp']; ax.scatter(np.repeat(j,len(values)),values,c='black',s=15,zorder=3)
    ax.axhline(0,c='black',ls='--'); ax.set(xticks=np.arange(6),xticklabels=['QAOA\nideal','Random\n32','Linear','Greedy','Anneal','Exact\nQUBO'],
        ylabel='Test accuracy difference from dense (pp)',title=f'{pruning}% pruning — three fresh initializations')
fig.suptitle('16-group local comparison: dots show every initialization')
fig.tight_layout(); fig.savefig(OUT/'local_performance.png',dpi=170); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,s in zip(axes,hardware):
    values=[s['random_mean_test_accuracy']*100,s['quantum_mean_test_accuracy_successful_batches']*100]
    ax.bar([0,1],values,color=['#888888','#2563eb'])
    for i,v in enumerate(values): ax.text(i,v+1,f'{v:.2f}%',ha='center')
    ax.set(xticks=[0,1],xticklabels=['Random 32','Hardware QAOA 32'],ylim=(0,100),ylabel='Mean retrained test accuracy (%)',
           title=f"{s['pruning_percent']:.0f}% pruning; failed Q batches {s['quantum_failed_batches']}/128")
fig.suptitle('One hardware measurement batch per budget; reconstructed 32-shot selections')
fig.tight_layout(); fig.savefig(OUT/'hardware_performance.png',dpi=170); plt.close(fig)

table='| Method | 50% pruning: mean test accuracy | 75% pruning: mean test accuracy |\n|---|---:|---:|\n'
for method in methods:
    values=[comparison.loc[(comparison.method==method)&(comparison.pruning==p),'test_accuracy'].mean()*100 for p in [50,75]]
    table+=f'| {method} | {values[0]:.2f}% | {values[1]:.2f}% |\n'
hwtable='| Pruning | Q valid shots | Mean Q accuracy, successful batches | Mean random accuracy | Failed Q batches |\n|---|---:|---:|---:|---:|\n'
for s in hardware:
    q=next(r for r in s['hardware_distribution'] if r['layer']==1)
    hwtable+=f"| {s['pruning_percent']:.0f}% | {q['feasible_fraction']:.1%} | {100*s['quantum_mean_test_accuracy_successful_batches']:.2f}% | {100*s['random_mean_test_accuracy']:.2f}% | {s['quantum_failed_batches']}/128 |\n"
time_table='| Classical search | 50% pruning: median seconds | 75% pruning: median seconds |\n|---|---:|---:|\n'
for method in timing.method.unique():
    values=[timing.loc[(timing.method==method)&(timing.pruning==p),'seconds'].median() for p in [50,75]]
    time_table+=f'| {method} | {values[0]:.6f} | {values[1]:.6f} |\n'
ledger=json.loads((OUT/'hardware'/'ledger.json').read_text()); protocol=json.loads((OUT/'hardware'/'protocol.json').read_text())
qpu=sum(j['reported_usage_seconds'] for j in ledger['jobs'])
q_audit=audit.loc[audit.model=='Quadratic'].groupby('pruning')[['r2','mae','spearman']].mean()
all_greedy_exact=all(r['qubo_gap']<1e-8 for s in [1001,2002,3003] for k in [8,4]
    for r in json.loads((OUT/f'seed_{s}'/f'keep_{k}'/'direct_classical.json').read_text()) if r['method'] in ['Greedy swaps','Simulated annealing'])
headline='This scaling run did not establish a quantum speed advantage. '+('Greedy swaps and simulated annealing found an exact QUBO optimum in all six cases.' if all_greedy_exact else 'See every classical and quantum outcome below; performance varies by instance.')
text=f'''# Qismet: 16-group scaling experiment

{headline}

## What ran

Fashion-MNIST, the existing 5,000 training/1,000 validation/10,000 test split; fully connected network **784→32→10**. There are **16 groups of two neurons** and **16 logical qubits**. Three fresh initializations: 1001, 2002, 3003. The training budget is 20 epochs per mask, unchanged from the smaller experiment.

At 50% pruning, retain eight groups (16 neurons): **12,870** masks and **12,730 parameters**, versus **25,450** dense. At 75%, retain four groups (eight neurons): **1,820** masks and **6,370 parameters**.

For each initialization and pruning budget, 192 distinct masks fit the surrogate and 64 disjoint masks audit it. Total **1,536 retraining labels**. Linear, full-quadratic and ring-quadratic surrogates are audited; the full quadratic is the predeclared primary objective. Ridge regularization uses only cross-validation on the fit masks. All QAOA and classical selections are frozen before their final test evaluation.

## Local comparisons

QAOA here means an **ideal subspace simulation**, independently checked against the actual Qiskit circuit on the hardware initialization. One and two layers, three fixed optimizer starts per depth, then depth selection by predicted mean QUBO cost. It uses a number-preserving ring XY mixer and an initial state containing exactly the required number of retained groups.

{table}

These are actual selected-network test accuracies, averaged over three initializations. One frozen 32-candidate draw per method per initialization is used, so differences include sampling variation. They are not the expected-distribution metrics reported in the earlier eight-group experiment. All individual cases remain in `all_comparisons.csv`.

![Local results](local_performance.png)

## Approximation quality

The full quadratic's mean held-out R² is **{q_audit.loc[50,'r2']:.3f}** at 50% pruning and **{q_audit.loc[75,'r2']:.3f}** at 75%. Mean absolute validation-loss error is **{q_audit.loc[50,'mae']:.4f}** and **{q_audit.loc[75,'mae']:.4f}**. Linear and sparse-ring alternatives are reported in `all_surrogate_audits.csv`. A better QUBO minimum does not guarantee better true retraining performance, because this remains an approximation.

## Real 16-qubit hardware results

Four jobs ran on **ibm_fez**: an initial-state control and one-layer QAOA at each pruning budget, using 4,096 shots each. Reported QPU usage for this follow-up totals **{qpu:g} seconds**; completed-job reported usage across the project is **{protocol['previous_reported_seconds']+qpu:g} seconds**. Prior uncertain usage remains reserved separately. Two-layer circuits were simulated but were not submitted as extra hardware jobs.

{hwtable}

The aggregate hardware counts were shuffled with a fixed seed and partitioned into **128 reconstructed 32-shot selections**. Each quantum selection chooses its best feasible mask by QUBO score; each random selection has 32 feasible candidates scored by the same QUBO. All winning states were frozen before their networks were retrained and tested. These are partitions of **one measurement pool**, not 128 independent quantum jobs. No device-repeat confidence interval is claimed.

Invalid quantum outcomes consume the fixed 32-shot budget. Failed batches are reported explicitly. The successful-batch accuracy in the table excludes failed batches; `hardware/keep_*/summary.json` also reports the utility when an unusable batch receives zero. The initial-state controls and every raw outcome are retained.

![Hardware results](hardware_performance.png)

## Search times and total work

{time_table}

These direct classical baselines score proposed masks from coefficients; greedy and annealing do not access a precomputed exhaustive score table. The primary local simulation uses an exhaustive energy table for state evolution; its earlier cached-table baseline timings are retained in per-run metadata but are not used in the table above.

Classical exact timings include enumeration and feature construction, score evaluation and argmin. They are measured implementations, not lower bounds on optimized classical performance. QPU-reported usage is a different clock from CPU elapsed time. Each hardware job's queue-inclusive observed time is in the ledger. `all_qaoa_optimization.csv` records simulator angle-optimization time and evaluations. `preparation_times.csv` records label training and regression, including audit/ablation work. All solvers share access to the learned surrogate.

Increasing the search space did not force classical methods to enumerate every candidate: local heuristics can exploit the QUBO structure. This study therefore does not establish quantum advantage or an end-to-end speedup. The meaningful outcomes are the measured pruning quality, approximation quality, solver gaps and hardware acceptance at the larger size.

## Code and evidence

- `experiment.py`: neural network, initialization rewind, fit/audit labels, surrogate fitting, ideal QAOA, classical comparisons.
- `hardware.py`: actual 16-qubit circuit, independent equivalence checks, IBM submissions and inherited usage accounting.
- `direct_classical.py`: direct-coefficient greedy, annealing and exact-search timings.
- `evaluate_hardware.py`: frozen 32-shot comparisons and final network evaluation.
- `hardware/ledger.json`: actual job IDs, device, counts paths, usage and elapsed-time evidence.
'''
(OUT/'Full_Benchmark.md').write_text(text,encoding='utf-8')
md=nbformat.v4.new_markdown_cell; code=nbformat.v4.new_code_cell
cells=[md(text),code('''from pathlib import Path
import pandas as pd
from IPython.display import display
BASE=Path('.').resolve()
display(pd.read_csv(BASE/'all_comparisons.csv')[['seed','pruning','method','parameters','test_accuracy','validation_drop_pp','qubo_gap']])
display(pd.read_csv(BASE/'all_surrogate_audits.csv'))
display(pd.read_csv(BASE/'all_direct_search_times.csv'))
'''),md('## Hardware outcomes and actual circuit source'),code('''import inspect
import hardware
print(inspect.getsource(hardware.build))
display(pd.read_csv(BASE/'hardware'/'all_distribution_metrics.csv'))
display(pd.read_csv(BASE/'hardware'/'all_accuracy_comparisons.csv'))
''')]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
nbformat.write(nb,OUT/'Full_Benchmark.ipynb'); body,_=HTMLExporter().from_notebook_node(nb)
(OUT/'Full_Benchmark.html').write_text(body,encoding='utf-8')
import subprocess
subprocess.run([sys.executable,str(ROOT/'work'/'focus_exhaustive_comparison.py')],cwd=ROOT,check=True)
print('Completed scaling report, figures and executed notebook.',flush=True)
print(comparison.groupby(['pruning','method']).test_accuracy.mean().to_string(),flush=True)
print(hardware_frame[['pruning_percent','quantum_mean_test_accuracy_successful_batches','random_mean_test_accuracy','quantum_failed_batches']].to_string(index=False),flush=True)
