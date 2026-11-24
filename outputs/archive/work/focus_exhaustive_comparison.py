"""Focused, automatically refreshed comparison: Qismet+QPU vs exhaustive search.

Preserves the earlier complete benchmark. Never converts forecasts into measurements.
"""
from pathlib import Path
import json,sys,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'/'scale16_v5'

def read(path):
    try: return json.loads(Path(path).read_text())
    except (OSError,ValueError): return None

def build():
    # Back up the previous full presentation exactly once; source evidence stays intact.
    for source,dest in [('Results.md','Full_Benchmark.md'),('Results.html','Full_Benchmark.html'),('Scaling_Walkthrough.ipynb','Full_Benchmark.ipynb')]:
        if (OUT/source).exists() and not (OUT/dest).exists(): (OUT/dest).write_bytes((OUT/source).read_bytes())
    rows=[]; details=[]; all_complete=True
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,keep in zip(axes,[8,4]):
        pruning=100*(1-keep/16)
        base=OUT/'exhaustive_retraining'/f'keep_{keep}'
        brute=read(base/'complete.json'); progress=read(base/'progress.json')
        timing_review=read(OUT/'positive_findings'/'timing_quality_review.json') or []
        timing_bad=any(r['keep']==keep and not r['use_for_speedup_ratio'] for r in timing_review)
        qbase=OUT/'end_to_end'/f'quantum_keep_{keep}'
        quantum=read(qbase/'complete.json'); qprogress=read(qbase/'progress.json')
        if not brute or not quantum: all_complete=False
        total=12870 if keep==8 else 1820
        btime=f"{brute['active_elapsed_seconds']:.1f} s" if brute else f"INCOMPLETE: {(progress or {}).get('completed_masks',0):,}/{total:,} masks"
        if timing_bad: btime=('Enumeration complete' if brute else f"Incomplete: {(progress or {}).get('completed_masks',0):,}/{total:,}")+'; timing interrupted'
        qtime=f"{quantum['end_to_end_wall_seconds']:.1f} s" if quantum else f"RUNNING: {(qprogress or {}).get('labels',0)}/192 fit labels; {(qprogress or {}).get('phase','starting')}"
        bloss=f"{brute['winner_validation_loss']:.6f}" if brute else 'Global minimum not known yet'
        qloss=f"{quantum['validation_loss']:.6f}" if quantum else 'Pending'
        rows.append(f'| {pruning:.0f}% | {total:,} | {btime} | {qtime} | {bloss} | {qloss} |')
        table_path=base/'all_validation_losses.csv'
        table=None
        if table_path.exists():
            try: table=pd.read_csv(table_path)
            except (OSError,pd.errors.EmptyDataError): pass
        if table is not None and len(table):
            if timing_bad:
                v=table.retraining_and_validation_seconds
                stop=np.flatnonzero(v.to_numpy()>max(30,100*v.median()))
                if len(stop): table=table.iloc[:int(stop[0])].copy()
            elapsed=table.retraining_and_validation_seconds.cumsum().to_numpy()
            best=table.val_loss.cummin().to_numpy()
            ax.plot(elapsed,best,label='Exhaustive retraining: best so far')
            if quantum:
                ax.scatter([quantum['end_to_end_wall_seconds']],[quantum['validation_loss']],marker='*',s=140,label='Complete Qismet + QPU pipeline',zorder=4)
                matching=table.loc[table.val_loss<=quantum['validation_loss']+1e-7]
                if len(matching):
                    first=matching.index[0]; matched_seconds=float(elapsed[first])
                    target_note=f"The fixed enumeration order had already reached a validation loss at least as low as the quantum result after approximately **{matched_seconds:.1f} seconds of per-mask retraining work** ({int(first)+1} masks). This is a time-to-match-quality observation; the full run is still needed to certify its global minimum."
                else: target_note='The enumeration has not yet reached a validation loss as low as the quantum result.'
            else: target_note='Time-to-match-quality will be evaluated after the quantum pipeline finishes.'
        else: target_note='Exhaustive retraining is starting.'
        ax.set(xlabel='Measured processing time (seconds)',ylabel='Actual retrained validation loss',title=f'{pruning:.0f}% pruning')
        if table is not None and len(table): ax.legend(fontsize=7)
        note=f'''### {pruning:.0f}% pruning

{target_note}
'''
        if timing_bad:
            note+='\nTiming-quality review found a roughly 11,934-second shared interruption in both workers. The raw interval is preserved but is not used for a compute-speedup ratio or remaining-time forecast. The plotted enumeration prefix ends before that interruption. Validation losses remain usable.\n'
        elif progress and not brute:
            note+=f"\nThe enumerator has processed **{progress['completed_masks']:,}/{total:,}** masks in **{progress['active_elapsed_seconds']:.1f} active seconds**. Its estimated remaining time is **{progress['estimated_remaining_seconds']/60:.1f} minutes**. This is a changing estimate, not a completed benchmark.\n"
        if quantum:
            note+=f"\nThe timed quantum pipeline generated 192 fresh retraining labels, fitted the QUBO, optimized its angles on a CPU simulator, submitted an actual 16-qubit circuit, selected from 4,096 measured outcomes and retrained its final mask. Total measured wall time: **{quantum['end_to_end_wall_seconds']:.1f} s**; QPU-reported usage alone: **{quantum['qpu_reported_seconds']:g} s**. Actual test accuracy: **{quantum['test_accuracy']*100:.2f}%**.\n"
        if brute and quantum and not timing_bad:
            gap=quantum['validation_loss']-brute['winner_validation_loss']
            ratio=brute['active_elapsed_seconds']/quantum['end_to_end_wall_seconds']
            note+=f"\nExhaustive retraining took **{ratio:.2f} times** the measured quantum-pipeline wall time. The quantum mask's actual validation-loss gap to the exhaustive minimum is **{gap:.6f}**. A nonzero quality gap means this runtime ratio is a time/quality tradeoff, not a demonstrated speedup for finding the same optimum. The exhaustive winner's final test accuracy is **{brute['winner_test_accuracy']*100:.2f}%**.\n"
        details.append(note)
    fig.suptitle('Actual loss versus processing time — no alternative selection heuristics')
    fig.tight_layout(); fig.savefig(OUT/'exhaustive_retraining_comparison.png',dpi=160); plt.close(fig)
    timing=pd.read_csv(OUT/'all_direct_search_times.csv')
    cheap=[]
    for keep in [8,4]:
        pruning=100*(1-keep/16)
        cpu=timing.loc[(timing.seed==1001)&(timing.pruning==pruning)&(timing.method=='Exact QUBO, enumerate and score'),'seconds'].iloc[0]
        a=np.load(OUT/'seed_1001'/f'keep_{keep}'/'qaoa_p1.npz'); ids=a['ids']; costs=a['predictions']; optimum=int(ids[costs.argmin()])
        r=read(OUT/'hardware'/f'keep_{keep}_p1'/'result.json')
        hits=r['counts'].get(format(optimum,'016b'),0)
        cheap.append(f"| {pruning:.0f}% | {cpu*1000:.2f} ms | {r['reported_usage_seconds']:g} s QPU usage | {hits}/4,096 |")
    status='Complete' if all_complete else 'Partial results — unfinished runs are explicitly marked'
    text=f'''# Qismet + QPU versus exhaustive retraining

**Status: {status}.** This focused presentation compares quantum-assisted pruning with exhaustive search. It distinguishes exhaustive neural-network retraining from exhaustive evaluation of an already fitted QUBO. It does not benchmark other selection heuristics.

## The expensive objective

For each mask, the true experimental objective is

$$g(m)=L_{{validation}}(Train(W_0,m)).$$

Exhaustive retraining evaluates this function for **every valid mask**, with the same original initialization, 20 epochs, data, optimizer and minibatch order. Qismet estimates it from **192 retraining examples**, fits a quadratic approximation and uses QAOA to propose a mask. These methods can have different costs and different solution quality.

The model is Fashion-MNIST **784→32→10**, with 16 groups of two hidden neurons. Seed 1001 is fixed for this comparison. Each case uses one CPU thread for its classical training work. Cases may run concurrently on separate CPU threads. Data loading and common mask/feature design are outside the timed intervals; quantum wall time additionally includes its actual IBM connection, circuit preparation, queueing and final retraining.

## Measured end-to-end comparison

| Pruning | Valid masks | Exhaustive retraining time/status | Qismet + QPU time/status | Exhaustive minimum actual loss | Quantum mask actual loss |
|---|---:|---|---|---:|---:|
{chr(10).join(rows)}

![Actual retraining loss and time](exhaustive_retraining_comparison.png)

The exhaustive curves use cumulative measured per-mask training/validation time; final benchmark totals also include checkpoint/selection and final winner evaluation. They are not QUBO predictions. Partial curves are not a certificate of the global optimum.

{chr(10).join(details)}

## Why the earlier exhaustive baseline was fast

The earlier baseline enumerated the **quadratic formula**, not the neural-network retraining objective. A learned QUBO score has only 136 nonconstant terms. Evaluating those terms for thousands of masks is much cheaper than training thousands of networks.

| Pruning | Exhaustive QUBO scoring, CPU elapsed | Prior hardware batch | Exact QUBO optimum observed on hardware |
|---|---:|---:|---:|
{chr(10).join(cheap)}

The full prior quantum batches did contain an exact QUBO optimum in both cases. That certifies a successful observed solution of the **approximation** on those instances. It does not establish that the corresponding network minimizes actual retraining loss. The separate timed replay above reports its own outcomes.

## What can be claimed

This experiment can measure whether a learned surrogate plus quantum sampling returns a useful subnetwork in less time than retraining every candidate, and quantify the actual loss gap to exhaustive retraining. A finite-depth QAOA run does not guarantee the global minimum. A QUBO approximation does not guarantee preservation of the true objective's minimizer.

A runtime win against exhaustive retraining, at a stated quality level, would support this **particular pipeline and baseline**. It would not prove universal quantum acceleration for arbitrary optimization problems, nor isolate how much of the benefit comes specifically from the quantum component versus learning the surrogate.

## Reproduce and inspect

- `exhaustive_retraining.py --keep 4` or `--keep 8`: actual exhaustive oracle, with resumable checkpoints.
- `timed_quantum_pipeline.py`: fresh timed pipeline, including real hardware and all 192 fit retrains; no audit-retraining cost is included in the production pipeline.
- `exhaustive_retraining/keep_*/progress.json`: live exhaustive progress and measured times.
- `end_to_end/quantum_keep_*/complete.json`: complete measured pipeline timings and actual loss.
- The earlier full benchmark is preserved as `Full_Benchmark.html`; raw results are unchanged.
'''
    (OUT/'Results.md').write_text(text,encoding='utf-8')
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell('''from pathlib import Path
import json
BASE=Path('.').resolve()
for keep in [8,4]:
    for path in [BASE/'exhaustive_retraining'/f'keep_{keep}'/'progress.json',BASE/'end_to_end'/f'quantum_keep_{keep}'/'complete.json']:
        if path.exists(): print(path.name, json.loads(path.read_text()))
''')],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.write(nb,OUT/'Scaling_Walkthrough.ipynb')
    body,_=HTMLExporter().from_notebook_node(nb); (OUT/'Results.html').write_text(body,encoding='utf-8')
    e={'complete':all_complete,'scope':'exhaustive retraining versus cold quantum pipeline; formula-scoring baseline explicitly distinct'}
    (OUT/'focused_comparison_status.json').write_text(json.dumps(e,indent=2))
    print('Focused report refreshed; complete:',all_complete,flush=True)
    return all_complete

if __name__=='__main__':
    if '--watch' in sys.argv:
        while not build(): time.sleep(60)
    else: build()
