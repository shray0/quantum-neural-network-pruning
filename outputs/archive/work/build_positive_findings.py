"""Review saved evidence and produce positive highlights. No training or IBM calls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; ASSETS=OUT/'positive_findings'
ASSETS.mkdir(exist_ok=True)
def read(relative): return json.loads((OUT/relative).read_text())

# All values come from pre-existing run files. These are different, explicitly scoped findings.
hardware=read('constrained_v4/hardware/summary.json')
depths=pd.read_csv(OUT/'constrained_v4/hardware/depth_comparison.csv')
chosen=depths.loc[depths.layers==hardware['chosen_layers']].iloc[0]
fresh=pd.read_csv(OUT/'constrained_v4/fresh_seed_summary.csv')
effects=read('constrained_v4/fresh_seed_effects.json')
uncertainty=read('constrained_v4/hardware/finite_shot_uncertainty.json')
dense8=pd.read_csv(OUT/'hardware_v3/runs/hardware/final_comparison.csv')
dense8_acc=float(dense8.loc[dense8.method=='Dense','test_accuracy'].iloc[0])
dense16=read('scale16_v5/seed_1001/dense.json')
hardware16=read('scale16_v5/hardware/keep_4/summary.json')
local16=pd.read_csv(OUT/'scale16_v5/all_comparisons.csv')
local50=local16.loc[local16.pruning==50]
q16=float(local50.loc[local50.method=='QAOA best32','test_accuracy'].mean())
r16=float(local50.loc[local50.method=='Random best32','test_accuracy'].mean())
audit16=pd.read_csv(OUT/'scale16_v5/all_surrogate_audits.csv')
auditmeans=audit16.loc[audit16.model=='Quadratic'].groupby('pruning')[['r2','mae']].mean()
hardware_gain=100*(chosen.best32_expected_test_accuracy-hardware['uniform_best32_expected_test_accuracy'])
ideal_gain=effects['best32_']['mean_pp']
cases=[{'case':'8 groups; 50% neuron pruning','qubits':8,'scope':'One preselected initialization; frozen 32-shot hardware selection',
        'dense_parameters':12730,'pruned_parameters':6370,'dense_accuracy':dense8_acc,
        'pruned_accuracy':hardware['selected_test_accuracy']},
       {'case':'16 groups; 75% neuron pruning','qubits':16,'scope':'One preselected initialization; first frozen 32-shot hardware selection',
        'dense_parameters':25450,'pruned_parameters':6370,'dense_accuracy':dense16['test_accuracy'],
        'pruned_accuracy':hardware16['first_frozen_quantum_test_accuracy']}]
for c in cases:
    c['parameter_reduction_percent']=100*(1-c['pruned_parameters']/c['dense_parameters'])
    c['accuracy_drop_pp']=100*(c['dense_accuracy']-c['pruned_accuracy'])
pd.DataFrame(cases).to_csv(ASSETS/'hardware_compression_examples.csv',index=False)

gains=[{'execution':'Real IBM hardware, 8 groups','quantum_accuracy':float(chosen.best32_expected_test_accuracy),
        'random_accuracy':hardware['uniform_best32_expected_test_accuracy'],'gain_pp':hardware_gain,
        'scope':'Expected best-of-32 accuracy inferred from one actual 4096-shot measurement batch; invalid outcomes consume the raw-shot budget.'},
       {'execution':'Ideal simulation, 8 groups','quantum_accuracy':float(fresh.qaoa_best32_expected_test_accuracy.mean()),
        'random_accuracy':float(fresh.random_best32_expected_test_accuracy.mean()),'gain_pp':ideal_gain,
        'scope':'Mean expected best-of-32 accuracy across five fresh initializations, all positive gains.'},
       {'execution':'Ideal simulation, 16 groups, 50% pruning','quantum_accuracy':q16,'random_accuracy':r16,
        'gain_pp':100*(q16-r16),'scope':'Mean selected-mask accuracy across three initializations; one frozen 32-candidate draw per method.'}]
pd.DataFrame(gains).to_csv(ASSETS/'positive_random_comparisons.csv',index=False)

budget_rows=[]
for keep in [8,4]:
    result=read(f'scale16_v5/end_to_end/quantum_keep_{keep}/complete.json')
    budget_rows.append({'pruning_percent':100*(1-keep/16),'possible_masks':12870 if keep==8 else 1820,
        'fit_retrainings':result['fit_retrains'],'final_retraining':1,'observed_training_runs':result['fit_retrains']+1,
        'pipeline_wall_seconds':result['end_to_end_wall_seconds'],'qpu_seconds':result['qpu_reported_seconds'],
        'test_accuracy':result['test_accuracy'],'job_id':result['job_id']})
pd.DataFrame(budget_rows).to_csv(ASSETS/'evaluation_budget_and_pipeline_times.csv',index=False)

# Verify the saved actual hardware evidence without opening credentials or contacting IBM.
evidence=[]
paths=['hardware_v3/runs/hardware/ledger.json','constrained_v4/hardware_kingston_partial/ledger.json',
       'constrained_v4/hardware/ledger.json','scale16_v5/hardware/ledger.json','scale16_v5/end_to_end/hardware_ledger.json']
for relative in paths:
    ledger=read(relative)
    assert not ledger.get('test_mode',False)
    for job in ledger['jobs']:
        if job['state']=='complete':
            evidence.append({'source':relative,'backend':ledger['backend'],'job_id':job['job_id'],
                'stage':job['tag'],'shots':job['shots'],'reported_qpu_seconds':job['reported_usage_seconds']})
pd.DataFrame(evidence).to_csv(ASSETS/'completed_ibm_jobs.csv',index=False)

# Detect, retain and flag interrupted old timing intervals; never report them as speedup evidence.
timing_flags=[]
for keep in [4,8]:
    path=OUT/'scale16_v5/exhaustive_retraining'/f'keep_{keep}'/'all_validation_losses.csv'
    if path.exists():
        d=pd.read_csv(path); v=d.retraining_and_validation_seconds
        bad=v>max(30,100*v.median())
        timing_flags.append({'keep':keep,'recorded_masks':len(d),'median_mask_seconds':float(v.median()),
            'outlier_count':int(bad.sum()),'largest_interval_seconds':float(v.max()),
            'use_for_speedup_ratio':not bool(bad.any()),
            'interpretation':'Near-identical long intervals in both workers indicate a shared host interruption; raw timing is not a clean compute benchmark.' if bad.any() else 'No large interval detected.'})
(ASSETS/'timing_quality_review.json').write_text(json.dumps(timing_flags,indent=2))

fig,axes=plt.subplots(1,2,figsize=(10.5,4))
for ax,c in zip(axes,cases):
    values=[100*c['dense_accuracy'],100*c['pruned_accuracy']]
    ax.bar([0,1],values,color=['#94a3b8','#2563eb'])
    for i,v in enumerate(values): ax.text(i,v+1,f'{v:.2f}%',ha='center')
    ax.set(xticks=[0,1],xticklabels=['Dense network','QPU-selected subnetwork'],ylim=(0,100),
           ylabel='Test accuracy (%)',title=c['case'])
    ax.text(.5,.08,f"{c['dense_parameters']:,} → {c['pruned_parameters']:,} parameters\nAccuracy drop: {c['accuracy_drop_pp']:.2f} percentage points",
            transform=ax.transAxes,ha='center',fontsize=9,color='white',bbox={'facecolor':'#1e3a8a','edgecolor':'none','pad':7})
fig.suptitle('Preselected hardware examples: smaller networks retain accuracy')
fig.tight_layout(); fig.savefig(ASSETS/'hardware_compression.png',dpi=170); plt.close(fig)

fig,ax=plt.subplots(figsize=(8.5,4.4))
values=[g['gain_pp'] for g in gains]
ax.bar(np.arange(3),values,color=['#2563eb','#8b5cf6','#a78bfa'])
for i,v in enumerate(values): ax.text(i,v+.012,f'+{v:.3f} pp',ha='center',fontweight='bold')
ax.set(xticks=[0,1,2],xticklabels=['8 groups\nIBM hardware','8 groups\nIdeal: 5 initializations','16 groups, 50% pruning\nIdeal: 3 initializations'],
       ylabel='Accuracy gain over random (pp)',ylim=(0,max(values)*1.3),
       title='Selected positive comparisons against random pruning')
ax.axhline(0,color='black'); fig.tight_layout(); fig.savefig(ASSETS/'positive_random_gains.png',dpi=170); plt.close(fig)

claim='Qismet identifies high-performing subnetworks using far fewer retraining evaluations than exhaustive search, and improved selection accuracy over uniform random pruning in an eight-group IBM hardware demonstration.'
table='| Hardware example | Dense → pruned parameters | Dense → pruned test accuracy | Accuracy drop |\n|---|---:|---:|---:|\n'
for c in cases: table+=f"| {c['case']} | {c['dense_parameters']:,} → {c['pruned_parameters']:,} | {c['dense_accuracy']*100:.2f}% → {c['pruned_accuracy']*100:.2f}% | {c['accuracy_drop_pp']:.2f} pp |\n"
randomtable='| Execution and setting | Qismet | Random baseline | Improvement |\n|---|---:|---:|---:|\n'
for g in gains: randomtable+=f"| {g['execution']} | {g['quantum_accuracy']*100:.3f}% | {g['random_accuracy']*100:.3f}% | +{g['gain_pp']:.3f} pp |\n"
budgettable='| Hidden-neuron pruning | All possible masks | Observed Qismet training runs | Share of full candidate count | Full pipeline wall time | QPU usage within that time |\n|---|---:|---:|---:|---:|---:|\n'
for r in budget_rows: budgettable+=f"| {r['pruning_percent']:.0f}% | {r['possible_masks']:,} | {r['observed_training_runs']} | {100*r['observed_training_runs']/r['possible_masks']:.2f}% | {r['pipeline_wall_seconds']:.1f} s | {r['qpu_seconds']:g} s |\n"
hwci=uncertainty['selected_best32_accuracy_gain_pp_interval_95']
text=f'''# Qismet — positive findings and implications

> **{claim}**

This section highlights positive observations from the saved experiments. Each result retains its setting and metric. It complements the separate report on limitations; it is not a pooled claim that every run improved. No retraining or IBM submission was performed to prepare this review.

## 1. Real hardware found compact subnetworks with strong accuracy retention

{table}

Both masks were selected before their final test evaluation. Surviving weights were restored to the original random initialization and retrained with the same 20-epoch budget. These are individual, preselected hardware examples rather than averages over all masks.

**Implication:** the end-to-end process can turn a measured quantum bitstring into a physically smaller, useful classifier. The 16-group example uses approximately **four times fewer parameters** while remaining within **1.04 percentage points** of the dense model. The eight-group example uses approximately **half the parameters**, within **0.52 points**.

![Hardware compression examples](positive_findings/hardware_compression.png)

## 2. The eight-group hardware demonstration improved on random picking

{randomtable}

For the eight-group hardware row, the metric is expected accuracy after choosing the lowest-QUBO-cost mask from 32 raw proposals. Invalid quantum outcomes consume the budget. The expectation is estimated from one actual 4,096-shot measurement batch and the measured retrained performance of its candidate masks. Circuit depth was chosen by QUBO score before test evaluation. A finite-shot bootstrap that repeats depth selection gives a sampling-only 95% interval of **[{hwci[0]:.3f}, {hwci[1]:.3f}] pp** for the gain. This interval does not include device drift or dataset uncertainty.

The eight-group ideal row averages expected best-of-32 accuracy over five fresh initializations; **all five gains were positive**. The 16-group ideal row is the selected-network average for the specified 50% pruning setting over three initializations. Simulation and physical hardware results are deliberately labeled separately.

**Implication:** the learned cost and quantum proposal distribution can guide selection toward useful subnetworks rather than relying only on uninformed random selection. The hardware evidence supports this specific eight-group demonstration.

![Positive random comparisons](positive_findings/positive_random_gains.png)

## 3. Qismet used a small fraction of exhaustive retraining's candidate evaluations

{budgettable}

The observed production runs used **192 subnetworks to fit the loss model plus one final selected-network retraining**. The table compares that count with the number of candidates a full exhaustive-retraining procedure would evaluate. For 50% pruning, Qismet used about **1.5% of the full candidate count**—approximately **67 times fewer network-training evaluations**. At 75%, it used about **10.6%**, approximately **9.4 times fewer**.

The wall times above are measured cold replays: fresh fit-label training, regression, CPU simulation-based angle optimization, real IBM connection and circuit preparation, queueing, measurement processing, and final selected-network retraining/testing. Common data/model loading and feasible-mask feature preparation are outside the timed interval, as are earlier research and audit runs. These timed replays are separate from the illustrative 32-shot masks above.

**Implication:** representing retraining performance with a learned objective can substantially reduce how often the neural network must be trained during pruning. This is an evaluation-budget advantage over exhaustive retraining. The budget comparison does not assert equal global-optimum guarantees or a universal runtime speedup; the benefit of learning the surrogate is not attributed exclusively to quantum hardware.

## 4. The retraining-loss approximation captured useful structure

On the 16-group study, with 192 fit masks and 64 disjoint audit masks per case, mean held-out R² across three initializations was **{auditmeans.loc[50,'r2']:.3f}** at 50% pruning and **{auditmeans.loc[75,'r2']:.3f}** at 75%. Mean absolute validation-loss errors were **{auditmeans.loc[50,'mae']:.4f}** and **{auditmeans.loc[75,'mae']:.4f}**, respectively.

In the eight-group follow-up, the mean held-out R² across five fresh initializations was **{fresh.audit_r2.mean():.3f}**, with mean absolute loss error **{fresh.audit_mae.mean():.4f}**. Ideal QAOA assigned an average **{fresh.top10_probability.mean()*100:.2f}%** probability to the best-scoring 10% of masks, compared with 10% under uniform sampling.

**Implication:** the approach has an empirically useful link between measured retrainability, a quadratic objective and quantum sampling. It is a working bridge between machine-learning experiments and quantum optimization.

## Presentation-ready wording

“We implemented an end-to-end pruning pipeline that learns a quadratic approximation of retraining performance and uses QAOA on IBM quantum hardware to propose subnetworks. Preselected hardware examples retained strong accuracy with roughly 50–75% fewer parameters. In our eight-group hardware demonstration, the quantum-assisted sampler improved expected best-of-32 accuracy over uniform random selection. In the 16-group timed pipeline, only 192 fit subnetworks and one final retraining were needed, compared with 12,870 candidates in exhaustive retraining.”

Use **‘more retraining-evaluation-efficient than exhaustive retraining’** for the efficiency statement, and **‘better than random in the eight-group hardware demonstration’** for the selection statement. ‘The globally best true subnetwork’ and ‘universal quantum speedup’ are not claims made by this positive section.

## Traceable saved evidence

- [Eight-group hardware comparison](constrained_v4/hardware/depth_comparison.csv), [selected result](constrained_v4/hardware/summary.json), [finite-shot uncertainty](constrained_v4/hardware/finite_shot_uncertainty.json).
- [Five fresh ideal initializations](constrained_v4/fresh_seed_summary.csv).
- [16-group hardware compression example](scale16_v5/hardware/keep_4/summary.json), [dense reference](scale16_v5/seed_1001/dense.json).
- [16-group local comparison](scale16_v5/all_comparisons.csv), [held-out surrogate audits](scale16_v5/all_surrogate_audits.csv).
- [Timed 50% pipeline](scale16_v5/end_to_end/quantum_keep_8/complete.json), [timed 75% pipeline](scale16_v5/end_to_end/quantum_keep_4/complete.json).
- [Completed IBM job inventory](positive_findings/completed_ibm_jobs.csv), [positive comparison metrics](positive_findings/positive_random_comparisons.csv), [evaluation budgets](positive_findings/evaluation_budget_and_pipeline_times.csv).

Review note: the saved exhaustive-retraining timing contains a shared long host interruption. That timing ratio is not used in this positive section. The original records are preserved, with the timing-quality review in `positive_findings/timing_quality_review.json`.
'''
(OUT/'Positive_Findings.md').write_text(text,encoding='utf-8')
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text)])
body,_=HTMLExporter().from_notebook_node(nb)
(OUT/'Positive_Findings.html').write_text(body,encoding='utf-8')
(ASSETS/'claim_scope.json').write_text(json.dumps({'headline':claim,'data_review_only':True,'new_model_training_runs':0,'new_ibm_jobs':0,
    'hardware_random_comparison_gain_pp':hardware_gain,'ideal_8group_best32_gain_pp':ideal_gain,
    'ideal_16group_50percent_gain_pp':100*(q16-r16),'hardware_examples':cases,
    'scope':'Selected positive observations, not an all-run superiority assertion. Raw evidence preserved.'},indent=2))
print('Created Positive_Findings.md and Positive_Findings.html using saved results only.')
print('Hardware random gain pp:',hardware_gain,'Ideal 8-group best32 gain pp:',ideal_gain)
print('Completed IBM jobs reviewed:',len(evidence),'Reported QPU seconds:',sum(r['reported_qpu_seconds'] for r in evidence))
