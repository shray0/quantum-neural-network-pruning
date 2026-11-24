"""Package existing evidence into the demo. No model training or IBM calls."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; DEMO=OUT/'demo'; ASSETS=DEMO/'assets'
ASSETS.mkdir(exist_ok=True)
def read(path): return json.loads((OUT/path).read_text())
base=OUT/'scale16_v5'
run=read('scale16_v5/end_to_end/quantum_keep_8/complete.json')
dense=read('scale16_v5/seed_1001/dense.json')
result=read('scale16_v5/end_to_end/cold_keep_8/result.json')
assert run['selected_state']==14926 and result['job_id']==run['job_id']
assert not result['test_mode']
with np.load(OUT/'fashion_v2/dataset_subset.npz') as d: arrays={k:d[k] for k in d.files}
model=np.load(base/'end_to_end/quantum_keep_8/model.npz')
arrays.update({k:model[k] for k in ['linear','pairs','constant','theta','scale']})
arrays['feasible_states']=model['ids']; arrays['feasible_scores']=model['predictions']; arrays['ideal_probabilities']=model['probabilities']
labels=pd.read_csv(base/'end_to_end/quantum_keep_8/fit_labels.csv')
arrays['fit_states']=labels.state.to_numpy(dtype=np.int64); arrays['fit_loss']=labels.val_loss.to_numpy()
audit=pd.read_csv(base/'seed_1001/keep_8/labels.csv').query("split=='audit'")
arrays['audit_states']=audit.state.to_numpy(dtype=np.int64); arrays['audit_loss']=audit.val_loss.to_numpy()
assert not set(arrays['fit_states'])&set(arrays['audit_states'])
np.savez_compressed(ASSETS/'data.npz',**arrays)
initial=torch.load(base/'seed_1001/original_initialization.pt',weights_only=True)
selected=torch.load(base/'seed_1001/keep_8/Exact_QUBO.pt',weights_only=True)
assert int(np.array(selected['mask'])@(2**np.arange(16)))==run['selected_state']
torch.save({'initial':initial,'selected':selected['state_dict']},ASSETS/'weights.pt')
for old,new in [('logical.qpy','logical_circuit.qpy'),('transpiled.qpy','physical_circuit.qpy')]:
    (ASSETS/new).write_bytes((base/'end_to_end/cold_keep_8'/old).read_bytes())
random=read('constrained_v4/hardware/summary.json')
depth=pd.read_csv(OUT/'constrained_v4/hardware/depth_comparison.csv')
row=depth.loc[depth.layers==random['chosen_layers']].iloc[0]
evidence={'run':run,'dense':dense,'hardware':result,
          'job_metrics':read('scale16_v5/end_to_end/cold_keep_8/job_metrics.json'),
          'circuit_metadata':read('scale16_v5/end_to_end/cold_keep_8/circuit_metadata.json'),
          'class_names':['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'],
          'separate_8group_random_comparison':{'logical_qubits':8,'quantum_expected_best32_accuracy':float(row.best32_expected_test_accuracy),
             'random_expected_best32_accuracy':random['uniform_best32_expected_test_accuracy'],
             'scope':'Separate eight-group hardware experiment, expected best-of-32 accuracy inferred from measured counts; invalid quantum shots count toward the proposal budget.'},
          'provenance':'Copied from the completed saved runs. No new training or hardware execution during packaging.',
          'mode':'Recorded IBM hardware; optional live CPU retraining is disabled by default.'}
(ASSETS/'recorded_run.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')

md=nbformat.v4.new_markdown_cell; code=nbformat.v4.new_code_cell
cells=[md('''# Qismet — the demo

**One example, end to end:** Fashion-MNIST → pruning choices → learned QUBO → QAOA circuit → real IBM measurements → smaller working network.

This walkthrough uses a completed **16-qubit IBM Fez job**. It is a recorded hardware execution, not a new submission. Cached data lets the presentation run reliably. The optional live CPU retraining cell is off by default.

The only source file you need to read is **qismet.py**. The assets folder contains automatically loaded data and evidence.
'''),code('''import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import qismet as q
arrays, weights, evidence = q.load()
run = evidence['run']
'''),md('''## 1. What does the network do?

An image contains 784 grayscale pixel values. The network predicts one of ten clothing classes.
'''),code('''image_index = 0  # Change this to show another dataset image.
plt.figure(figsize=(3,3))
plt.imshow(arrays['test_images'][image_index], cmap='gray')
plt.title(evidence['class_names'][int(arrays['test_labels'][image_index])])
plt.axis('off'); plt.show()
print('Network: 784 pixels → 32 hidden neurons → 10 clothing-class scores')
'''),md('''## 2. Which neurons can we remove?

The 32 hidden neurons are grouped into 16 pairs. A bit of 1 retains a pair; 0 removes it. Keeping eight pairs leaves 16 hidden neurons. There are 12,870 valid masks.
'''),code('''mask = q.mask_from_state(run['selected_state'])
print('Selected group mask:', mask.tolist())
print('Qiskit printed bitstring:', format(run['selected_state'], '016b'))
print('Valid possibilities:', math.comb(q.GROUPS, q.KEEP))
display(q.show_groups())
plt.close('all')
'''),md(r'''## 3. Approximate the expensive score

The expensive score is validation loss **after restoring the original weights and retraining**. The learned approximation is

$\hat g(m)=c+\sum_i a_i m_i+\sum_{i<j}b_{ij}m_i m_j$.

This run used 192 fit masks. Another 64 disjoint masks audited the approximation during development. The following points are held-out measured losses, not new retraining runs.
'''),code('''prediction = q.qubo_scores(arrays['audit_states'])
fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(arrays['audit_loss'], prediction, s=25)
limits=[min(arrays['audit_loss'].min(),prediction.min()),max(arrays['audit_loss'].max(),prediction.max())]
ax.plot(limits,limits,'k--',label='Exact agreement')
ax.set(xlabel='Measured retrained validation loss',ylabel='QUBO predicted loss',title='64 held-out masks')
ax.legend(); plt.show()
print('192 fitted examples out of', math.comb(16,8), 'possible masks.')
'''),md('''## 4. Build the actual quantum circuit

The initial Bell pairs prepare valid masks. Cost phases use the QUBO coefficients. Paired RXX/RYY gates form an XY mixer that preserves eight retained groups in an ideal circuit. One layer uses two angles; the saved angles were optimized on a CPU simulator.

The functions `refit_from_saved_labels`, `optimize_angles`, and `build_circuit` in qismet.py show those three steps. Nothing in this notebook submits another IBM job.
'''),code('''circuit=q.build_circuit()
print('Logical qubits:', circuit.num_qubits)
print('Saved angles [gamma, beta]:', arrays['theta'])
fig=circuit.draw(output='mpl', fold=70)
display(fig); plt.close(fig)
'''),md('''## 5. What came back from IBM?

Each measurement returns one pruning mask. We score feasible observed masks with the QUBO and choose the lowest-scoring one. The most frequent mask need not be the chosen one. Device errors can produce the wrong number of retained groups.
'''),code('''display(pd.DataFrame([{
    'Backend':evidence['hardware']['backend'],
    'IBM job ID':evidence['hardware']['job_id'],
    'Shots':sum(evidence['hardware']['counts'].values()),
    'Reported QPU seconds':run['qpu_reported_seconds'],
    'Whole timed pipeline seconds':round(run['end_to_end_wall_seconds'],1)
}]))
candidates=q.recorded_candidates()
display(candidates.head(8))
assert int(candidates.iloc[0]['state'])==run['selected_state']
print('The selected state was actually observed on the QPU:',format(run['selected_state'],'016b'))
print('Physical-circuit gate counts:',evidence['circuit_metadata']['circuits']['transpiled']['gate_counts'])
'''),md('''## 6. Restore the original initialization, then retrain

Pruning slices out the surviving neurons. Their starting weights are copied from the original untrained network. A fresh optimizer retrains that smaller network.

The first cell verifies initialization restoration without training. The second cell is the optional live demo. Set its flag to True only when you want to retrain on your CPU.
'''),code('''rewound=q.rewind()
neurons=np.flatnonzero(np.repeat(mask,2))
import torch
assert torch.equal(rewound.fc1.weight,weights['initial']['fc1.weight'][neurons])
print('Verified: surviving input weights equal their original initialization.')
'''),code('''RUN_LIVE_RETRAINING=False
if RUN_LIVE_RETRAINING:
    live_model, live_metrics=q.retrain_selected(epochs=20)
    print('LIVE CPU retraining result:',live_metrics)
else:
    print('Using the already-retrained saved model. No new training performed.')
'''),md('''## 7. Show the smaller network working

This scorecard is the saved 16-group timed run. The accuracy comparison below can be verified by forward evaluation of its saved weights.
'''),code('''selected=q.saved_model()
metrics=q.evaluate(selected)
assert abs(metrics['accuracy']-run['test_accuracy'])<1e-6
display(pd.DataFrame([
    {'Model':'Dense','Parameters':25450,'Test accuracy (%)':100*evidence['dense']['test_accuracy']},
    {'Model':'QPU-selected subnetwork','Parameters':sum(p.numel() for p in selected.parameters()),'Test accuracy (%)':100*metrics['accuracy']}
]))
with torch.no_grad():
    x=torch.tensor(arrays['test_images'][image_index].reshape(1,784)/255.,dtype=torch.float32)
    predicted=int(selected(x).argmax(1))
print('Prediction for the example:',evidence['class_names'][predicted])
print('Training evaluations: 192 fit masks + 1 final retraining, versus 12,870 candidates in exhaustive retraining.')
print('This is an evaluation-budget comparison; the QUBO approximates actual retraining loss.')
'''),md('''## Separate result: the eight-group comparison with random pruning

This is a different saved experiment. Its metric is expected accuracy after choosing the best QUBO score from 32 raw proposals, estimated from recorded hardware measurements. It is not the accuracy of the 16-group example above.
'''),code('''r=evidence['separate_8group_random_comparison']
display(pd.DataFrame([
    {'Method':'Eight-group hardware QAOA','Expected best-of-32 accuracy (%)':100*r['quantum_expected_best32_accuracy']},
    {'Method':'Uniform random pruning','Expected best-of-32 accuracy (%)':100*r['random_expected_best32_accuracy']}
]))
print('Hardware improvement:',round(100*(r['quantum_expected_best32_accuracy']-r['random_expected_best32_accuracy']),3),'percentage points')
'''),md('''## What to say at the end

“Qismet links measured retraining performance to a quadratic objective and a real quantum circuit. This saved hardware example retained nearly the same accuracy with half the parameters, using a small fraction of exhaustive retraining's candidate evaluations. A separate eight-group hardware demonstration improved selection accuracy over uniform random pruning.”

The recorded hardware portion makes the demo reliable; the optional CPU cell demonstrates rewinding and retraining live. Older experiments and full evidence are archived outside this demo folder.
''')]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3 (Qismet)','language':'python'}})
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(DEMO)}}).execute()
nbformat.write(nb,DEMO/'Demo.ipynb')
body,_=HTMLExporter().from_notebook_node(nb); (DEMO/'Demo.html').write_text(body,encoding='utf-8')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ASSETS.iterdir() if p.is_file()}
print('Demo prepared and executed using inference and saved evidence only. No training or IBM calls.')
print('Asset integrity hashes:',json.dumps(manifest))
