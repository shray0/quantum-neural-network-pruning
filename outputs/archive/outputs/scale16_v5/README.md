# 16-group scaling study

**Presentation focus updated:** `Results.html` and `Scaling_Walkthrough.ipynb` now compare Qismet + QPU with exhaustive retraining and explicitly distinguish it from exhaustive QUBO scoring. Live measured end-to-end benchmarks are running; incomplete results and remaining-time estimates are labeled. The earlier complete multi-method presentation is preserved as `Full_Benchmark.html` and `Full_Benchmark.ipynb`.

`exhaustive_retraining.py` actually retrains all 1,820 or 12,870 valid masks from the original weights. `timed_quantum_pipeline.py` measures a fresh 192-label Qismet pipeline, CPU angle optimization, a real QPU job and final retraining. No earlier fit-label timings are substituted into that fresh end-to-end measurement. Live checkpoints are under `exhaustive_retraining/` and `end_to_end/`. The report refreshes automatically.

Start with **Results.html** or **Scaling_Walkthrough.ipynb** once generation completes. The full fixed experiment runs three initializations at 50% and 75% pruning, using the existing Fashion-MNIST split and a 784→32→10 model. Groups contain two hidden neurons.

The study compares ideal QAOA with random search, greedy swaps, simulated annealing, exact QUBO and linear selection. Greedy and annealing timings in the final report score coefficients directly, without access to an exhaustive lookup table. Linear, full quadratic and sparse ring surrogates are audited separately.

Actual 16-qubit circuits were also run on IBM Fez for the first preselected initialization, including an initial-state control and one QAOA layer at both pruning budgets. Four jobs used 12 reported QPU seconds. Masks are decoded using all 16 logical measurement bits. Independent numerical checks confirm that the Qiskit circuits reproduce the subspace simulation and conserve the desired group count in the ideal case.

Hardware accuracy compares 128 reconstructed 32-shot selections from a single 4,096-shot pool against 128 uniform-random 32-candidate selections. They are not independent hardware repeats. Invalid shots consume the quantum candidate budget; failed batches are retained. Winning states are frozen before final retraining and test evaluation.

## Main source files

- `experiment.py`: fixed protocol, training, original-weight rewind, labels, surrogate audit, ideal QAOA and selected-mask evaluation.
- `hardware.py`: actual 16-qubit state preparation, cost/mixer circuit and IBM execution with inherited usage limits.
- `direct_classical.py`: greedy, annealing and full enumeration from QUBO coefficients.
- `evaluate_hardware.py`: raw-shot comparisons and final network evaluations.

The original credential file remains in the sibling `hardware_v3` folder; it is not copied into this experiment. All prior usage, including the cancelled job's conservative reservation, is subtracted before allocating this experiment's hardware allowance. No new paid instance is selected.

## Interpretation

The larger mask space does not require classical heuristics to enumerate it. Hardware noise and routing cost also grow with circuit size. The report retains every outcome rather than selecting only favorable initializations or individual shots. No quantum speed advantage is assumed or promised.
