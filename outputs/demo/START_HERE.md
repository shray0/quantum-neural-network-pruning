# Neural Network Pruning with Quantum Algorithms

Open **Demo.html** for the saved walkthrough, or **Demo.ipynb** to run it. **pruning.py** contains the implementation; **assets/** contains the data and recorded IBM hardware evidence.

Install `requirements.txt` in a Python environment and select it as the notebook kernel. Run the cells in order. No IBM account is needed and no hardware jobs are submitted. Set `RUN_LIVE_RETRAINING=True` only to run the optional CPU retraining step.

The demo retains eight of sixteen neuron groups, reducing 25,450 parameters to 12,730 with 82.73% test accuracy versus 82.75% before pruning. The separate eight-group comparison is labeled independently.

Earlier experiments and the 100-qubit hardware study are in [the archive](../archive/).
