"""Neural Network Pruning with Quantum Algorithms demo: one supported Fashion-MNIST network, one saved IBM hardware run.

Importing this file never trains a model or contacts IBM. The live training helper
runs only when explicitly called. All required data lives in assets/.
"""
from pathlib import Path
from functools import lru_cache
from itertools import combinations
import json
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from qiskit import QuantumCircuit, qpy

ASSETS = Path(__file__).resolve().parent / 'assets'
GROUPS = 16
NEURONS_PER_GROUP = 2
KEEP = 8
SEED = 1001
PAIRS = list(combinations(range(GROUPS), 2))
MIXER_EDGES = [(i, i+1) for i in range(0, 16, 2)] + [(i, (i+1) % 16) for i in range(1, 16, 2)]


@lru_cache(maxsize=1)
def load():
    with np.load(ASSETS / 'data.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    weights = torch.load(ASSETS / 'weights.pt', map_location='cpu', weights_only=True)
    evidence = json.loads((ASSETS / 'recorded_run.json').read_text(encoding='utf-8'))
    return arrays, weights, evidence


class Network(nn.Module):
    """784 pixels → hidden ReLU neurons → ten clothing-class scores."""
    def __init__(self, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def mask_from_state(state):
    """Group i is bit i; Qiskit prints the highest-index bit on the left."""
    if not 0 <= int(state) < 2**GROUPS:
        raise ValueError('Expected a 16-bit mask integer.')
    return ((int(state) >> np.arange(GROUPS)) & 1).astype(int)


def rewind(state=None):
    """Keep the requested neurons and restore their ORIGINAL initialization."""
    _, weights, evidence = load()
    state = evidence['run']['selected_state'] if state is None else int(state)
    neurons = np.flatnonzero(np.repeat(mask_from_state(state), NEURONS_PER_GROUP))
    if not len(neurons):
        raise ValueError('The mask must retain at least one neuron.')
    indices = torch.tensor(neurons)
    initial = weights['initial']
    model = Network(len(neurons))
    model.load_state_dict({
        'fc1.weight': initial['fc1.weight'][indices],
        'fc1.bias': initial['fc1.bias'][indices],
        'fc2.weight': initial['fc2.weight'][:, indices],
        'fc2.bias': initial['fc2.bias'],
    })
    return model


def saved_model():
    """Load the already-retrained selected subnetwork; no training happens here."""
    _, weights, evidence = load()
    hidden = int(mask_from_state(evidence['run']['selected_state']).sum()) * 2
    model = Network(hidden)
    model.load_state_dict(weights['selected'])
    return model.eval()


def split(name):
    arrays, _, _ = load()
    if name not in ('train', 'val', 'test'):
        raise ValueError('Use train, val, or test.')
    x = torch.tensor(arrays[name+'_images'].reshape(-1, 784) / 255., dtype=torch.float32)
    y = torch.tensor(arrays[name+'_labels'], dtype=torch.long)
    return x, y


def evaluate(model, name='test'):
    x, y = split(name)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return {'loss': float(nn.functional.cross_entropy(logits, y)),
                'accuracy': float((logits.argmax(1) == y).float().mean())}


def retrain_selected(epochs=20):
    """Optional LIVE demo: rewind then retrain. Never called automatically."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = rewind()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(SEED + 100)
    x, y = split('train')
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    return model, evaluate(model)


def features(masks):
    masks = np.atleast_2d(masks)
    return np.column_stack([masks] + [masks[:, i]*masks[:, j] for i, j in PAIRS])


def qubo_scores(states):
    """Cheap predicted loss, not neural-network retraining."""
    arrays, _, _ = load()
    masks = np.array([mask_from_state(s) for s in np.atleast_1d(states)])
    coefficients = np.r_[arrays['linear'], arrays['pairs']]
    return float(arrays['constant']) + features(masks) @ coefficients


def refit_from_saved_labels():
    """Refit the QUBO using cached measurements; does not retrain neural networks."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    arrays, _, _ = load()
    masks = np.array([mask_from_state(s) for s in arrays['fit_states']])
    model = RidgeCV(alphas=np.logspace(-4, 3, 15),
                    cv=KFold(5, shuffle=True, random_state=SEED), scoring='neg_mean_squared_error')
    model.fit(features(masks), arrays['fit_loss'])
    return model


def optimize_angles(restarts=3, max_iterations=100):
    """CPU optimization of the one-layer QAOA state; Qiskit builds its actual circuit."""
    from scipy.optimize import minimize
    arrays, _, _ = load()
    states = arrays['feasible_states']
    masks = np.array([mask_from_state(s) for s in states])
    lookup = {int(s): i for i, s in enumerate(states)}
    swaps = []
    for i, j in MIXER_EDGES:
        left = np.flatnonzero((masks[:, i] == 0) & (masks[:, j] == 1))
        right = np.array([lookup[int(states[r]) ^ (1 << i) ^ (1 << j)] for r in left])
        swaps.append((left, right))
    initial = np.all(masks[:, ::2] + masks[:, 1::2] == 1, axis=1).astype(complex) / 16
    cost = arrays['feasible_scores']
    normalized = (cost - cost.mean()) / float(arrays['scale'])

    def objective(theta):
        state = initial * np.exp(-1j * theta[0] * normalized)
        c, s = np.cos(theta[1]), -1j * np.sin(theta[1])
        for left, right in swaps:
            a, b = state[left].copy(), state[right].copy()
            state[left] = c*a + s*b
            state[right] = s*a + c*b
        return float(np.abs(state)**2 @ normalized)

    runs = []
    for restart in range(restarts):
        theta = np.random.default_rng(SEED + KEEP*100 + 10 + restart).uniform(-1, 1, 2)
        runs.append(minimize(objective, theta, method='BFGS',
                             options={'maxiter': max_iterations, 'gtol': 1e-5}))
    return min(runs, key=lambda r: r.fun)


def build_circuit(theta=None):
    """The actual 16-qubit, one-layer, constraint-preserving QAOA circuit."""
    arrays, _, _ = load()
    theta = arrays['theta'] if theta is None else np.asarray(theta)
    b = np.zeros((GROUPS, GROUPS))
    b[np.triu_indices(GROUPS, 1)] = arrays['pairs']
    h = -arrays['linear']/2 - (b+b.T).sum(1)/4
    j = b/4
    scale = float(arrays['scale'])
    circuit = QuantumCircuit(GROUPS, name='quantum_pruning_demo')
    for i in range(0, GROUPS, 2):
        circuit.h(i)
        circuit.x(i+1)
        circuit.cx(i, i+1)
    circuit.barrier(label='valid initial masks')
    for i in range(GROUPS):
        circuit.rz(2*theta[0]*h[i]/scale, i)
    for i, k in PAIRS:
        circuit.rzz(2*theta[0]*j[i, k]/scale, i, k)
    circuit.barrier(label='predicted-loss phase')
    for i, k in MIXER_EDGES:
        circuit.rxx(theta[1], i, k)
        circuit.ryy(theta[1], i, k)
    circuit.barrier(label='size-preserving mixer')
    return circuit


def recorded_candidates():
    """Real saved IBM counts, scored by QUBO; this never submits a new job."""
    _, _, evidence = load()
    rows = []
    for bits, count in evidence['hardware']['counts'].items():
        state = int(bits, 2)
        if state.bit_count() == KEEP:
            rows.append({'mask': bits, 'state': state, 'shots': count,
                         'predicted_loss': float(qubo_scores([state])[0])})
    return pd.DataFrame(rows).sort_values('predicted_loss').reset_index(drop=True)


def recorded_circuit(physical=False):
    name = 'physical_circuit.qpy' if physical else 'logical_circuit.qpy'
    with (ASSETS / name).open('rb') as stream:
        return qpy.load(stream)[0]


def show_groups(state=None):
    import matplotlib.pyplot as plt
    _, _, evidence = load()
    state = evidence['run']['selected_state'] if state is None else state
    mask = mask_from_state(state)
    fig, ax = plt.subplots(figsize=(10, 2.4))
    for group, retained in enumerate(mask):
        color = '#2563eb' if retained else '#d1d5db'
        ax.scatter([group, group], [.35, -.35], s=240, color=color)
        ax.text(group, -.85, str(group), ha='center')
    ax.set(xlim=(-1, 16), ylim=(-1.1, 1), title='Each pair is a group: blue stays, gray is removed')
    ax.axis('off')
    return fig
