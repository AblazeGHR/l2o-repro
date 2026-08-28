import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src.config
from src.config import SEED
from pennylane import numpy as np
import numpy as onp
import pennylane as qml
import math
from tqdm import tqdm
from utilities import *
import collections

def qaoa(G:Graph, shots:int=1000, n_layers:int=1, const=0, sample_method:str='max', init_gammas=None, init_betas=None):
    '''
    Optimized QAOA for max cut using Adjoint Differentiation
    --------------------------
    G : Graph 

    shots : number of circuit shots

    n_layers : number of QAOA layers

    const : constant in max cut objective function

    sample_method : 'max' return the bitstring with largest cut value

    init_gammas : list of initial gamma values (length must equal n_layers)
    
    init_betas : list of initial beta values (length must equal n_layers)

    Return cut value and solution
    '''
    n_wires = G.n_v
    edges = G.e
    
    if not edges:
        return const, format(0, "0{}b".format(n_wires))[::-1]

    edge_array = onp.array(edges)
    u_indices = edge_array[:, 0].astype(int)
    v_indices = edge_array[:, 1].astype(int)
    weights = edge_array[:, 2]
    
    dev_opt = qml.device('lightning.qubit', wires=n_wires, shots=None)

    def qaoa_layer(gamma, beta):
        for i in range(len(edges)):
            u, v, w = edges[i]
            qml.MultiRZ(gamma * w, wires=[u, v])
            
        for i in range(n_wires):
            qml.RX(2 * beta, wires=i)

    obs = [qml.PauliZ(u) @ qml.PauliZ(v) for u, v, _ in edges]
    coeffs = [0.5 * w for _, _, w in edges]
    H_C = qml.Hamiltonian(coeffs, obs)

    @qml.qnode(dev_opt, diff_method="adjoint")
    def circuit_opt(params):
        for wire in range(n_wires):
            qml.Hadamard(wire)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.expval(H_C)

    if init_gammas is not None and init_betas is not None:
        assert len(init_gammas) == n_layers
        assert len(init_betas) == n_layers
        init_params = np.array([init_gammas, init_betas], requires_grad=True)
    else:
        init_params = np.ones((2, n_layers), requires_grad=True) * 0.01 

    opt = qml.GradientDescentOptimizer()
    params = init_params
    steps = 20
    
    pbar = tqdm(range(steps), desc='Optimize QAOA', leave=False, ascii=False)
    for step in pbar:
        params = opt.step(circuit_opt, params)
        
        if step % 5 == 0 or step == steps - 1:
            cur_val = circuit_opt(params)
            try:
                cur_scalar = cur_val.item() if hasattr(cur_val, 'item') else float(cur_val)
            except Exception:
                cur_scalar = cur_val
            pbar.set_postfix({"expval": f"{cur_scalar:.4f}"})
    
    dev_sample = qml.device('lightning.qubit', wires=n_wires, shots=shots, seed=SEED)
    
    @qml.qnode(dev_sample)
    def circuit_sample(params):
        for wire in range(n_wires):
            qml.Hadamard(wires=wire)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return [qml.sample(qml.PauliZ(i)) for i in range(n_wires)]

    raw_samples_pl = circuit_sample(params)
    raw_samples = onp.array(raw_samples_pl).T
    spin_products = raw_samples[:, u_indices] * raw_samples[:, v_indices]

    obj_values_per_shot = 0.5 * onp.dot(spin_products, weights)
    
    if sample_method == 'max':
        best_idx = onp.argmin(obj_values_per_shot)
        best_obj = obj_values_per_shot[best_idx]
        best_sample = raw_samples[best_idx]

        sol_int = ((best_sample + 1) / 2).astype(int)
        sol = "".join(sol_int.astype(str))
        
        return const - best_obj, sol

    else:
        rows_as_strings = [''.join(row) for row in ((raw_samples + 1) / 2).astype(int).astype(str)]
        counts = collections.Counter(rows_as_strings)
        most_freq_bit_string = counts.most_common(1)[0][0]
        
        sol_array = onp.array([1 if c == '1' else -1 for c in most_freq_bit_string])
        obj = 0.5 * onp.sum(weights * sol_array[u_indices] * sol_array[v_indices])
        
        return const - obj, most_freq_bit_string