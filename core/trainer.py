import os
import json
import csv
import pennylane as qml
import torch
import pennylane.numpy as np
import multiprocessing as mp
import time
import yaml

from datetime import datetime
from scipy.optimize import minimize, dual_annealing, differential_evolution, basinhopping
from numpy import inf
from tqdm import tqdm
from .utils.arguments import optarguments, circuitarguments, otherarguments
from .circuit import Circuit

pi = np.pi

# (GPT) QFI calculations
def _symmetrize(mat):
    return 0.5 * (mat + mat.conj().T)

# (GPT) QFI calculations
def _five_point_derivative(rho_func, theta, h, w):
    # 5-point centered finite difference: O(h^4)
    return (-rho_func(theta + 2*h, w) + 8*rho_func(theta + h, w)
            - 8*rho_func(theta - h, w) + rho_func(theta - 2*h, w)) / (12*h)

# (GPT) QFI calculations
def _central_derivative(rho_func, theta, h, w):
    # simple centered difference: O(h^2)
    return (rho_func(theta + h, w) - rho_func(theta - h, w)) / (2*h)

# (GPT) QFI calculations
def _stable_drho(rho_func, theta, w, h=1e-6, method="five_point", richardson=True):
    """
    Compute drho/dtheta with higher accuracy and reduced round-off.
    """
    if method == "five_point":
        drho_h  = _five_point_derivative(rho_func, theta, h, w)
        if richardson:
            # Richardson extrapolation using h and h/2 (order 4 -> order 6 effective)
            drho_h2 = _five_point_derivative(rho_func, theta, h/2, w)
            # For a 5-point stencil (O(h^4)), Richardson factor = (2^4 - 1) = 15
            drho = drho_h2 + (drho_h2 - drho_h) / 15.0
        else:
            drho = drho_h
    else:
        drho_h  = _central_derivative(rho_func, theta, h, w)
        if richardson:
            drho_h2 = _central_derivative(rho_func, theta, h/2, w)
            # For centered O(h^2), factor = (2^2 - 1) = 3
            drho = drho_h2 + (drho_h2 - drho_h) / 3.0
        else:
            drho = drho_h

    return _symmetrize(drho)

# (GPT) QFI calculations
def quantum_fisher_information_mixed(
    rho_func, theta, w, *,
    h=1e-3,
    method="five_point",       # "five_point" | "central"
    richardson=False,
    sum_cutoff=1e-12,
    project_hermitian=True,
    chain=1.                 # d(phi)/d(theta) 등 체인룰 스케일링
):
    """
    안정화된 혼합상태 QFI 계산 (SLD 분해식).

    Parameters
    ----------
    rho_func : callable
        theta -> np.ndarray (complex)  밀도행렬  (dtype=complex128 권장)
    theta : float
        파라미터 값
    h : float
        유한차분 스텝
    method : str
        "five_point"(권장) 또는 "central"
    richardson : bool
        Richardson 외삽 사용 여부
    eig_cutoff : float
        아주 작은 고유값 제거(음수 수치잡음 포함)
    sum_cutoff : float
        분모 (λ_i + λ_j) 컷오프
    project_hermitian : bool
        rho를 Hermitian으로 투영(0.5*(ρ+ρ†))
    chain : float
        체인룰 스케일(예: phi=k*theta면 chain=k)

    Returns
    -------
    FQ : float
        QFI(theta)
    """
    # ρ(θ)
    rho = rho_func(theta, w)
    rho = np.array(rho, dtype=np.complex128)
    if project_hermitian:
        rho = _symmetrize(rho)

    # dρ/dθ
    drho = _stable_drho(rho_func, theta, w, h=h, method=method, richardson=richardson)

    # 분해
    eigvals, eigvecs = np.linalg.eigh(rho)

    # (선택) 음수 수치잡음 제거 + 재정규화는 보통 불필요하나
    # 극단적 잡음이면 아래 두 줄을 활성화 고려
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = eigvals / eigvals.sum()

    FQ = 0.0
    dim = rho.shape[0]

    # SLD 분해식: FQ = sum_{ij} 2 |<i|drho|j>|^2 / (λ_i + λ_j)
    for i in range(dim):
        vi = eigvecs[:,i]
        lam_i = eigvals[i]
        for j in range(dim):
            lam_j = eigvals[j]
            denom = lam_i + lam_j
            if denom > sum_cutoff:
                vj = eigvecs[:,j]
                elem = np.vdot(vi, drho @ vj)  # <i|drho|j>
                FQ += (2.0 / denom) * (abs(elem)**2)

    FQ = float(np.real_if_close(FQ))
    # 체인룰 스케일 적용 (예: F(θ) = (dφ/dθ)^2 F(φ))
    return (chain**2) * np.real_if_close(FQ)

# Circuit trainer class
class Trainer:

    def __init__(self, params: optarguments, circuitarg: circuitarguments, seed: int = 42, raw_config: dict = None):

        # Trainer configuration
        self.p = params
        # Circuit configuration
        self.cp = circuitarg
        # Seed for reproducibility
        self.seed = seed
        # Raw config dict for saving
        self.raw_config = raw_config or {}

        # Bias signal
        self.B = np.array(circuitarg.B)
        # Gyromagnetic ratio
        self.gm_ratio = circuitarg.gm_ratio
        # Circuit class
        self.circuit = Circuit(circuitarg, seed=seed)

        self.sweep_list = np.linspace(0.0, self.p.t_obs, self.p.num_points+1)[1:]

    # CFI as cost function
    def cost_function(self, w, circuit, B):

        return -qml.qinfo.classical_fisher(circuit.circuit)(B, w)

    # actual training method
    def train(self, save_to):

        print(self.circuit.w)
        print(self.circuit.bound)

        # Collect optimization logs
        opt_log = []

        self._iter_count = 0
        def callback(x, f, context):
            self._iter_count += 1
            cfi = -f if not hasattr(f, '__len__') else -f.item()
            t_s = x[self.circuit.ramsey.offset]
            print(f'[iter {self._iter_count:4d}] CFI = {cfi:2.4f}, t_s = {t_s*1e6:.4f} μs')
            opt_log.append({'iter': self._iter_count, 'cfi': cfi, 't_s': t_s})

        res = dual_annealing(self.cost_function, bounds=self.circuit.bound, args=(self.circuit, self.B),
                             maxiter=self.p.maxiter, initial_temp=self.p.initial_temp,
                             restart_temp_ratio=self.p.restart_temp_ratio, visit=self.p.visit,
                             accept=self.p.accept, maxfun=self.p.maxfun,
                             no_local_search=self.p.no_local_search,
                             seed=self.seed, callback=callback)

        self.circuit.w = res.x
        max_cfi = -res.fun
        if type(max_cfi) is not float:
            max_cfi = max_cfi.item()

        print(res)

        print(f'\nCFI : {max_cfi} , at sensing time : {self.circuit.w[self.circuit.ramsey.offset]*1e6:6f} μs')
        qfi = quantum_fisher_information_mixed(self.circuit.circuit, self.B, self.circuit.w)
        print(f'QFI = {qfi}')

        print(f'\nDensity Matrix')
        print(self.circuit.circuit(self.B, self.circuit.w))
        print(f'\nParameters')
        self.circuit.view_param()

        # Save all results to structured directory
        self._save_results(save_to, res, max_cfi, qfi, opt_log)

    def _save_results(self, save_to, res, max_cfi, qfi, opt_log):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = f'results/{save_to}_seed{self.seed}_{timestamp}'
        os.makedirs(run_dir, exist_ok=True)

        t_s_optimal = float(self.circuit.w[self.circuit.ramsey.offset])

        # 1. Save config
        with open(f'{run_dir}/config.yaml', 'w') as f:
            yaml.dump(self.raw_config, f, default_flow_style=False)

        # 2. Save result summary
        result_summary = {
            'timestamp': timestamp,
            'seed': self.seed,
            'max_cfi': float(max_cfi),
            'qfi': float(qfi),
            'cfi_qfi_ratio': float(max_cfi / qfi) if qfi > 0 else None,
            'optimal_sensing_time_s': t_s_optimal,
            'optimal_sensing_time_us': t_s_optimal * 1e6,
            'num_function_evals': int(res.nfev),
            'num_iterations': self._iter_count,
            'optimizer_message': str(res.message),
            'optimizer_success': bool(res.success),
        }
        with open(f'{run_dir}/result.json', 'w') as f:
            json.dump(result_summary, f, indent=2)

        # 3. Save optimized parameters
        np.save(f'{run_dir}/optimized_params.npy', self.circuit.w)

        # 4. Save optimization log
        if opt_log:
            with open(f'{run_dir}/opt_log.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['iter', 'cfi', 't_s'])
                writer.writeheader()
                writer.writerows(opt_log)

        # 5. Save density matrix
        dm = self.circuit.circuit(self.B, self.circuit.w)
        np.save(f'{run_dir}/max_dm.npy', dm)

        print(f'\nResults saved to: {run_dir}/')

        # 8. Append to README
        self._append_readme(result_summary, run_dir)

    def _append_readme(self, result, run_dir):
        readme_path = 'README.md'

        cfi_qfi = result['cfi_qfi_ratio']
        cfi_qfi_str = f"{cfi_qfi:.4f}" if cfi_qfi is not None else "N/A"

        row = (
            f"| {result['timestamp']} "
            f"| {result['seed']} "
            f"| {self.cp.num_wires} "
            f"| {self.cp.num_entangler} "
            f"| {self.cp.t2*1e6:.1f} "
            f"| {self.cp.p} "
            f"| {self.cp.B} "
            f"| {self.cp.ps} "
            f"| {result['max_cfi']:.4f} "
            f"| {result['qfi']:.4f} "
            f"| {cfi_qfi_str} "
            f"| {result['optimal_sensing_time_us']:.4f} "
            f"| `{run_dir}` |\n"
        )

        with open(readme_path, 'a') as f:
            f.write(row)