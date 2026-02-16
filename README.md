# CFI Simulation

Classical Fisher Information (CFI) optimization for multi-qubit quantum sensing circuits, targeting nitrogen-vacancy (NV) center magnetometry with parameterized entangling and Ramsey protocols.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture / File Structure](#2-architecture--file-structure)
3. [Per-File Breakdown](#3-per-file-breakdown)
4. [Physics Implementation](#4-physics-implementation)
5. [Simulation Workflow / Data Flow](#5-simulation-workflow--data-flow)
6. [Key Parameters and Configurations](#6-key-parameters-and-configurations)
7. [Visualization / Output](#7-visualization--output)
8. [Getting Started](#8-getting-started)

---

## 1. Project Overview

This project simulates and optimizes **Classical Fisher Information (CFI)** for parameterized quantum circuits designed for DC magnetic field sensing. The physical system is an ensemble of qubits modeling **NV center electron spins** undergoing Ramsey-type free precession in an external magnetic field **B**, subject to **T2 dephasing**.

The core idea: given a quantum sensing protocol (state preparation, entanglement, free evolution under B, dephasing, measurement), find the circuit parameters (sensing time, rotation angles, entangler timings) that **maximize the CFI** with respect to the magnetic field parameter. The CFI sets the Cramer-Rao lower bound on the variance of any unbiased estimator of B, so maximizing it yields the most precise magnetometry.

The simulation supports:
- **Single-qubit Ramsey magnetometry** (1 wire, RX initialization, Ramsey-Z sensing)
- **Multi-qubit entangled sensing** (N wires, RY initialization, parameterized ZZ entangling layers, Ramsey-Z sensing)
- **Optional post-selection** on the output density matrix
- **QFI benchmarking** via the SLD (Symmetric Logarithmic Derivative) decomposition of the mixed-state density matrix, providing an upper bound against which the optimized CFI is compared

The quantum circuit is built and simulated using **PennyLane** (`default.mixed` device for mixed-state support), and the optimization uses **SciPy's `dual_annealing`** global optimizer.

---

## 2. Architecture / File Structure

```
CFIsimulation/
├── main.py                     # Entry point: parse config, build circuit, run optimization
├── config.yaml                 # User-facing configuration file (all tunable parameters)
├── pyproject.toml              # Project metadata and dependencies
├── requirements.txt            # Pip dependency list
├── core/
│   ├── circuit.py              # Circuit class: assembles layers into a PennyLane QNode
│   ├── layers.py               # Layer definitions: Initialization, Entangler, RamseyZ, PostSelection
│   ├── trainer.py              # Trainer class: CFI cost function, dual_annealing optimization, QFI benchmark
│   └── utils/
│       ├── arguments.py        # Dataclass definitions for typed configuration (circuitarguments, optarguments, etc.)
│       ├── customparser.py     # YAML config parser → dataclass instances
│       └── utils.py            # Hamiltonians, dephasing, density matrix plotting, file I/O helpers
├── dmplots/                    # Output directory for density matrix visualizations
└── paramplots/                 # Output directory for optimized parameter plots
```

### Module dependency graph

```
main.py
 ├── core/utils/customparser.py  →  core/utils/arguments.py   (config parsing)
 ├── core/utils/utils.py                                       (file I/O helpers)
 ├── core/circuit.py             →  core/layers.py             (circuit assembly)
 │                               →  core/utils/arguments.py
 │                               →  core/utils/utils.py        (Hamiltonians)
 └── core/trainer.py             →  core/circuit.py            (optimization loop)
                                 →  core/utils/utils.py        (plotting)
```

---

## 3. Per-File Breakdown

### `main.py`

**Purpose:** Entry point. Orchestrates the full pipeline: config parsing → circuit construction → CFI optimization → result saving.

| Step | Code | Description |
|------|------|-------------|
| 1 | `customparser(config_file).parse_custom_args()` | Parses `config.yaml` into three typed dataclass objects |
| 2 | `clean_container()` | Clears previous output directories (`dmplots/`, `paramplots/`) |
| 3 | `Trainer(optarg, circuitarg)` | Builds the quantum circuit and prepares the optimizer |
| 4 | `t.train(save_to)` | Runs dual_annealing optimization to maximize CFI |
| 5 | `parse_data(save_to)` | Copies results into a named output directory |

**Input:** Command-line argument — path to a YAML config file (e.g., `python main.py config.yaml`).
**Output:** `.npy` data files and plots in the named output folder.

---

### `core/circuit.py` — `Circuit` class

**Purpose:** Assembles parameterized quantum circuit layers into a PennyLane QNode that maps `(B, w) → density_matrix`.

**Constructor logic (`__init__`):**

1. **Single-qubit mode** (`num_wires == 1`):
   - `Initialization('X')` → RX(pi/2) on each wire (superposition along X)
   - `RamseyZ` layer for sensing

2. **Multi-qubit mode** (`num_wires > 1`):
   - `Initialization('Y')` → RY(pi/2) on each wire
   - `num_entangler` × `Entangler` layers (ZZ interaction + parameterized rotations)
   - `RamseyZ` layer for sensing

3. **Optional PostSelection** layer (if `ps: true` in config)

4. Wraps everything in a `@qml.qnode(default.mixed)` returning `qml.density_matrix()`

**Key attributes:**
- `self.w` — flat numpy array of all trainable parameters (initialized randomly within bounds, seed=42)
- `self.bound` — list of `(min, max)` tuples for each parameter (used by the optimizer)
- `self.circuit` — the callable QNode: `circuit(B, w) → density_matrix`
- `self.n_params` — total number of trainable parameters

**Key methods:**
- `view_param()` — prints current parameter values grouped by layer
- `set_params(x)` — sets parameters from an external array
- `draw_circuit()` — saves a circuit diagram to `circuit.png`
- `plot_params(data)` — delegates per-layer parameter plotting

---

### `core/layers.py` — Circuit layer definitions

**Purpose:** Defines each building block of the quantum circuit as a callable class. Each layer knows its parameter count, parameter bounds, and how to apply itself to a PennyLane circuit.

#### `CircuitLayer` (base class)
- Common attributes: `num_wires`, `id`, `n_params`, `offset` (index into the global parameter vector `w`), `bound`
- `get_param_bound()` → returns list of `(min, max)` tuples

#### `Initialization`
- **Parameters:** 0 (no trainable params)
- **Action:** Applies RX(pi/2) (type `'X'`) or RY(pi/2) (type `'Y'`) to all wires, creating an equal superposition state
- For single-qubit: `|0⟩ → |+⟩` via RX; for multi-qubit: `|0⟩ → |+y⟩` via RY

#### `Entangler`
- **Parameters:** 3 per layer — `[tau, theta, tau']`
- **Bounds:** `[-2pi, 2pi]` for each
- **Action (in order):**
  1. `ApproxTimeEvolution(H_ZZ, tau, 1)` — evolves under the all-pairs ZZ Hamiltonian for time `tau`
  2. `RX(theta)` + `RY(-pi/2)` on each wire — parameterized single-qubit rotation
  3. `ApproxTimeEvolution(H_ZZ, tau', 1)` — second ZZ evolution for time `tau'`
  4. `RY(pi/2)` on each wire — rotation back
- **Hamiltonian:** `H_ZZ = 0.5 * sum_{i<j} Z_i Z_j` (Ising-type entangling interaction)

#### `RamseyZ` (primary sensing layer)
- **Parameters:** `num_wires + 1` — `[t_s, theta_Z1, theta_Z2, ..., theta_ZN]`
- **Bounds:** `t_s ∈ [1 ns, 2*T2]`, each `theta_Zi ∈ [-2pi, 2pi]`
- **Action (in order):**
  1. Constructs Ramsey Hamiltonian: `H = sum_i (gamma * t_s / 2) * Z_i`
  2. `ApproxTimeEvolution(H, B, 1)` — free precession under B for effective time `t_s` (the sensing time is encoded in the Hamiltonian coefficient; B is the evolution parameter)
  3. For each wire: `PhaseDamping(tau)` → `RZ(theta_Zi)` → `RX(pi/2)`
  4. Dephasing factor: `tau = 1 - exp(-2 * (t_s/T2)^p)`, applied via PennyLane's `PhaseDamping` channel

#### `Ramsey` (alternative, not used by default)
- Uses a fixed `B` in the Hamiltonian and treats `t` as the input parameter
- Includes depolarizing channel (commented out) and a final RX(pi/2) readout rotation

#### `PostSelection`
- **Parameters:** `3 * num_wires` — `[gamma_1..N, theta_Z1, theta_X1, ..., theta_ZN, theta_XN]`
- **Bounds:** `gamma_i ∈ [0, 1-epsilon]`, angles `∈ [-2pi, 2pi]`
- **Action:** Applies a Kraus operator `K = diag(sqrt(1-gamma_i), 1)` per qubit via tensor product, renormalizes the density matrix (`K rho K† / Tr(K rho K†)`), then applies RX and RZ rotations
- Implements weak-measurement post-selection to probabilistically filter states

#### `Rotate` (deprecated)
- Single-parameter rotation (RX/RY/RZ) on all wires

---

### `core/trainer.py` — `Trainer` class and QFI computation

**Purpose:** Defines the optimization loop that maximizes CFI over circuit parameters, and provides a QFI benchmark via SLD decomposition.

#### CFI Calculation: `cost_function(self, w, circuit, B)`

```python
return -qml.gradients.classical_fisher(circuit.circuit)(B, w)
```

- Uses **PennyLane's built-in `classical_fisher`** function
- `classical_fisher` computes the CFI matrix of the QNode with respect to the input parameters
- Here it computes `F_C(B)` — the classical Fisher information about B given the probability distribution `p(outcome | B)` obtained from the circuit's density matrix
- The negative sign converts maximization into a minimization problem for `dual_annealing`

#### QFI Calculation: `quantum_fisher_information_mixed()`

Computes the **Quantum Fisher Information** via the SLD eigendecomposition for mixed states:

```
F_Q = sum_{i,j: lambda_i + lambda_j > cutoff} 2 * |<i|drho/dB|j>|^2 / (lambda_i + lambda_j)
```

Where:
- `rho(B)` is the density matrix, `drho/dB` its derivative w.r.t. B
- `{lambda_i, |i⟩}` are eigenvalues/eigenvectors of `rho`
- The derivative is computed via **5-point centered finite difference** with optional Richardson extrapolation (up to O(h^6) accuracy)

This QFI value serves as an **upper bound** on the CFI: `F_C <= F_Q`. The ratio `F_C/F_Q` indicates how close the chosen measurement basis is to optimal.

#### Optimization: `train(self, save_to)`

1. Calls `scipy.optimize.dual_annealing` with:
   - **Cost function:** negative CFI (to minimize)
   - **Bounds:** from `circuit.bound` (sensing time, rotation angles, etc.)
   - **Max iterations:** 4000
   - **Seed:** 42 (reproducible)
2. After convergence:
   - Prints optimized CFI, QFI, optimal sensing time, and final density matrix
   - Saves density matrix plot and parameter data to disk

---

### `core/utils/arguments.py` — Configuration dataclasses

| Dataclass | Fields | Description |
|-----------|--------|-------------|
| `circuitarguments` | `num_wires`, `num_entangler`, `t2`, `p`, `gm_ratio`, `B`, `t`, `ps` | Quantum circuit / physical parameters |
| `optarguments` | `opt`, `t_obs`, `num_points`, `steps_per_point`, `patience`, `num_process` | Optimization settings |
| `otherarguments` | `save_to` | Output directory name |

---

### `core/utils/customparser.py` — YAML configuration parser

**Purpose:** Reads a YAML config file and maps its keys into the three argument dataclasses.

- `customparser(file_path)` — loads YAML into a dict
- `parse_custom_args()` → returns `(circuitarguments, optarguments, otherarguments)` tuple

---

### `core/utils/utils.py` — Utility functions

| Function | Description |
|----------|-------------|
| `get_entangler(num_wires)` | Returns `H_ZZ = 0.5 * sum_{i<j} Z_i Z_j` PennyLane Hamiltonian for all-pairs Ising interaction |
| `get_ramsey(num_wires, gm_ratio, t_obs)` | Returns `H = sum_i (gamma * t_obs / 2) * Z_i` — Ramsey free-precession Hamiltonian |
| `dephase_factor(tau)` | `1 - exp(-2*tau)` (PyTorch version) |
| `dephase_factor_nontorch(tau)` | `1 - exp(-2*tau)` (NumPy version) |
| `get_noise_channel(tau)` | Returns `{I, X, Y, Z}` Kraus operators for a depolarizing channel (test/unused) |
| `plot_density_matrix(rho, t, save_as)` | 3D bar plot of the real part of the density matrix, saved to `dmplots/` |
| `clean_container()` | Clears `dmplots/` and `paramplots/` directories |
| `parse_data(save_to)` | Copies output files into a named results directory |

---

## 4. Physics Implementation

### 4.1 Physical System

The system models **N NV center electron spins** (qubits) sensing a DC magnetic field **B** (in Hz). Each spin precesses around the Z axis at the Larmor frequency `omega = gamma * B`, where `gamma` is the gyromagnetic ratio (default: `2.8 MHz/G` for NV centers). Dephasing at rate `1/T2` degrades the quantum coherence during the sensing interval.

### 4.2 Sensing Protocol

The circuit implements a generalized Ramsey interferometry protocol:

```
|0⟩^N  →  [Init: R_Y(pi/2)]  →  [Entangler layers]  →  [Ramsey-Z: free evolution + dephasing + RZ + RX(pi/2)]  →  measure
```

**Single-qubit case:** Standard Ramsey measurement — prepare `|+⟩` via RX(pi/2), accumulate phase `phi = gamma * B * t_s` under H_Z, apply dephasing, then project back.

**Multi-qubit case:** Adds parameterized entangling layers (ZZ evolution + single-qubit rotations) before the Ramsey block. This can create entangled probe states (e.g., GHZ-like states) that achieve Heisenberg-limited scaling `F ~ N^2` instead of the standard quantum limit `F ~ N`.

### 4.3 Noise Model

**Phase damping (dephasing):** The dominant decoherence mechanism for NV centers. Applied via PennyLane's `PhaseDamping(gamma_pd)` channel on each qubit after free precession:

```
gamma_pd = 1 - exp(-2 * (t_s / T2)^p)
```

- `T2` — coherence time (seconds)
- `p` — stretched exponential exponent (default 1.0 for standard exponential decay; `p < 1` models non-Markovian / 1/f noise environments; `p = 2` would model Gaussian decay typical of spin-bath environments)

The `PhaseDamping` channel acts as:
```
rho → (1 - gamma_pd) * rho + gamma_pd * Z rho Z
```
(in the single-qubit Pauli representation), which decays off-diagonal elements while preserving populations.

### 4.4 Classical Fisher Information

The CFI is computed by PennyLane's `qml.gradients.classical_fisher()`. Internally, this:

1. Evaluates the circuit to get the output density matrix `rho(B)`
2. Computes measurement outcome probabilities `p_k(B)` (diagonal of `rho` in the computational basis)
3. Computes the CFI:

```
F_C(B) = sum_k  (dp_k/dB)^2 / p_k(B)
```

The derivative `dp_k/dB` is obtained via PennyLane's automatic differentiation (parameter-shift rules on the QNode).

### 4.5 Quantum Fisher Information (Benchmark)

The QFI is computed from the density matrix eigendecomposition (SLD formula):

```
F_Q(B) = sum_{i,j: lambda_i + lambda_j > eps} 2 |<i| drho/dB |j>|^2 / (lambda_i + lambda_j)
```

Where `drho/dB` is computed via a 5-point finite difference stencil with step size `h = 1e-3`. For pure states, this reduces to `F_Q = 4 * Var(H)`.

The QFI provides the **ultimate precision bound** — no measurement scheme can yield `F_C > F_Q`. The ratio `F_C / F_Q` quantifies the efficiency of the computational-basis measurement.

### 4.6 Optimization

The optimizer (`scipy.optimize.dual_annealing`) is a global optimization method combining classical simulated annealing with local search. It searches over:

- Sensing time `t_s ∈ [1 ns, 2*T2]`
- Entangler parameters `tau, theta, tau' ∈ [-2pi, 2pi]`
- Post-Ramsey RZ angles `theta_Zi ∈ [-2pi, 2pi]`
- (Optional) Post-selection parameters `gamma_i ∈ [0, 1)`, rotation angles

The optimizer finds the parameter set that maximizes `F_C(B)`, balancing the tradeoff between longer sensing times (more phase accumulation) and stronger dephasing.

---

## 5. Simulation Workflow / Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  config.yaml                                                     │
│  (num_wires, T2, gamma, B, num_entangler, ps, ...)               │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  customparser.parse_custom_args()                                │
│  → circuitarguments, optarguments, otherarguments                │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Circuit.__init__(circuitarg)                                    │
│                                                                  │
│  1. Build layers:                                                │
│     Initialization → [Entangler × N] → RamseyZ → [PostSelection] │
│                                                                  │
│  2. Assemble QNode:                                              │
│     @qml.qnode(default.mixed)                                    │
│     def circuit(B, w):                                           │
│         init() → entanglers(w) → ramseyZ(w, B) → density_matrix  │
│                                                                  │
│  3. Initialize parameters w randomly within bounds               │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Trainer.__init__(optarg, circuitarg)                            │
│  → stores B, gm_ratio, creates Circuit instance                  │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Trainer.train(save_to)                                          │
│                                                                  │
│  dual_annealing(                                                 │
│    cost = -classical_fisher(circuit)(B, w),                      │
│    bounds = circuit.bound,                                       │
│    maxiter = 4000                                                │
│  )                                                               │
│                                                                  │
│  Each evaluation:                                                │
│    w_candidate → circuit(B, w) → rho(B) → CFI(B) → -CFI          │
│                                                                  │
│  After convergence:                                              │
│    optimal w* → max CFI, QFI benchmark                           │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Output                                                          │
│  • Console: CFI, QFI, optimal sensing time, density matrix       │
│  • Files:  {save_to}_data.npy, {save_to}_max_dm.npy              │
│  • Plots:  dmplots/*.png, paramplots/*.png                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Key Parameters and Configurations

All parameters are set in `config.yaml`:

### Circuit / Physics Parameters

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Number of qubits | `num_wires` | 4 | Number of NV spins (1 = standard Ramsey, >1 = entangled sensing) |
| Entangler count | `num_entangler` | 1 | Number of ZZ entangling layers before the Ramsey block |
| T2 coherence time | `t2` | 2.0 us | Dephasing time; sets the noise scale and sensing time upper bound |
| Stretched exponent | `p` | 1.0 | Exponent in `exp(-(t/T2)^p)`; 1.0 = Markovian, <1 = 1/f noise, 2 = Gaussian |
| Contrast / visibility | `gamma` | 0.75 | (Defined but not directly used in RamseyZ; available for extensions) |
| Gyromagnetic ratio | `gm_ratio` | 2.8 MHz/G | NV center electron spin gamma; scales phase accumulation rate |
| Bias field | `B` | 5.0 | External DC magnetic field to estimate (in units consistent with gm_ratio) |
| Post-selection | `ps` | false | Enable post-selection layer for probabilistic state filtering |

### Optimization Parameters

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Optimizer | `opt` | Adam | (Legacy; actual optimization uses `dual_annealing`) |
| Observation time | `t_obs` | 3.0 us | Maximum observation time (used for sweep list, not directly constraining t_s) |
| Number of sweep points | `num_points` | 200 | Points in the time sweep array (currently used for sweep_list generation) |
| Steps per point | `steps_per_point` | 10000 | (Legacy; relevant for gradient-based training, not used by dual_annealing) |
| Patience | `patience` | 200 | (Legacy; early stopping patience for gradient-based training) |
| Parallel processes | `num_process` | 1 | (Reserved for multiprocessing; not currently active) |

### Output

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Save directory | `save_to` | `xz_test` | Name for the output folder and file prefix |

---

## 7. Visualization / Output

### Console Output

After optimization completes, the trainer prints:
- **Optimized CFI value** and the **optimal sensing time** `t_s` (in microseconds)
- **QFI value** for comparison (upper bound on CFI)
- **Full density matrix** at the optimal parameters
- **Per-layer parameter values** (entangler timings, RZ angles, etc.)
- **Dual annealing result object** (convergence info, number of function evaluations)

### Saved Files

| File | Location | Description |
|------|----------|-------------|
| `{save_to}_data.npy` | `./{save_to}/` | NumPy array: `[max_CFI, QFI, w_0, w_1, ..., w_N]` — all results in one vector |
| `{save_to}_max_dm.npy` | `./` | Density matrix at optimal parameters (complex numpy array) |
| `{save_to}_max_dm.png` | `./dmplots/` | 3D bar plot of the real part of the optimal density matrix |
| Parameter plots | `./paramplots/` | Per-layer plots of the optimized parameter values |

### Interpreting the Density Matrix Plot

The 3D bar chart shows the **real part** of each element of the output density matrix `rho` in the computational basis. Key features to look for:
- **Diagonal elements** → population distribution across computational basis states
- **Off-diagonal elements** → coherences; large off-diagonals at `(0,N)` and `(N,0)` suggest GHZ-like entangled states
- **Color scale** → amplitude magnitude (rainbow colormap)
- The title includes the sensing time `t_s` at which this state was produced

### Interpreting CFI vs QFI

- `F_C / F_Q ~ 1` → the computational basis measurement is near-optimal; the circuit is well-designed
- `F_C / F_Q << 1` → a different measurement basis would extract more information; consider adding post-selection or adjusting the final rotation angles
- For N entangled qubits, the Heisenberg limit is `F_Q ~ N^2 * t_s^2 * gamma^2`, compared to the SQL `F ~ N * t_s^2 * gamma^2`

---

## 8. Getting Started

### Installation

```bash
# Clone and set up environment
cd CFIsimulation
pip install -r requirements.txt
# or with uv:
uv sync
```

### Required directories

```bash
mkdir -p dmplots paramplots
```

### Running a simulation

```bash
python main.py config.yaml
```

Edit `config.yaml` to change the number of qubits, coherence time, magnetic field, or other parameters. Results will be saved to the directory specified by `save_to`.

### Example: Single-qubit Ramsey

```yaml
num_wires:       1
num_entangler:   0
t2:              2.0e-6
p:               1.0
gm_ratio:        2.8e+6
B:               5.0
ps:              false
opt:             Adam
t_obs:           3.0e-6
num_points:      200
steps_per_point: 10000
patience:        200
num_process:     1
save_to:         single_qubit_ramsey
```

### Example: 4-qubit entangled sensing with post-selection

```yaml
num_wires:       4
num_entangler:   2
t2:              2.0e-6
p:               1.0
gm_ratio:        2.8e+6
B:               5.0
ps:              true
opt:             Adam
t_obs:           3.0e-6
num_points:      200
steps_per_point: 10000
patience:        200
num_process:     1
save_to:         entangled_4q_ps
```
