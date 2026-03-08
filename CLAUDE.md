# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (preferred)
uv sync

# Or with pip
pip install -r requirements.txt

# Run a simulation
python main.py config.yaml

# Run with a custom config
python main.py path/to/custom_config.yaml
```

Python 3.12 is required (see `.python-version`). The project uses `uv` for dependency management (`pyproject.toml` + `uv.lock`).

## Architecture

This project optimizes **Classical Fisher Information (CFI)** for parameterized quantum circuits to maximize magnetic field sensing precision in NV center magnetometry. The optimizer is `scipy.optimize.dual_annealing` (global search, max 10000 iterations, seed=42).

### Data flow

```
config.yaml → customparser → (circuitarguments, optarguments, otherarguments)
                                    ↓
                            Circuit.__init__()   ← layers.py
                            (builds QNode, initializes params w)
                                    ↓
                            Trainer.train()
                            dual_annealing(-CFI, bounds=circuit.bound)
                                    ↓
                            {save_to}_data.npy   [max_CFI, QFI, w...]
                            {save_to}_max_dm.npy [density matrix]
                            dmplots/, paramplots/, circuit.png
```

### Key classes and their responsibilities

- **`core/circuit.py` — `Circuit`**: Assembles PennyLane QNode (`default.mixed` device). `circuit.circuit(B, w)` returns the output density matrix. Key attributes: `circuit.w` (parameter vector), `circuit.bound` (optimizer bounds), `circuit.ramsey` (reference to the RamseyZ layer, used to index the sensing time `t_s` in `w`).

- **`core/layers.py`**: Layer classes (`Initialization`, `Entangler`, `RamseyZ`, `PostSelection`). Each layer tracks its `offset` into the global `w` vector and its parameter `bound`. The `RamseyZ` layer's first parameter (`w[ramsey.offset]`) is always the sensing time `t_s`.

- **`core/trainer.py` — `Trainer`**: The CFI cost is `qml.qinfo.classical_fisher(circuit.circuit)(B, w)`. QFI is computed via SLD eigendecomposition with 5-point finite differences (functions prefixed with `# (GPT)`). QFI serves as the upper bound to benchmark against the optimized CFI.

- **`core/utils/arguments.py`**: Typed dataclasses for config — `circuitarguments` (physics/circuit), `optarguments` (optimizer settings), `otherarguments` (output dir name).

### Circuit topology

- **`num_wires == 1`**: `Initialization('X')` → `RamseyZ`  (standard single-qubit Ramsey)
- **`num_wires > 1`**: `Initialization('Y')` → `Entangler` × `num_entangler` → `RamseyZ` → (optional `PostSelection`)

The `Entangler` layer applies ZZ Hamiltonian evolution (`H_ZZ = 0.5 * sum_{i<j} Z_i Z_j`) with 3 trainable parameters per layer `[tau, theta, tau']`. The `RamseyZ` layer has `num_wires + 1` parameters: sensing time `t_s` plus one RZ angle per wire. Dephasing is `gamma_pd = 1 - exp(-2*(t_s/T2)^p)`.

### Output files

After `train()`:
- `{save_to}_data.npy` — 1D array: `[max_CFI, QFI, w_0, ..., w_N]`
- `{save_to}_max_dm.npy` — complex density matrix at optimal params
- `dmplots/{save_to}_max_dm.png` — 3D bar plot of density matrix real part
- `circuit.png` — circuit diagram

`clean_container()` wipes `dmplots/` and `paramplots/` at the start of each run. `parse_data(save_to)` copies results into a named subdirectory.

### config.yaml key parameters

| Key | Meaning |
|-----|---------|
| `num_wires` | Number of qubits (1 = standard Ramsey, >1 = entangled) |
| `num_entangler` | Number of ZZ entangling layers |
| `t2` | T2 coherence time (seconds, e.g. `2.0e-6`) |
| `p` | Stretched exponential exponent (1.0 = Markovian) |
| `gm_ratio` | Gyromagnetic ratio (Hz/G, default `2.8e6` for NV) |
| `B` | Magnetic field to estimate (in units matching `gm_ratio`) |
| `ps` | Enable post-selection layer (`true`/`false`) |
| `t_obs` | Used to build `sweep_list`; does not cap `t_s` directly |
| `save_to` | Output file prefix and subdirectory name |
