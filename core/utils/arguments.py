from dataclasses import dataclass

# Arguments dataclass definition for convenient execution

# Arguments for quantum circuit
@dataclass
class circuitarguments:

    num_wires:      int
    num_entangler:  int
    t2:             float
    p:              float
    gm_ratio:       float
    B:              float
    t:              float
    ps:             bool
    fixed_ts:       float = None

# Arguments for optimization
@dataclass
class optarguments:

    opt:                str
    maxiter:            int   = 10000
    initial_temp:       float = 5230.0
    restart_temp_ratio: float = 2e-5
    visit:              float = 2.62
    accept:             float = -5.0
    maxfun:             int   = 10000000
    no_local_search:    bool  = False
    t_obs:              float = 0.0
    num_points:         int   = 200
    steps_per_point:    int   = 10000
    patience:           int   = 200
    num_process:        int   = 1

# other arguments (file path to save data)
@dataclass
class otherarguments:

    save_to:    str
    seed:       int = 42