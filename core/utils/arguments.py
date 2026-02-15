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

# Arguments for optimization
@dataclass
class optarguments:

    opt:             str
    t_obs:           float
    num_points:      int
    steps_per_point: int
    patience:        int
    num_process:     int

# other arguments (file path to save data)
@dataclass
class otherarguments:

    save_to:    str