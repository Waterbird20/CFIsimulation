#!/bin/bash

# Sweep over p values and gamma_pd values with fixed t_s
# t_s = T2 * (-ln(1 - gamma_pd) / 2)^(1/p)

T2=2.0e-6

for p in 1.0 1.25 1.5 1.75 2.0 2.5 3.0; do
  for gamma_pd in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90; do
    # Compute t_s from gamma_pd
    fixed_ts=$(python3 -c "
import math
gamma_pd = ${gamma_pd}
T2 = ${T2}
p = ${p}
t_s = T2 * (-math.log(1 - gamma_pd) / 2) ** (1/p)
print(f'{t_s:.10e}')
")
    seed=$((RANDOM % 90000 + 10000))
    tag="fixed_p${p}_g${gamma_pd}"

    cat > config_${tag}.yaml <<EOF
num_wires:       2
num_entangler:   1
t2:              ${T2}
p:               ${p}
gm_ratio:        2.8e+6
B:               5.0
ps:              true
seed:            ${seed}
fixed_ts:        ${fixed_ts}

opt:             Adam
maxiter:         10000
initial_temp:    5230.0
restart_temp_ratio: 2.0e-5
visit:           2.62
accept:          -5.0
maxfun:          10000000
no_local_search: false
t_obs:           1.0e-6
num_points:      200
steps_per_point: 10000
patience:        200

num_process:     1

save_to:         ${tag}
EOF
    echo "=== Running p=${p}, gamma_pd=${gamma_pd}, fixed_ts=${fixed_ts}, seed=${seed} ==="
    python main.py config_${tag}.yaml
  done
done
