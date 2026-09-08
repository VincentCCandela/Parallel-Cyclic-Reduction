#!/bin/bash -l
#$ -P ec527
#$ -N tri_diag_sweep
#$ -l h_rt=06:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -j y
#$ -o sweep.log
#$ -m ea
#$ -M vcandela@bu.edu

set -u

module load cuda

SRC=tri_diag_par.cu
BIN=tri_diag_par
CSV=sweep_results_qsub_v5.csv

: > "$CSV"

for ((nx=16; nx<=4096; nx*=2)); do
    for ((ny=16; ny<=4096; ny*=2)); do
        echo "=== NX=$nx NY=$ny ==="
        if nvcc -arch=compute_60 -code=sm_60 -DNX=$nx -DNY=$ny \
                "$SRC" -o "$BIN" >/dev/null 2>&1; then
            ./"$BIN" 2>/dev/null | tee -a "$CSV"
        else
            echo "compile_error"
        fi
    done
done

echo "results written to $CSV"