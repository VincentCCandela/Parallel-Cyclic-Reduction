#!/bin/bash
set -u
SRC=tri_diag_par_v2.cu
BIN=tri_diag_par
CSV=sweep_results_gpu_v2.csv

echo "NX,NY,gpu_time" > "$CSV"

extract_gpu_time() {
    awk -F',' 'NF>=3 {gsub(/[[:space:]]/,"",$3); if ($3!="") {print $3; exit}}'
}


# extract_gpu_time() {
#     awk '/^GPU time:/ {print $3; exit}'
# }

# extract_cpu_time() {
#     awk '/^CPU completed in/ {print $4; exit}'
# }

for ((nx=16; nx<=4096*8; nx*=2)); do
    for ((ny=16; ny<=4096*8; ny*=2)); do
        if nvcc -arch=compute_60 -code=sm_60 -DNX=$nx -DNY=$ny \
                "$SRC" -o "$BIN" >/dev/null 2>&1; then
            out=$(./"$BIN" 2>/dev/null)
            tg=$(printf '%s\n' "$out" | extract_gpu_time)
            # tc=$(printf '%s\n' "$out" | extract_cpu_time)
            tg=${tg:-NA}
            tc=${tc:-NA}
        else
            tg=compile_error
            tc=compile_error
        fi
        echo "$nx,$ny,$tg" >> "$CSV"
        echo "NX=$nx NY=$ny -> gpu=$tg"
    done
done

echo "results written to $CSV"
