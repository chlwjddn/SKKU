#!/bin/bash
# Seed ensemble + real-LightGCN merge, end to end.
# Trains anything whose scores/{tag}_test.pt is missing, then blends:
#   blend_seed: softmax MF x3 + EASE + fut7_2ch x3          (7 models)
#   blend9    : blend_seed + lightgcn_basic (BPR+DNS) x2    (9 models)
# Usage: bash run_ensemble.sh
set -e
export D0=${D0:-MIG-b96c2293-1d8b-5c06-af55-e32e6aec3bb8}
export D1=${D1:-MIG-8e1f0d17-dd71-54ca-8315-68ce187d0c0b}
M="--data ~/RS/data --weighted --input two_channel --future_cuts 0.3,0.4,0.5,0.6,0.7,0.75,0.8"
mkdir -p logs scores

need() { [ ! -f "scores/$1_test.pt" ]; }

# -------- softmax MF seeds (lightgcn.py defaults = SOTA) --------
need lgcn_best && CUDA_VISIBLE_DEVICES=$D0 python lightgcn.py --data ~/RS/data --seed 42 --tag lgcn_best > logs/ens_lgcn_best.txt 2>&1 &
need lg_s7     && CUDA_VISIBLE_DEVICES=$D1 python lightgcn.py --data ~/RS/data --seed 7    --tag lg_s7  > logs/ens_lg_s7.txt  2>&1 &
wait
need lg_s26    && CUDA_VISIBLE_DEVICES=$D0 python lightgcn.py --data ~/RS/data --seed 2026 --tag lg_s26 > logs/ens_lg_s26.txt 2>&1 &
# -------- fut7_2ch seeds --------
need fut7_2ch  && CUDA_VISIBLE_DEVICES=$D1 python multvae.py $M --seed 42   --tag fut7_2ch > logs/ens_fut42.txt 2>&1 &
wait
need fut_s7    && CUDA_VISIBLE_DEVICES=$D0 python multvae.py $M --seed 7    --tag fut_s7  > logs/ens_fut7.txt  2>&1 &
need fut_s26   && CUDA_VISIBLE_DEVICES=$D1 python multvae.py $M --seed 2026 --tag fut_s26 > logs/ens_fut26.txt 2>&1 &
wait
# -------- EASE (deterministic, one copy) --------
need ease && python base_scores.py --data ~/RS/data > logs/ens_ease.txt 2>&1
# -------- real LightGCN (BPR + DNS + L2, user's lightgcn_basic.py) --------
need lgb_s0 && CUDA_VISIBLE_DEVICES=$D0 python lightgcn_basic.py --data ~/RS/data --seed 0 --tag lgb_s0 > logs/ens_lgb0.txt 2>&1 &
need lgb_s7 && CUDA_VISIBLE_DEVICES=$D1 python lightgcn_basic.py --data ~/RS/data --seed 7 --tag lgb_s7 > logs/ens_lgb7.txt 2>&1 &
wait

echo "== singles =="
grep -h "best epoch\|\[val\]" logs/ens_*.txt 2>/dev/null | tail -20

echo "== blend_seed (7 models) =="
python blend.py --data ~/RS/data \
  --tags lgcn_best,lg_s7,lg_s26,ease,fut7_2ch,fut_s7,fut_s26 \
  --out submission_blend_seed.csv --save_tag blend_seed | tail -4

echo "== blend9 (+ real LightGCN x2) =="
python blend.py --data ~/RS/data \
  --tags lgcn_best,lg_s7,lg_s26,ease,fut7_2ch,fut_s7,fut_s26,lgb_s0,lgb_s7 \
  --out submission_blend9.csv --save_tag blend9 | tail -4

echo "baseline to beat: blend4 val 0.1779"
