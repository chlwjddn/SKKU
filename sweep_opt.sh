#!/bin/bash
# Optimizer sweep on the current SOTA LightGCN config. Usage: bash sweep_opt.sh
D0=MIG-b96c2293-1d8b-5c06-af55-e32e6aec3bb8
D1=MIG-8e1f0d17-dd71-54ca-8315-68ce187d0c0b
C="--data ~/RS/data --loss ce --reg 0 --temp 0.1 --dim 128 --rating_w 4:0.7,5:1 --time_beta 1.25 --no_final"
mkdir -p logs
JOBS=()
# 1) AdamW: lr x weight decay (no schedule)
for lr in 2e-3 5e-3 1e-2; do for wd in 0 1e-3 1e-2; do
  JOBS+=("--opt adamw --lr $lr --wd $wd|aw_lr${lr}_wd${wd}")
done; done
# 2) AdamW + cosine schedule with warmup (600-epoch horizon)
for lr in 5e-3 1e-2 2e-2; do
  JOBS+=("--opt adamw --lr $lr --wd 0 --sched cosine --warmup 10 --epochs 600|aw_cos_lr${lr}")
done
# 3) batch size
for bs in 2048 8192; do
  JOBS+=("--opt adamw --lr 5e-3 --wd 0 --batch $bs|aw_bs${bs}")
done

i=0; TAGS="tb1.25"
for j in "${JOBS[@]}"; do
  opts="${j%%|*}"; tag="${j##*|}"; TAGS="$TAGS,$tag"
  if [ $((i % 2)) -eq 0 ]; then dev=$D0; else dev=$D1; fi
  CUDA_VISIBLE_DEVICES=$dev python lightgcn.py $C $opts --tag $tag > logs/$tag.txt 2>&1 &
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
for j in "${JOBS[@]}"; do tag="${j##*|}"; echo "$tag: $(grep 'best epoch' logs/$tag.txt)"; done
python temp_report.py --data ~/RS/data --tags $TAGS
