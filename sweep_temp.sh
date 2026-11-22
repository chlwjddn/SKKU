#!/bin/bash
# Temperature sweep: LightGCN softmax (masked positives), 3 layers, dim 64, graph rating>=4.
# Usage: bash sweep_temp.sh
D0=MIG-b96c2293-1d8b-5c06-af55-e32e6aec3bb8
D1=MIG-8e1f0d17-dd71-54ca-8315-68ce187d0c0b
TEMPS="0.05 0.07 0.1 0.15 0.2 0.3 0.5"
mkdir -p logs
i=0; TAGS=""
for T in $TEMPS; do
  tag=ce_t${T}
  TAGS="${TAGS:+$TAGS,}$tag"
  if [ $((i % 2)) -eq 0 ]; then dev=$D0; else dev=$D1; fi
  CUDA_VISIBLE_DEVICES=$dev python lightgcn.py --data ~/RS/data --loss ce --lr 5e-3 --reg 0 \
    --temp $T --no_final --tag $tag > logs/$tag.txt 2>&1 &
  i=$((i + 1))
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
echo "=== best epoch per temperature ==="
for T in $TEMPS; do echo "T=$T  $(grep 'best epoch' logs/ce_t${T}.txt)"; done
echo
python temp_report.py --data ~/RS/data --tags $TAGS
