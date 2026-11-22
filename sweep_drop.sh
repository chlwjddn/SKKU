#!/bin/bash
# 엣지 dropout 비율만 바꾸는 스윕 (dns32, 평점≥4 고정). drop=0 = dns32 (0.1512)
# --curve로 train/val 간격도 같이 기록. 사용: bash sweep_drop.sh  (D0, D1 export 필요)
cd ~/RS
mkdir -p logs
grep -q "edge_drop" lightgcn_basic.py || { echo "lightgcn_basic.py에 edge_drop 없음 — 파일 교체 확인"; exit 1; }

run() { CUDA_VISIBLE_DEVICES=$1 python lightgcn_basic.py --data ~/RS/data --edge_drop $2 --curve \
          --tag lgcn_drop$2 > logs/lgcn_drop$2.log 2>&1; }
run $D0 0.1 & run $D1 0.2 & wait
run $D0 0.3 & run $D1 0.5 & wait

echo "---- 요약 ----"
echo "drop=0    0.1512 (dns32)"
for P in 0.1 0.2 0.3 0.5; do
  BEST=$(grep '\[val\]' logs/lgcn_drop$P.log | awk '{print $4}')
  LINE=$(grep "^ep *$BEST " logs/lgcn_drop$P.log | sed 's/ *(.*s)$//')
  echo "drop=$P  $(grep '\[val\]' logs/lgcn_drop$P.log | awk '{print $NF}')  | best: $LINE"
done
