#!/bin/bash
# 최근성 가중 tw만 바꾸는 스윕 (dns32, 평점≥4 고정). tw=0 = dns32 (0.1512)
# 사용: bash sweep_tw.sh   (D0, D1 export 필요)
cd ~/RS
mkdir -p logs
grep -q "a.tw" lightgcn_basic.py || { echo "lightgcn_basic.py에 tw 없음 — 파일 교체 확인"; exit 1; }

run() { CUDA_VISIBLE_DEVICES=$1 python lightgcn_basic.py --data ~/RS/data --tw $2 --tag lgcn_tw$2 > logs/lgcn_tw$2.log 2>&1; }
run $D0 0.5 & run $D1 1 & wait
run $D0 2   & run $D1 3 & wait

echo "---- 요약 (최고/최신 가중 비) ----"
echo "tw=0    0.1512 (dns32)"
for T in 0.5 1 2 3; do
  echo "tw=$T  $(grep '\[val\]' logs/lgcn_tw$T.log | awk '{print $NF, "(best ep", $4")"}')"
done
