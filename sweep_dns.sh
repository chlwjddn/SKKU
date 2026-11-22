#!/bin/bash
# DNS 후보 수 M만 바꾸는 스윕 (나머지는 basic 고정). M=1(0.1408), 8(0.1451)은 완료.
# 사용: bash sweep_dns.sh   (D0 export 필요)
set -e
cd ~/RS
mkdir -p logs
grep -q "dns_m" lightgcn_basic.py || { echo "lightgcn_basic.py에 dns_m 없음 — 파일 교체 확인"; exit 1; }

for M in 16 32 64; do
  CUDA_VISIBLE_DEVICES=$D0 python lightgcn_basic.py --data ~/RS/data --dns_m $M --tag lgcn_dns$M \
    > logs/lgcn_dns$M.log 2>&1
  echo "M=$M  $(grep '\[val\]' logs/lgcn_dns$M.log)"
done

echo "---- 요약 ----"
echo "M=1   0.1408 (basic)"
echo "M=8   0.1451"
for M in 16 32 64; do
  echo "M=$M  $(grep '\[val\]' logs/lgcn_dns$M.log | awk '{print $NF, "(best ep", $4")"}')"
done
