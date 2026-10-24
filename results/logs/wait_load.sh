#!/bin/bash
# Waits until the machine is quiet (1-minute load below ${LOADMAX:-24} for 10 checks in a row, one per minute), then runs the
# given command. Our exact runs build every training batch on the CPU, so they crawl when other jobs fill the cores.
# usage: [LOADMAX=24] wait_load.sh <command...>   (the machine has 30 cores; our exact runs use about 8)
cd /home/ubuntu/Uriel/data_centric/work
ok=0
while [ $ok -lt 10 ]; do
  l=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$l" -v m="${LOADMAX:-24}" 'BEGIN{exit !(l < m)}'; then ok=$((ok+1)); else ok=0; fi
  sleep 60
done
echo "load low, start $* $(date +%T)" >> runs/night.log
exec "$@"
