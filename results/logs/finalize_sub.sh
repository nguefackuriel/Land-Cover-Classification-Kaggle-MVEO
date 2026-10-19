#!/bin/bash
# set.pt -> PNG set -> official baseline.py seed 42 (Kaggle CSV) -> second official run, predictions must be identical
# -> archive exact PNGs + train.csv + sha256 manifest. usage: finalize_sub.sh <set.pt> <name>
set -e
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
SRC=$1; NAME=$2
$PY export_synth.py $SRC sets/$NAME
$PY make_submission.py sets/$NAME $NAME
$PY /home/ubuntu/Uriel/data_centric/ext/challenge_2026_public/baseline.py train --data sets/$NAME --output runs/recheck_$NAME > runs/recheck_$NAME.log 2>&1
cmp <(tail -n +2 runs/sub_$NAME/submission.csv) <(tail -n +2 runs/recheck_$NAME/submission.csv) && echo "RECHECK identical predictions"
rm -f runs/recheck_$NAME/best.pt runs/sub_$NAME/best.pt
mkdir -p submissions/${NAME}_images
cp sets/$NAME/train/*.png sets/$NAME/train.csv submissions/${NAME}_images/
(cd submissions/${NAME}_images && sha256sum *.png train.csv) > submissions/${NAME}_manifest.txt
echo "archived submissions/${NAME}_images"
