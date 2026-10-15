"""Compare the test predictions of a baseline.py run with a submitted Kaggle csv (the one in submissions/<name>).
usage: python src/compare_predictions.py runs/final/submission.csv submissions/<name>/kaggle_submission.csv"""
import csv, sys

run = {r['filename']: r['label'] for r in csv.DictReader(open(sys.argv[1]))}
sub = {r['filename']: r['label'] for r in csv.DictReader(open(sys.argv[2])) if r['filename'] != '__num_train_images__'}
same = sum(run[k] == v for k, v in sub.items())
print(f'{same} / {len(sub)} test predictions identical' + (' (exact reproduction)' if same == len(sub) else ''))
