"""Replica accuracy (float resize vs PROTO_PIL=1 PIL-faithful uint8 resize) of saved sets over seeds, to compare with
the official per-seed numbers. Run once per mode: PROTO_PIL=0|1 fidelity_pil_eval.py cache/a.pt cache/b.pt"""
import json, os, sys, torch
import dd_lib as D

SEEDS = (42, 0, 1, 2, 3, 4)
for f in sys.argv[1:]:
    x, y = D.load_set(f)
    accs = [round(D.train_eval(x, y, s), 4) for s in SEEDS]
    print(json.dumps({'set': f, 'pil': os.environ.get('PROTO_PIL', '0'), 'accs': accs,
                      'mean': round(sum(accs) / len(accs), 4)}), flush=True)
