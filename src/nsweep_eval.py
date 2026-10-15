"""Proto accuracy of medoid sets over 6 seeds, to locate the score optimum inside the 20-step band (n = 97..128).
usage: nsweep_eval.py 104 108 112 116 120      (reads cache/medoids_n{n}.pt; 120 = cache/medoids_ipc12.pt)
       nsweep_eval.py cache/medoids_n112_km1.pt  (any index file)"""
import json, sys, torch
import dd_lib as D

for arg in sys.argv[1:]:
    path = arg if arg.endswith('.pt') else ('cache/medoids_ipc12.pt' if arg == '120' else f'cache/medoids_n{arg}.pt')
    idx = torch.load(f'{path}')
    n = len(idx)
    x, y = D.as_x01(idx)
    accs = [round(D.train_eval(x, y, s), 4) for s in (42, 0, 1, 2, 3, 4)]
    mean = sum(accs) / len(accs)
    print(json.dumps({'set': path, 'n': n, 'last_batch': n % 32 or 32, 'accs': accs, 'mean': round(mean, 4),
                      'score_mean': round(D.score(mean, n), 4)}), flush=True)
