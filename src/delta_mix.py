"""Perturbation arithmetic around the medoid init: x0 + combos of the PFD deltas (quantized), proto acc over 3 seeds."""
import json, sys, torch
import dd_lib as D
x0, y = D.as_x01(torch.load('cache/medoids_ipc12.pt'))
d1 = torch.load('cache/pfd3_n120_best.pt')['x01'] - x0           # max-norm optimum (it 250)
d2 = torch.load('cache/pfd3_n120_adam025_best.pt')['x01'] - x0   # pixel-Adam optimum (it 125)
print(json.dumps({'cos_d1_d2': round(torch.nn.functional.cosine_similarity(d1.flatten(), d2.flatten(), dim=0).item(), 3),
                  'mean_abs_d1': round(d1.abs().mean().item() * 255, 3), 'mean_abs_d2': round(d2.abs().mean().item() * 255, 3)}))
for name, d in [('d1', d1), ('d2', d2), ('(d1+d2)/2', (d1 + d2) / 2), ('d1+d2', d1 + d2), ('1.5*d1', 1.5 * d1), ('2*d1', 2 * d1)]:
    xq = ((x0 + d).clamp(0, 1) * 255).round() / 255
    accs = [round(D.train_eval(xq, y, s), 4) for s in (42, 0, 1)]
    print(json.dumps({'combo': name, 'accs': accs, 'mean': round(sum(accs) / 3, 4)}), flush=True)
