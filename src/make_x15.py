"""1.5x the max-norm PFD3 perturbation around the n=120 medoids (delta_mix.py's best combo), quantized."""
import sys, torch
import dd_lib as D
x0, y = D.as_x01(torch.load('cache/medoids_ipc12.pt'))
d = torch.load('cache/pfd3_n120_best.pt')
xq = ((x0 + 1.5 * (d['x01'] - x0)).clamp(0, 1) * 255).round() / 255
torch.save({'x01': xq, 'y': y, 'it': 'x1.5 of it250', 'accs': None}, 'cache/pfd3_n120_x15.pt')
