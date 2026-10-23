# Is the train split shifted from val? Accuracy of a near target (trained on POOL rows) on HOLD rows vs on val.
import os, sys, torch
os.environ['DD_HOLDOUT'] = '4096'
import dd_lib as D, proto as P
m, _, _ = P.fresh_model(42); m.eval()
print('sizes: train', len(D.xtr), 'val', len(D.xva), 'hold', len(D.HOLD))
for f in sys.argv[1:]:
    t = torch.load(f)
    with torch.no_grad():
        for k, p in m.named_parameters(): p.copy_(t[k].cuda())
    print(f.split('/')[-1], 'val', round(P.evaluate(m, D.xva, D.yva), 4), 'hold', round(P.evaluate(m, D.xtr[D.HOLD], D.ytr[D.HOLD]), 4), flush=True)
