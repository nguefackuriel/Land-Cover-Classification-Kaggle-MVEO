"""Are the losses of our runs (and of ranked-flip corners) a class-bias problem? For each model state: val accuracy,
predicted-class histogram vs truth, and val accuracy after per-class logit offsets fitted on the hold rows (CE at a
fitted temperature). A large gain means the trained network sorts the classes well but with shifted class priors.
usage: DD_HOLDOUT=4096 PROTO_PIL=1 bias_diag.py <state1.pt> [<state2.pt> ...]"""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D
dev = 'cuda'
m, _, _ = P.fresh_model(42); m.eval()
HR = D.HOLD if len(D.HOLD) else torch.randperm(len(D.xtr), generator=torch.Generator().manual_seed(33))[:8192]


@torch.no_grad()
def logits(xu8):
    return torch.cat([m(P.to_eval(xu8[j:j + 256].to(dev))).float() for j in range(0, len(xu8), 256)])


for path in sys.argv[1:]:
    sd = torch.load(path, map_location=dev)
    sd = sd.get('model', sd) if isinstance(sd, dict) and 'model' in sd else sd
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=False)
    lv, lh = logits(D.xva), logits(D.xtr[HR])
    yv, yh = D.yva.to(dev), D.ytr[HR].to(dev)
    Ts = torch.logspace(-3, 1, 81, device=dev)
    T = Ts[torch.stack([F.cross_entropy(lh / t, yh) for t in Ts]).argmin()].item()
    b = torch.zeros(lh.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=200, line_search_fn='strong_wolfe')

    def clo():
        opt.zero_grad(); l = F.cross_entropy(lh / T + b, yh); l.backward(); return l
    opt.step(clo)
    acc = lambda l, y: round((l.argmax(1) == y).float().mean().item(), 4)
    bv = lv / T + b.detach()
    print(json.dumps({'state': path, 'T': round(T, 4), 'val': acc(lv, yv), 'val_offset': acc(bv, yv),
                      'hold': acc(lh, yh), 'hold_offset': acc(lh / T + b.detach(), yh),
                      'pred_hist': torch.bincount(lv.argmax(1), minlength=10).tolist(),
                      'true_hist': torch.bincount(yv, minlength=10).tolist(),
                      'offsets': [round(x, 2) for x in b.detach().tolist()]}), flush=True)
