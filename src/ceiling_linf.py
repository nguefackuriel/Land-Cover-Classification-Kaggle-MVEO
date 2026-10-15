"""Reachable-set ceiling: Adam's first steps are ~sign descent, so K official steps can move each weight by at
most ~sum(lr_t). Fine-tune inside that L-inf box (sign-PGD on real data) to see the best accuracy any
distilled set could possibly unlock at that step budget."""
import sys, torch, torch.nn.functional as F
import proto as P

xtr, ytr = P.load_split('train'); xva, yva = P.load_split('validation')
LR_SUM_5 = 3e-5 * (1 + 0.9045 + 0.6545 + 0.3455 + 0.0955)
for mult in [1, 2]:                      # 1 -> n<=32 (5 steps), 2 -> n in 33..64 (10 steps)
    R = LR_SUM_5 * mult
    m, _, _ = P.fresh_model(42)
    theta0 = [p.detach().clone() for p in m.parameters()]
    iters, bs = 150, 128
    for it in range(iters):
        idx = torch.randperm(len(xtr))[:bs]
        xb = P.augment(xtr[idx].cuda().permute(0, 3, 1, 2).float().div(255))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(m(xb), ytr[idx].cuda())
        m.zero_grad(set_to_none=True); loss.backward()
        step = R * 0.1 * (1 - it / iters) + R * 0.01
        with torch.no_grad():
            for p, p0 in zip(m.parameters(), theta0):
                p.sub_(step * p.grad.sign())
                p.copy_(torch.max(torch.min(p, p0 + R), p0 - R))
        if (it + 1) % 50 == 0:
            print(f'R={R:.2e} ({5*mult} steps) iter={it+1} loss={loss.item():.3f} val_acc={P.evaluate(m, xva, yva):.4f}', flush=True)
