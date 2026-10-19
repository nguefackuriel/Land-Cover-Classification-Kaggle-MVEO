"""Val accuracy of sign-projected versions of a reachable network: theta0 + a*R*sign(theta*-theta0) for a in (.5,.8,1),
and of our trajectory end point projected the same way. usage: corner_eval.py <target.pt> [<ours.pt>]"""
import sys, torch
import proto as P, dd_lib as D
R = 9e-5
m, _, _ = P.fresh_model(42)
th0 = {k: p.detach().clone() for k, p in m.named_parameters()}
for path in sys.argv[1:]:
    sd = torch.load(path)
    for a in (None, 0.5, 0.8, 1.0):
        with torch.no_grad():
            for k, p in m.named_parameters():
                t = sd[k].cuda()
                p.copy_(t if a is None else th0[k] + a * R * torch.sign(t - th0[k]))
        print(path.split('/')[-1], 'as is' if a is None else f'corner a={a}', round(P.evaluate(m, D.xva, D.yva), 4), flush=True)
