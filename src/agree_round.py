"""What does one sign-matching round do to the net sign pattern? For the round's near target N, start state A (run end
of the start set) and end state B (run end of the round's set): disagreement of A and B with N, how many of A's
disagreements B fixed and how many new ones it made, corner accuracies of A, B and N, and the corner where exactly the
fixed weights take N's sign (same count of flips, but only fixes), to compare with the random-mix curve of corner_mix.py.
usage: DD_HOLDOUT=4096 PROTO_PIL=1 agree_round.py <near.pt> <start state.pt> <end state.pt>"""
import json, math, sys, torch
import proto as P, dd_lib as D
dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]; WD = 0.05
m, _, _ = P.fresh_model(42); m.eval()
for p in m.parameters():
    p.requires_grad_(False)
th0 = {k: p.detach().clone() for k, p in m.named_parameters()}


def signs(path):
    t = torch.load(path, map_location=dev)
    t = t.get('model', t) if isinstance(t, dict) and 'model' in t else t
    return {k: torch.sign(t[k].float() - th0[k]) for k in th0}


@torch.no_grad()
def corner_acc(S):
    sd = m.state_dict(); saved = {k: sd[k].clone() for k in th0}
    for k, t in th0.items():
        for lr in LRS:
            t = t * (1 - lr * WD) + lr * S[k]
        sd[k].copy_(t)
    acc = P.evaluate(m, D.xva, D.yva)
    for k, t in saved.items():
        sd[k].copy_(t)
    return round(acc, 4)


N, A, B = (signs(p) for p in sys.argv[1:4])
NT = sum(t.numel() for t in th0.values())
cnt = lambda f: sum(f(k).sum().item() for k in th0) / NT
dA = cnt(lambda k: A[k] != N[k]); dB = cnt(lambda k: B[k] != N[k])
fixed = cnt(lambda k: (A[k] != N[k]) & (B[k] == N[k])); broke = cnt(lambda k: (A[k] == N[k]) & (B[k] != N[k]))
moved = cnt(lambda k: A[k] != B[k])
only_fix = {k: torch.where((A[k] != N[k]) & (B[k] == N[k]), N[k], A[k]) for k in th0}
print(json.dumps({'dis_start': round(dA, 4), 'dis_end': round(dB, 4), 'fixed': round(fixed, 4), 'broke': round(broke, 4),
                  'start_vs_end': round(moved, 4), 'acc_start_corner': corner_acc(A), 'acc_end_corner': corner_acc(B),
                  'acc_near_corner': corner_acc(N), 'acc_start_plus_fixes': corner_acc(only_fix),
                  'fix_frac_of_start_dis': round(fixed / dA, 4)}), flush=True)
