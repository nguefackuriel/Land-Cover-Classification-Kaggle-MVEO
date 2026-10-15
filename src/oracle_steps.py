"""Upper-bound probe: how far can K AdamW steps (official lr schedule) go if each step used a large
real batch instead of the distilled one? steps_per_epoch=1 mimics n<=32, 2 mimics n<=64, etc."""
import random, sys, time, torch
import proto as P

xtr, ytr = P.load_split('train'); xva, yva = P.load_split('validation')
big = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
for seed in [42, 0]:
    for spe in [1, 2, 4]:
        m, opt, sch = P.fresh_model(seed)
        accs = []
        for ep in range(5):
            for _ in range(spe):
                idx = torch.randperm(len(xtr))[:big]
                opt.zero_grad(set_to_none=True)
                for j in range(0, big, 64):
                    xb = P.augment(xtr[idx[j:j + 64]].cuda().permute(0, 3, 1, 2).float().div(255))
                    loss = torch.nn.functional.cross_entropy(m(xb), ytr[idx[j:j + 64]].cuda()) * len(xb) / big
                    loss.backward()
                opt.step()
            sch.step()
            accs.append(round(P.evaluate(m, xva, yva), 4))
        print(f'seed={seed} steps/epoch={spe} big_batch={big} val_acc_per_epoch={accs}', flush=True)
