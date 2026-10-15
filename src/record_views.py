"""Record the exact training views of an official baseline.py run: for each epoch and batch, the image indices
(train.csv row order) and their RandomResizedCrop box + flips, by iterating baseline.py's own DataLoader (seeded
generator, 2 workers, seed_worker) with a transform that draws the random numbers exactly like
RRC(224, .75-1) + RandomHorizontalFlip + RandomVerticalFlip but returns them instead of an image.
usage: record_views.py <set_dir> <seed> <out.pt>"""
import argparse, csv, json, sys
from pathlib import Path
import torch
from torchvision import transforms
sys.path.insert(0, 'ext/challenge_2026_public')
import baseline as B

set_dir, seed, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def draw(img):
    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.75, 1.0), ratio=(3 / 4, 4 / 3))
    hf = torch.rand(1).item() < 0.5          # RandomHorizontalFlip(p=.5): torch.rand(1) < p
    vf = torch.rand(1).item() < 0.5
    return torch.tensor([i, j, h, w, int(hf), int(vf)])


classes = json.loads((set_dir / 'classes.json').read_text())
data = B.Images(set_dir, 'train', classes, augment=True)
data.transform = draw
args = argparse.Namespace(batch_size=32, workers=2, device='auto', seed=seed)
torch.manual_seed(seed)
loader = B.loader(data, args, True)
rows = [r['filename'] for r in csv.DictReader(open(set_dir / 'train.csv'))]
epochs = []
for ep in range(5):
    batches = []
    for views, targets, names in loader:
        idx = torch.tensor([rows.index(nm) for nm in names])
        batches.append((idx, views))
    epochs.append(batches)
torch.save({'set': str(set_dir), 'seed': seed, 'epochs': epochs}, out)
print(json.dumps({'seed': seed, 'n': len(rows), 'batches_per_epoch': [len(b) for b in epochs],
                  'first_batch_views': epochs[0][0][1][:3].tolist()}))
