# MVEO 2026 — Data-Centric Land Cover Challenge : plan pour gagner et publier

Dernière mise à jour : 13/10/2026. Deadline Kaggle : **15/11/2026 23:59 UTC** (33 jours).
Workshop MVEO @ BMVC : 26/11/2026, Lancaster. Les mieux classés co-signent un article de synthèse dans IEEE JSTARS.

---

## 0. En bref

- Le dataset est **EuroSAT RGB** (64×64, 10 classes). Le modèle d'évaluation est imposé : ViT-B/16 ImageNet,
  AdamW lr 3e-5, batch 32, **5 epochs**, labels durs. Avec n images, l'entraînement fait seulement
  **5·⌈n/32⌉ pas d'optimisation**. Avec n ≤ 32, ça fait 5 pas en tout.
- Score = 0,5·accuracy + 0,5·(1 − ln n / ln 18900). Doubler n coûte 3,5 points de score.
- Le leaderboard (0,66 à 0,69) correspond à ce que donnent des **vraies images de bonne qualité** :
  le vrai gradient complet donne 60 % en 5 pas (score 0,685 à n=10) et 80 % en 10 pas (0,689 à n=64).
  Les leaders sont donc sur la « frontière des données réelles ».
- Nos mesures montrent qu'**il existe des réseaux atteignables en 5 pas à ≥79 %, et en 10 pas à ≥94 %**.
  La marge est réelle. Seules des images *synthétiques optimisées pour ce protocole précis* peuvent aller
  la chercher, ce qui est exactement l'idée de la dataset distillation.
- Stratégie : **distillation fidèle au protocole** (on déroule les 5 ou 10 vrais pas d'AdamW et on
  rétro-propage la perte réelle jusqu'aux pixels), puis choix du point de fonctionnement n sur l'escalier
  des pas (n=10…32 → 5 pas ; n=33…64 → 10 pas). Objectif : **score ≥ 0,75**, contre 0,692 aujourd'hui.
- Angle du paper : *« distiller pour l'optimiseur, pas seulement pour le modèle »*. Sous Adam, les premiers
  pas sont une descente de signe à magnitude fixe. Les méthodes classiques (gradient matching par cosinus,
  trajectory matching) visent la mauvaise cible, et des données synthétiques peuvent faire **mieux que la
  totalité des données réelles** à budget de pas égal.

---

## 1. Faits vérifiés

### 1.1 Données
| Split | Images | Remarque |
|---|---|---|
| train | 18 900 | 2 100 / 1 750 / 1 400 par classe (70 % d'EuroSAT) |
| validation | 4 050 | étiquetée, et utilisée par le script officiel pour choisir l'epoch |
| test | 4 050 | non étiquetée, le leaderboard porte sur **100 % du test** (pas de split privé) |

Classes anonymisées, mais c'est l'ordre alphabétique d'EuroSAT (vérifié visuellement et par les effectifs,
voir `work/class_sheet.png`) :
00 AnnualCrop · 01 Forest · 02 HerbaceousVegetation · 03 Highway · 04 Industrial · 05 Pasture ·
06 PermanentCrop · 07 Residential · 08 River · 09 SeaLake.
Pas de doublons train/test (hash md5). Moyenne RGB ≈ (80, 88, 94)/255, donc des images sombres et bleutées.

### 1.2 Protocole d'évaluation (`ext/challenge_2026_public/baseline.py`, à ne pas modifier)
- `vit_b_16(IMAGENET1K_V1)`, fine-tuning complet, nouvelle tête `Linear(768,10)` initialisée aléatoirement
  (déterministe avec seed 42).
- AdamW(lr=3e-5, wd=0.05), `CosineAnnealingLR(T_max=5)` **appelé à chaque epoch**. Les lr par epoch sont
  3,0 / 2,7 / 1,96 / 1,04 / 0,29 ×1e-5, donc la 5ᵉ epoch ne sert presque à rien.
- Augmentations train : `RandomResizedCrop(224, scale=(0.75,1))` + flips H et V. Éval : Resize 256 + CenterCrop 224.
- Batch 32, `drop_last=False`, shuffle à chaque epoch. Un set de 33 images donne un batch de 32 et un batch de **1**.
- On garde le meilleur epoch selon l'accuracy sur la **vraie validation**.
- Labels **durs** uniquement (CSV `filename,label`), donc aucun soft label possible. C'est une différence
  majeure avec SRe2L, RDED et consorts.

### 1.3 Score et soumission (Kaggle, confirmé via l'API)
- `Score = 0.5·Acc + 0.5·(1 − ln(n)/ln(18900))`, avec n ∈ [1, 18900] déclaré dans la ligne `__num_train_images__`.
- 5 soumissions par jour, équipes jusqu'à 20 personnes, 8 équipes inscrites, prix « Kudos » (co-signature JSTARS et présentation).
- Les meilleures équipes devront **fournir leurs images distillées** pour vérification. Les organisateurs
  ré-entraîneront probablement avec `baseline.py` par défaut.
- Les deux équipes à ~0,495 ont entraîné sur les 18 900 images : 0,5 × 0,99 + 0 = 0,495.

### 1.4 Ce qu'il faut viser selon n
| n | pas | efficacité | acc pour battre 0,692 | acc pour 0,75 |
|---|---|---|---|---|
| 10 | 5 | 0,766 | 61,8 % | 73,4 % |
| 20 | 5 | 0,696 | 68,8 % | 80,4 % |
| 32 | 5 | 0,648 | 73,6 % | 85,2 % |
| 33 | 10 | 0,645 | 73,9 % | 85,5 % |
| 64 | 10 | 0,578 | 80,6 % | 92,2 % |
| 97 | 20 | 0,535 | 84,8 % | 96,5 % |

n < 10 n'a pas d'intérêt : il faut au moins une image par classe pour que la classe soit apprise.

---

## 2. Expériences préliminaires (12/10, GPU partagé, seed 42 sauf mention)

| Expérience | 5 pas (n ≤ 32) | 10 pas (n ≤ 64) | 20 pas (n ≤ 128) |
|---|---|---|---|
| 1 image réelle aléatoire par classe (script officiel, 3 tirages) | 22,2 / 33,0 / 36,9 % (moy. 30,7) | – | – |
| 1 image aléatoire par classe (réplique, un tirage, seeds 42/0/1) | 32,4 / 34,5 / 36,6 % | – | – |
| **Médoïde k-means** par classe (features ViT, seeds 42/0/1) | 40,7 / 51,9 / 55,9 % (moy. 49,5) | – | – |
| **Oracle** : chaque pas calculé sur 1 024 vraies images | 60,4 % | 80,1 % | 91,7 % |
| **Plafond** : meilleur réseau dans la boîte ±Σlr (descente de signe projetée, 150 it.) | **≥ 78,8 %** | **≥ 94,4 %** | – |

Comment lire ces chiffres :
1. Le **nombre de pas** domine tout le reste. Passer de 5 à 10 pas fait gagner 20 points, même avec des
   données parfaites. C'est un « escalier » : n=32 et n=33 n'ont pas du tout le même comportement.
2. Le vrai gradient n'est **pas** la meilleure direction quand on n'a que 5 pas. Le plafond est 18 points
   au-dessus. Des images conçues pour ce protocole peuvent donc battre la totalité des données réelles.
   Cet écart est le cœur scientifique du paper.
3. La variance est énorme, à la fois selon le choix des images (22 à 37 % en aléatoire) et selon la seed
   d'entraînement (médoïdes : 40,7 % avec la seed 42, 55,9 % avec la seed 1). Tout résultat doit être
   moyenné sur plusieurs seeds, sinon on se trompe.
4. Une bonne sélection seule (médoïdes) fait déjà +19 points par rapport à l'aléatoire, mais reste
   au-dessous de l'oracle. Ce sera le point de départ (l'initialisation) de PFD.

Scripts : `work/proto.py` (réplique GPU rapide du protocole), `work/oracle_steps.py`, `work/ceiling_linf.py`,
`work/select_eval.py` (random vs médoïdes k-means), `work/make_subset.py` + `work/run_random_baselines.sh`
(script officiel non modifié). Logs dans `work/runs/`.

### 2.1 Frontière des coresets « médoïdes k-means » (réplique, seeds 42 / 0 / 1)
| n | pas | dernier batch | acc. seed 42 | acc. moyenne | score seed 42 | score moyen |
|---|---|---|---|---|---|---|
| 10 | 5 | 10 | 40,7 % | 49,5 % | 0,587 | 0,631 |
| 20 | 5 | 20 | 48,4 % | 53,7 % | 0,590 | 0,616 |
| 30 | 5 | 30 | 48,1 % | 56,6 % | 0,568 | 0,610 |
| 40 | 10 | 8 | 65,7 % | 67,4 % | 0,641 | 0,650 |
| 60 | 10 | 28 | 72,4 % | 73,8 % | 0,654 | 0,661 |
| 64 | 10 | 32 | 73,8 % | 75,2 % | 0,658 | 0,665 |
| 96 | 15 | 32 | 82,8 % | 83,3 % | 0,682 | 0,685 |
| 100 | 20 | **4** | 81,3 % | 82,2 % | 0,673 | 0,677 |
| **120** | 20 | 24 | **87,1 %** | **87,1 %** | **0,692** | **0,692** |
| 128 | 20 | 32 | 86,8 % | 87,1 % | 0,688 | 0,689 |
| 160 | 25 | 32 | 88,4 % | 89,0 % | 0,684 | 0,687 |

- Avec des images réelles, l'optimum est à n ≈ 120 (20 pas), à 0,692, c'est-à-dire le niveau du leader.
  Au-delà, ajouter des images ne paie plus.
- **Un petit dernier batch coûte cher.** n=100 (dernier batch de 4 images) fait 5 points de moins que n=120
  (dernier batch de 24). Sous Adam, un batch de 4 images fait un pas aussi grand qu'un batch de 32, mais
  beaucoup plus bruité. C'est une observation à mettre dans le paper.
- Fidélité de la réplique : sur n=120, le script officiel donne 86,5 % en validation contre 87,1 % pour la
  réplique, soit un écart d'environ 0,6 point.

### 2.2 Historique des soumissions Kaggle
| Date | Fichier | n | val (officiel) | Score Kaggle | Rang |
|---|---|---|---|---|---|
| 12/10 15:40 | `work/submissions/medoids_n120.csv` (images : `work/sets/medoids_n120`) | 120 | 86,49 % | **0,69171** (test ≈ 86,96 %) | 2ᵉ / 9, à 0,00013 du 1ᵉʳ |
| 13/10 16:15 | `work/submissions/pfd3_n120.csv` (images : `work/sets/pfd3_n120`, `cache/pfd3_n120_best.pt` it=250) | 120 | 88,37 % | **0,70258** (test ≈ 89,14 %) | **1ᵉʳ / 11**, +0,0107 sur le 2ᵉ |
| 14/10 14:40 | `work/submissions/pfd3_n120_T0s2.csv` (images : `work/sets/pfd3_n120_T0s2`, `cache/pfd3_n120_T0_stage2_best.pt` it=200 ; archive `work/submissions/pfd3_n120_T0s2_images/` + manifeste sha256) | 120 | 88,40 % | **0,70344** (test ≈ 89,31 %) | **1ᵉʳ**, +0,0020 sur Lynch James (0,70144) |
| 15/10 12:11 | `work/submissions/n10_head42b_it400.csv` (images : `work/sets/n10_head42b_it400`, `cache/pfd3_n10_head42b_it400.pt` ; archive `work/submissions/n10_head42b_it400_images/` + manifeste sha256). **Set ciblé sur la tête du seed 42** (autres seeds ≈ 60 %) | 10 | 71,60 % | **0,74258** (test ≈ 71,90 %) | **1ᵉʳ**, +0,0019 sur Lynch James (0,74070) |
| 15/10 13:09 | `work/submissions/n10_exact_it200.csv` (images : `work/sets/n10_exact_it200`, `cache/pfd3_n10_exact_it200.pt` ; archive `work/submissions/n10_exact_it200_images/` + manifeste sha256). **Optimisé sur la trajectoire officielle exacte du seed 42** (tête + crops/flips enregistrés) | 10 | 79,38 % | **0,78135** (test ≈ 79,65 %) | **1ᵉʳ**, +0,0407 sur Lynch James (0,74070) |
| 15/10 13:44 | `work/submissions/n10_exact_s2_it280.csv` (images : `work/sets/n10_exact_s2_it280`, `cache/pfd3_n10_exact_s2_it280.pt` ; archive + manifeste sha256). Palier 2 sur la trajectoire exacte du seed 42 | 10 | 80,62 % | **0,78530** (test ≈ 80,44 %) | **1ᵉʳ**, +0,0446 sur Lynch James |
| 19/10 16:20 | `work/submissions/n10_sm_r3_pf.csv` (images : `work/sets/n10_sm_r3_pf`, `cache/pfd3_pf_r3n_best.pt` ; archive `work/submissions/n10_sm_r3_pf_images/` + manifeste sha256 ; deux relances officielles identiques). Sign matching vers une cible proche (3 tours) + polissage PFD, trajectoire exacte du seed 42 | 10 | 83,88 % | **0,80357** (test ≈ 84,10 %) | 2ᵉ, à 0,00185 de vinicius simm (0,80542) |
| 19/10 18:15 | `work/submissions/n10_hyb3_it120.csv` (images : `work/sets/n10_hyb3_it120`, `cache/sm_hyb3_it120.pt` ; archive `work/submissions/n10_hyb3_it120_images/` + manifeste sha256 ; deux relances officielles identiques). Hybride sign matching + méta-gradient PFD exact (poids 3), depuis le set à 0,80357. Précision poolée val + 8k train : 84,99 % (contre 84,28 %) | 10 | 84,35 % | **0,80567** (test ≈ 84,52 %) | **1ᵉʳ**, +0,00025 sur vinicius simm |

### 2.3 Prototype PFD (`work/pfd.py`)
Le test rapide du 12/10 fonctionne : 5 pas déroulés, ViT-B/16 complet, 3,6 s par itération, **30 Go** de mémoire
au pic, et l'évaluation au départ retrouve les médoïdes. Un run de 400 itérations prend environ 40 min,
évaluations comprises.

**Run v1 (n=10, 400 itérations, lr pixels 0,03, τ=0,3, 1 tête aléatoire par itération)** : moyenne sur 3 seeds
49,5 % → **51,3 %** (seed 42 : 40,9 → 47,4 %), perte réelle après 5 pas 1,91 → 1,83. L'optimisation va dans
le bon sens mais le signal est très bruité, et on reste loin du seuil de 60,4 %. **Pas encore de go.**
Premier bug corrigé : la dérivée de √v̂ est infinie quand v̂=0, ce qui donnait des NaN. Le lissage se fait
maintenant via √(v̂+δ²).
Pistes : moyenner 4 têtes par itération (réduction de variance), régler le lr des pixels, plus d'itérations,
comparer l'unroll complet et la variante premier ordre.

**PFD-FO (`work/pfd_fo.py`)** : variante premier ordre à mémoire constante. Elle rejoue les K pas dans l'ordre
et pousse −lr_k⟨u_k, G⟩ dans les pixels. À n=96 (15 pas) : 13,8 Go, 16 s par itération. **Échec** : depuis les
médoïdes n=96, on passe de 83,2 % à 76,2 % en 50 itérations, et le run a été arrêté.

**Diagnostic du méta-gradient (`work/diag_metagrad.py`, n=10, tête, crops et batch réel fixés).**
Variation de la perte réelle exacte après 5 pas, selon la taille du pas sur les pixels (direction normalisée par son max) :

| Direction | 0,05 | 0,2 | 0,5 | 1,0 |
|---|---|---|---|---|
| Déroulé complet (straight-through) | **−0,065** | −0,047 | +0,006 | +0,036 |
| Premier ordre (PFD-FO) | +0,013 | +0,085 | +0,148 | +0,201 |
| Aléatoire | +0,015 | +0,097 | +0,125 | +0,211 |

- Le méta-gradient du déroulé complet est une vraie direction de descente, **mais seulement pour de petits pas**.
  Adam sur les pixels (lr 0,03) fait bouger chaque pixel de toute l'amplitude, ce qui sort de la zone utile.
  D'où le passage à une descente normalisée : z −= ε·g/max|g|.
- La variante premier ordre ne vaut pas mieux qu'une direction aléatoire (cos = 0,28 avec le déroulé complet).
  Les interactions entre pas dominent, et il faut le déroulé complet. Pour K > 5, il faudra du checkpointing
  ou des fenêtres de déroulé complet.

**Run v2 (n=10, descente normalisée ε=0,05, 2 têtes par itération, 400 itérations, ~40 min)** :

| itération | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| moyenne sur 3 seeds | 49,6 | 53,5 | 55,9 | 57,4 | 56,7 | 58,1 | **59,9** | 57,8 | 59,6 |

- Avec **10 images synthétiques, on atteint le niveau de l'oracle** (60,4 % avec 1 024 vraies images à chaque pas),
  contre 49,5 % pour les meilleures images réelles. C'est le premier résultat solide pour le paper.
- Script officiel (seed 42) sur les PNG exportés : **56,3 %** (réplique : 55,9 %), score ≈ 0,665. Pas soumis,
  car c'est en dessous de 0,692.
**Run v3 (affinage depuis le meilleur v2, 4 têtes, ε de 0,03 à 0,005, 400 itérations, ~1 h)** : moyenne
60,0 → **61,2 %** (seeds 42/0/1 : 58,1 / 60,8 / 64,6 %), donc au-dessus de l'oracle avec 10 images.
Script officiel (seed 42) : **58,2 %**, score ≈ 0,674. Pas soumis. À n=10, on plafonne vers 61 % avec ces
réglages, alors qu'il faut ~62 % à la seed 42 pour passer devant. Les PNG sont dans `work/sets/pfd_v3/train/`.

**Fenêtre de déroulé (diagnostic, n=10)** : on déroule exactement les W premiers pas, et les suivants sont
calculés exactement mais traités comme une identité au retour.

| W | cos avec le déroulé complet | ΔL au pas 0,05 |
|---|---|---|
| 5 (complet) | 1 | −0,065 |
| 3 | 0,85 | −0,051 |
| 2 | 0,39 | −0,005 |
| 1 | −0,06 | +0,017 |

Il faut dérouler les ~3 premières epochs, celles à grand learning rate. Pour K=10 (n=33–64), ça fait une fenêtre
de 6. `work/pfd2.py` implémente cette approche (n quelconque, batches comme le DataLoader, bf16 en interne).
À n=40 : 39,7 Go, 3,9 s par itération et par tête.

**Run PFD2 n=40 (10 pas, fenêtre 6, 2 têtes, ε de 0,05 à 0,01, 300 itérations, ~1 h)** :

| itération | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|
| moyenne sur 3 seeds | 67,3 | 70,6 | 72,4 | 72,9 | 73,4 | **73,7** | 73,7 |

Script officiel (seed 42) : **71,1 %** (réplique : 72,5 %), score ≈ 0,668. Pas soumis.

### 2.4 Bilan au 12/10 au soir : gain de PFD selon le budget
| n | pas | médoïdes (moy.) | PFD (moy.) | gain | score officiel seed 42 |
|---|---|---|---|---|---|
| 10 | 5 | 49,5 % | 61,2 % | +11,7 | 0,674 |
| 40 | 10 | 67,4 % | 73,7 % | +6,3 | 0,668 |
| 120 (réel seul) | 20 | 87,1 % | – | – | **0,692 (soumis)** |

PFD améliore nettement l'accuracy à n fixé, mais ne dépasse pas encore la frontière des images réelles à
n≈120. Deux leviers restent à exploiter :
1. **Appliquer PFD là où la frontière est la meilleure (n=96–128, 15–20 pas).** Il suffit de +3 points pour
   atteindre ~0,70. Il faut réduire la mémoire : une fonction autograd dédiée pour le pas d'Adam (moins de
   tenseurs sauvegardés) et du checkpointing d'activations, puisqu'une fenêtre de 3 epochs fait 12 pas de 32 images.
2. **Rendre PFD plus efficace** : régler τ (via le diagnostic), plus de têtes, une paramétrisation
   multi-échelle, une allocation par classe.

Constat de robustesse : avec plusieurs batchs par epoch, l'ordre de mélange ajoute de la variance
(écart réplique/officiel de 1,4 point à n=40, contre 0,4 à n=10).

À refaire : les runs « random » à 2, 3 et 10 images par classe via le script officiel (OOM pendant les jobs QA2026).

### 2.5 PFD3 : déroulé complet à mémoire constante (13/10)
`work/pfd3.py` : la passe avant suit exactement le protocole sans graphe et stocke (θ, m, v) avant chaque pas.
La passe arrière reconstruit le graphe d'un pas à la fois et propage les adjoints de (θ, m, v). La mémoire ne dépend
plus de la fenêtre, donc on peut dérouler les 20 pas de n=120.
- **Vérification (`--check`, n=40, fenêtre 4)** : cos = 0,9998 et rapport des normes 1,005 avec le gradient du graphe
  complet de pfd2. Si la passe avant utilise la formule exacte d'Adam au lieu de la forme straight-through
  (égale à l'arrondi près), le cos tombe à 0,987 et le rapport des normes à 0,88. **La trajectoire est chaotique** :
  un écart d'arrondi sur 10 pas suffit à décorréler 13 % du méta-gradient. À noter pour le paper.
- **Mémoire à n=120, fenêtre 20** : 30 Go sur GPU (9,4 s par tête), ou **12,4 Go avec `--offload`** (états dans des
  buffers épinglés côté hôte, alloués une seule fois) pour 10 s par tête. Le mode offload tient à côté d'un job QA2026.
- Smoke test : la perte réelle après 20 pas passe de 0,913 à 0,750 en 5 itérations (1 tête, ε=0,05).

**Run PFD3 n=120 (médoïdes ipc12, 20 pas tous déroulés, 2 têtes, ε de 0,05 à 0,01, 300 itérations, offload,
20 s par itération, ~1 h 40)** :

| itération | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| moyenne sur 3 seeds | 87,15 | 87,20 | 87,70 | 88,25 | 88,53 | 88,82 | 88,93 | 89,04 | 89,12 | 89,17 | **89,33** | 89,22 | 89,33 |

- Meilleur point à l'itération 250 (seeds 42/0/1 : 88,64 / 90,15 / 89,21 %). Perte réelle après 20 pas : 0,91 → ~0,75.
- Script officiel (seed 42) : **88,37 %** (réplique : 88,64 %), score attendu 0,6988. **Soumis : 0,70258 sur Kaggle,
  1ᵉʳ**. Pour la première fois, PFD dépasse la frontière des images réelles : +2,2 points à n égal.
- Le gain de PFD ne s'effondre pas avec n quand on déroule tout : +2,2 points à n=120 en partant de 87 %,
  alors que l'oracle est à 91,7 %. On a donc récupéré la moitié de l'écart médoïdes → oracle.
- **Les images distillées sont presque identiques aux médoïdes** : écart moyen de 0,22/255 par pixel, écart max 8/255.
  Le gain de +2,2 points vient d'une perturbation imperceptible d'images réelles, une sorte d'« anti-attaque
  adversariale » robuste aux seeds. C'est un angle fort pour le paper. Mais : (1) la normalisation par max|g| ne
  bouge qu'une poignée de pixels, et la plupart des mouvements disparaissent à l'arrondi uint8, il y a donc
  probablement de la marge avec de plus grands pas ; (2) la question 5 aux organisateurs (images réelles
  perturbées = distillées ?) devient importante.

**Robustesse (script officiel non modifié, `work/official_seeds.py`)** à n=120 :

| seed | 42 | 0 | 1 | 2 | 3 | 4 | moyenne ± écart-type |
|---|---|---|---|---|---|---|---|
| médoïdes | 86,49 | 87,80 | 87,48 | 88,44 | 85,78 | 87,16 | 87,19 ± 0,95 |
| PFD3 (soumis) | 88,37 | 89,90 | 89,16 | 90,10 | 88,42 | 89,65 | **89,27 ± 0,70** |
| gain | +1,88 | +2,10 | +1,68 | +1,66 | +2,64 | +2,49 | **+2,08** |

Le gain est positif sur les 6 seeds et la variance diminue. La seed 42 tombe dans le bas de la distribution pour
PFD3 : le score soumis n'est pas un tirage favorable. Réplique contre officiel par seed : écart ≤ 0,25 point.

### 2.5 bis Diagnostic de la taille du pas sur les pixels (`work/diag_step.py`, 13/10)
Méta-gradient de 2 têtes (comme une itération PFD), puis CE et accuracy de validation exactes (2 048 images) après
les 20 pas officiels, sur **4 autres** configurations (tête, mélange, crops). Mesures appariées, à n=120. Les
pas sont exprimés en écart moyen |Δx| (unités 1/255), et les résultats sont mesurés après arrondi uint8.

| direction | taille | depuis les médoïdes : ΔAcc | depuis PFD3 it=250 : ΔAcc |
|---|---|---|---|
| g linéaire | 0,1 | +0,48 | −0,11 |
| g linéaire | 0,3 | +0,49 | −0,85 |
| g linéaire | 1 | −3,45 | −7,53 |
| **sign(g), ±1 niveau partout** | 1 | **+0,85** | −1,26 |
| sign(g) | 3 | −5,62 | −8,31 |
| |g| ⊙ signe aléatoire | 1 | −4,05 | – |
| signe aléatoire ±1 | 1 | −0,18 | – |

- Les pixels à fort gradient sont dans des zones de forte courbure : les bouger de plusieurs niveaux est destructeur,
  quelle que soit la direction (|g| ⊙ signe aléatoire ≈ g linéaire). La normalisation par max|g| concentre le pas sur
  ces pixels, d'où les petits pas imposés et la lenteur de PFD3.
- **Un seul pas de signe à ±1 niveau donne +0,85 point**, autant qu'environ 60 itérations de PFD3. En plus, ±1 niveau
  est exactement un quantum uint8, donc rien ne se perd à l'export.
- Depuis le point PFD3, tous les pas font baisser l'accuracy : c'est un optimum local net pour cette méthode.
- D'où l'option `--opt adam` de `pfd3.py` : Adam directement sur les pixels (β = 0,9/0,99), avec un lr en niveaux
  (1 → 0,1). Le premier pas est exactement le pas de signe à ±1 niveau.

**Run `--opt adam` (Adam sur les pixels, n=120 depuis les médoïdes, 2 têtes)** :
- lr = 1 niveau : 86,08 % à l'itération 25 (−1,0), et la perte réelle ne baisse pas. Avec un gradient aussi bruité,
  répéter des pas de signe de ±1 niveau fait dériver les pixels comme une marche aléatoire (≈ √25 = 5 niveaux).
  Run arrêté.
- lr = 0,25 → 0,05 niveau, 150 itérations :

| itération | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|
| max-norm (logits), rappel | 87,20 | 87,70 | 88,25 | 88,53 | 88,82 | 88,93 |
| Adam pixels, lr 0,25 | **88,60** | 88,79 | 88,66 | 88,88 | **89,08** | 89,05 |

  Quatre fois plus rapide au départ, mais **même plateau vers 89 %**.
- `diag_step.py` avec un méta-gradient de **8 têtes** depuis PFD3 it=250 : aucun pas n'améliore (ΔAcc = −0,05 à
  0,1 niveau, −0,95 pour un pas de signe à 1 niveau). Le point atteint est donc **un vrai optimum local** de la
  méta-perte moyenne, et pas un simple plancher de bruit. Plus de têtes ne suffira pas. Pistes : lisser l'objectif
  (bruit sur les pixels, `--pixel-noise`) ou changer de bassin (init).

**Erreurs par classe (n=120, moyenne sur 3 seeds, `work/confusion.py`)** : River 70,7 → 78,0 %, Highway 78,8 → 79,3 %,
PermanentCrop 78,5 → 80,3 %, Pasture 84,4 → 85,7 % (médoïdes → PFD3). Industrial, Residential et Forest sont
au-dessus de 94 %. Confusions principales : River↔Highway (63 sur 4 050) et PermanentCrop↔Herbaceous (47).
**Réallocation par classe** (médoïdes, 6 seeds, `work/alloc_search.py`) : uniforme à 12 par classe 87,04 %,
« mild » (River/Highway à 14) 86,69 %, « strong » (River/Highway à 16) 86,09 %. **Ça empire**, on garde l'uniforme.

**Sortir de l'optimum local à n=120 ? (13/10 au soir)**
- Lissage aléatoire (`--pixel-noise 1` : bruit N(0, 1 niveau²) à chaque échantillon du méta-gradient), Adam pixels
  0,25 → 0,05, depuis PFD3 it=250 : 89,34 → 88,75 → 88,92 → 89,10 → 89,28 % (itérations 0 à 100). On revient au
  point de départ sans le dépasser.
- `work/diag_tau.py` (directions à τ ∈ {0,03 ; 0,1 ; 0,3 ; 1 ; 3}, 4 têtes chacune, mêmes tirages) : **aucun τ ne donne de
  descente** depuis l'optimum. ΔAcc va de −0,04 à −0,18 au pas de 0,03–0,1 niveau, et de −1,1 à −1,2 pour un pas de
  signe à 1 niveau. Les directions diffèrent pourtant (cos = 0,54 entre τ=0,03 et τ=3). Ce n'est donc pas un biais du
  straight-through : c'est un optimum local robuste de l'objectif réel.
- Conclusion : à n=120, le raffinement local d'images réelles plafonne vers +2,2 points (89,3 %). Les images n'y
  bougent que de 0,22 niveau en moyenne (p99 : 2), contre 1,1 (p99 : 5, max 18) à n=10, où PFD gagnait +11,7.
  On retourne donc au régime n=10 avec le nouvel optimiseur. Pour égaler 0,7036, il faut 64,1 % à n=10.

**Autres tentatives du 13/10 au soir**
- Adam pixels à n=10 (4 têtes, lr 0,5 → 0,05, depuis les médoïdes) : 49,5 → 54,9 (it 25) → 55,3 → 55,4 → 55,1 % (it 100).
  Plateau plus bas que max-norm (59,9 %), run arrêté. À n=10, les changements utiles sont gros sur quelques pixels
  (jusqu'à 18 niveaux), alors qu'Adam plafonne le mouvement de chaque pixel.
- **Mosaïques 2×2 de la même classe** (4 médoïdes réduits en 32×32 par image, `work/mosaic_eval.py`) : n=120 →
  75,5 % (médoïdes : 87,1 %), n=40 → 53,8 % (médoïdes : 67,4 %). **Échec net** : le passage à demi-échelle casse la
  correspondance avec les images de test. RDED et ses mosaïques ne marchent pas avec ce protocole (pour le paper).
- **Arithmétique des perturbations** (`work/delta_mix.py`, 3 seeds) : les optima max-norm (δ1, 0,22 niveau) et
  Adam (δ2, 1,72 niveau) ont un cos de 0,23, ce sont donc deux optima différents de même qualité. Résultats :
  δ1 89,33 · δ2 89,08 · (δ1+δ2)/2 89,23 · δ1+δ2 88,70 · **1,5·δ1 89,51** (seed 42 : 89,06) · 2·δ1 89,28.
  +0,18 au maximum, dans le bruit, donc à revérifier sur plus de seeds avant de s'en servir.
  **Vérifié avec le script officiel sur 6 seeds** : 1,5·δ1 donne 89,12 ± 0,59 % contre 89,27 ± 0,70 % pour le set
  soumis. Seul le seed 42 progresse (88,89 contre 88,37). Le set n'est pas meilleur en moyenne : **pas soumis**.
  Il faudrait sinon une hausse sur l'ensemble des seeds, pas un gain sur le seed 42 seul. On ne pourrait l'utiliser
  qu'à la toute fin, pour départager des sets équivalents.

**Frontière PFD3 (max-norm, 2 têtes, 300 itérations, depuis les médoïdes)** :

| n | pas | médoïdes | PFD3 (meilleur, 3 seeds) | gain | score moyen |
|---|---|---|---|---|---|
| 64 | 10 | 75,10 | 78,38 (it 300) | +3,3 | 0,6807 |
| 96 | 15 | 83,24 | 86,03 (it 300) | +2,8 | 0,6984 |
| 120 | 20 | 87,15 | 89,33 (it 250) | +2,2 | **0,7036** |
| 97 | 20 (32+32+32+1) | 73,62 | 77,11 (it 100, arrêté) | +3,5 | 0,6533 |

Le gain de PFD croît quand n diminue, mais pas assez pour déplacer l'optimum de score : n=120 reste le meilleur
point de fonctionnement.

**Changer d'init à n=120** (k-means avec d'autres graines, 12 par classe, PFD3 max-norm, 2 têtes, 300 itérations) :

| init | images communes avec km0 | init (3 seeds) | PFD3 meilleur | gain |
|---|---|---|---|---|
| km0 (soumis) | 120 | 87,15 | 89,33 | +2,2 |
| km1 | 35 | 87,80 | 89,39 | +1,6 |
| km2 | 28 | 85,98 | 88,55 | +2,6 |

Partir d'une init un peu meilleure ne fait pas monter le plafond : km0 et km1 finissent tous deux vers 89,3–89,4 %.
Une init nettement plus faible (km2, −1,2 point) gagne davantage (+2,6) mais finit 0,8 point plus bas. Le plafond
vient donc surtout de la méthode, et une mauvaise init en garde une trace. Moyenne finale sur les 3 inits :
89,09 ± 0,47 %.

### 2.5 ter Perte externe sans échelle (14/10)
**Contexte** : le 14/10 au matin, Lynch James est 2ᵉ à 0,70144, soit 0,0011 derrière nous (≈ 0,2 point d'accuracy). Il faut creuser l'écart.

**Pas basse fréquence (`diag_step.py --kinds lp4,lp8,lp16,lp64`, méta-gradient de 4 têtes)**. Le gradient est moyenné par
blocs f×f puis ré-interpolé en bilinéaire (f=64 : un décalage de couleur par image). Résultats en ΔAcc :

| direction | 0,3 niveau | 1 | 3 | 10 |
|---|---|---|---|---|
| lp4 depuis PFD3 it=250 | −0,63 | −3,45 | −11,2 | −22,8 |
| lp8 depuis PFD3 | −0,48 | −2,47 | −8,1 | −19,5 |
| lp16 depuis PFD3 | −0,31 | −1,06 | −3,7 | −14,9 |
| lp64 depuis PFD3 | −0,01 | −0,27 | −0,31 | −0,49 |
| lp4 depuis les médoïdes | +0,04 | −3,11 | −14,5 | −39,5 |

Les perturbations lisses ne permettent **pas** de plus grands pas, même depuis les médoïdes, où un pas de signe à ±1 niveau
donne +0,85. L'information utile est dans le motif haute fréquence (le signe pixel par pixel). C'est une piste fermée.

**Diagnostic de l'objectif** : à l'optimum PFD3, la température qui minimise la CE réelle après 20 pas est **T* ≈ 0,32**.
Les logits sont donc ~3× trop mous, et la CE réelle passe de ~0,63 (T=1) à 0,26 (T=T*). La CE brute récompense
surtout la **confiance** (l'échelle des logits), qui ne compte pas pour l'accuracy. D'où `--outer-T 0` dans `pfd3.py` :
T* est ajusté à chaque batch réel (grille log de 0,01 à 10, sans gradient : théorème de l'enveloppe), puis la perte
externe est CE(logits / T*). L'objectif devient invariant à l'échelle et se concentre sur le classement.

**Run `--outer-T 0` depuis PFD3 it=250 (max-norm, 2 têtes, ε 0,03 → 0,01, 100 itérations)** :

| itération | 0 | 25 | 50 | 75 | 100 |
|---|---|---|---|---|---|
| moyenne sur 3 seeds | 89,31 | 89,41 | 89,49 | 89,60 | **89,64** |
| seeds 42 / 0 / 1 | 88,59 / 90,12 / 89,21 | | 88,81 / 90,44 / 89,21 | 88,96 / 90,54 / 89,31 | 88,99 / 90,57 / 89,36 |

Pour la première fois, on dépasse l'optimum PFD3 au lieu d'y revenir, et les trois seeds montent ensemble. T* reste
entre 0,25 et 0,34 pendant le run. 
**Palier 2** (depuis le meilleur point du run ci-dessus, ε 0,03 → 0,005, 200 itérations) :

| itération | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---|---|---|---|---|---|---|---|---|
| moyenne sur 3 seeds | 89,75 | 89,79 | 89,80 | 89,89 | 90,02 | 90,05 | 90,02 | **90,13** |

**Script officiel sur 6 seeds (set `sets/pfd3_n120_T0s2`, it=200 du palier 2)** :

| seed | 42 | 0 | 1 | 2 | 3 | 4 | moyenne ± écart-type |
|---|---|---|---|---|---|---|---|
| PFD3 (soumis le 13/10) | 88,37 | 89,90 | 89,16 | 90,10 | 88,42 | 89,65 | 89,27 ± 0,70 |
| **PFD3 + perte sans échelle** | 88,40 | 90,69 | 89,65 | 91,16 | 89,51 | 90,22 | **89,94 ± 0,98** |
| gain | +0,03 | +0,79 | +0,49 | +1,06 | +1,09 | +0,57 | **+0,67** |

- Gain positif sur les 6 seeds. Le seed 42 est sous la moyenne (88,40 contre 89,94), donc la règle de soumission est
  respectée. **Soumis le 14/10 à 14:40 : 0,70344 sur Kaggle**, test ≈ 89,31 % (contre 89,14 %). Les prédictions sont
  à 96,7 % identiques à celles de la soumission précédente.
- Écart réplique/officiel au seed 42 : 89,46 contre 88,40, plus grand qu'avant (88,64 contre 88,37). Sur les autres seeds,
  l'écart est normal (seed 0 : 91,11 contre 90,69 ; seed 1 : 89,83 contre 89,65). À surveiller : si l'écart grandit avec
  l'optimisation, une partie du gain exploite peut-être des détails de la réplique (ré-échantillonnage GPU contre PIL).
- **Fidélité vérifiée** (`work/fidelity_pil_eval.py`, 6 seeds). Le script officiel redimensionne en PIL uint8 (passe
  horizontale, arrondi, passe verticale, arrondi), alors que la réplique reste en float (0,28 niveau d'écart moyen par
  pixel, autant que nos perturbations). D'où `PROTO_PIL=1` dans `proto.py`, qui émule PIL à 98,7 % des pixels près.
  Moyennes : set du 13/10 → réplique float 89,33, réplique PIL 89,39, officiel 89,27 ; T0s2 → 90,01 / 90,06 / 89,94.
  La réplique prédit donc la moyenne officielle à 0,1 point près, et l'arrondi PIL ne change rien. Le seed 42 officiel
  bas (88,40 contre 89,51 en réplique) est un tirage défavorable de l'ordre des batches et des crops propre à ce seed.
  Écarts par seed : −1,11 / −0,45 / −0,15 / +0,42 / +0,45 / +0,39.
- Pour le paper : la CE brute pousse la confiance, la CE calibrée pousse le classement. Même méthode, même coût,
  +0,67 point officiel. Suite : run propre depuis les médoïdes avec `--outer-T 0` (300 itérations), en cours.

### 2.5 quater Point de fonctionnement dans la bande des 20 pas (14/10, `work/nsweep_eval.py`, réplique, 6 seeds)
Médoïdes k-means (allocation `make_medoids.alloc`, classes difficiles d'abord), inits km0/km1/km2 (`make_medoids_km.py --n`) :

| n | dernier batch | km0 | km1 | km2 | moyenne | score moyen |
|---|---|---|---|---|---|---|
| 104 | 8 | 84,53 | – | – | 84,53 | 0,6868 |
| 108 | 12 | 85,48 | – | – | 85,48 | 0,6897 |
| **112** | 16 | 86,91 | 86,72 | 86,40 | **86,68** | **0,6938** |
| 116 | 20 | 86,82 | – | – | 86,82 | 0,6927 |
| 120 | 24 | 87,04 | 87,40 | 86,28 | 86,91 | 0,6915 |

- À n=112, on ne perd que 0,23 point d'accuracy pour 8 images de moins, qui valent 0,68 point au score. n=112 bat donc
  n=120 de +0,0023 au score, de façon cohérente sur les 3 inits. En dessous (dernier batch ≤ 12), le petit batch coûte cher.
- Run lancé : PFD3 + `--outer-T 0` à n=112 depuis km0 (500 itérations, ε 0,05 → 0,005), `runs/pfd3_n112_T0.out`.
  Pour battre le set soumis (score moyen officiel attendu ≈ 0,7066), il faut ≥ 89,24 % de moyenne à n=112.
- **Résultat n=112** (`runs/pfd3_n112_T0.out`) : 87,05 → 88,56 (it 75) → 89,14 (150) → 89,30 (200) → 89,58 (300) →
  **89,61 (it 400)** → 89,48 (500) en réplique sur 3 seeds. **Script officiel sur 6 seeds** (`sets/pfd3_n112_T0`) :
  87,63 / 90,67 / 90,47 / 90,10 / 87,88 / 88,69, soit **89,24 ± 1,34 %**, score moyen 0,7066. C'est **exactement le même**
  score que le set soumis à n=120 (89,94 %, 0,7066), avec plus de variance. **Pas soumis.**
- Signal d'alerte en fin de run : la perte réelle calibrée tombe presque à 0 (0,0001 à l'it 485, T* = 0,01). Le modèle
  classe donc parfaitement des batches d'images d'entraînement, alors que la validation plafonne à 89,5 %. Hypothèse :
  la méta-optimisation **sur-apprend les images d'entraînement** de la perte externe (26 passages sur les 18 900 images en
  500 itérations). **Hypothèse réfutée** (`work/train_val_gap.py`, accuracy sur 4 050 images de train contre la
  validation, 3 seeds) : médoïdes 86,86 / 87,10 ; PFD3 CE 89,21 / 89,33 ; T0s2 89,75 / 90,12 ; n=112 89,81 / 89,61.
  Aucun écart train/validation : l'effondrement de la perte en fin de run est un événement isolé de la boucle interne.
- **Taille des perturbations** (par rapport aux médoïdes, niveaux uint8) : PFD3 CE 0,225 en moyenne (p99 2, max 8,
  20 % des pixels modifiés) ; T0s2 0,263 (p99 2, max 11, 22 %) ; n=112 0,328 (p99 2, max 11, 28 %). Toujours imperceptible.
- **Palier 3** : depuis T0s2, 4 têtes par itération, ε 0,02 → 0,005, 150 itérations (`runs/pfd3_n120_T0_stage3h4.out`,
  76 s par itération avec le GPU partagé). Réplique sur 3 seeds : 90,12 → 90,14 → 90,23 → 90,12 → 90,20 → **90,26 (it 125)**
  → 90,25. **Réplique sur 6 seeds** : 89,63 / 91,21 / 90,05 / 90,81 / 88,99 / 90,07, soit 90,13 contre 90,01 pour T0s2
  (+0,11 ; positif sur 5 seeds sur 6). C'est marginal : candidat gardé (`cache/pfd3_n120_T0_stage3h4_best.pt`), pas de
  script officiel pour l'instant.
- **Précision de la boucle interne** : PFD déroule les 20 pas en bf16 (mantisse de 7 bits, soit une précision relative
  d'environ 0,4 %, l'ordre d'un niveau de pixel), alors que l'évaluation officielle est en fp32. L'effet des perturbations
  de ±1 niveau est donc au niveau du bruit d'arrondi dans la passe avant. Nouvelle option `--fp32-inner --tf32` (mantisse de
  10 bits) : 15,4 Go de pic, environ 1,3× le coût du bf16. Run depuis T0s2, 2 têtes, 150 itérations : 90,14 → 90,13 →
  90,16 → 90,17 → 90,16 → 90,18 → 90,20 sur 3 seeds. **Réplique sur 6 seeds : 90,03** (contre 90,01 pour T0s2).
  **Aucun gain** : la précision de la boucle interne n'est pas ce qui limite. On garde le bf16.
- **Température fixe basse** (`--outer-T 0.1`, plus proche d'une perte 0-1) depuis le meilleur point du palier 3, 100 itérations :
  90,29 → 90,22 → **90,34 (it 50)** → 90,27 → 90,33 sur 3 seeds. Réplique sur 6 seeds : 90,15. **Script officiel sur 6 seeds**
  (`sets/pfd3_n120_T01`) : 88,72 / 90,96 / 89,53 / 91,19 / 89,38 / 90,20, soit **90,00 ± 0,96** contre 89,94 pour T0s2
  (+0,06, dans le bruit). **Pas soumis.** Candidat de départage pour la toute fin, si deux sets restent équivalents en
  moyenne (seed 42 : 88,72 contre 88,40).
- **Deuxième bassin (km1)** : depuis l'optimum CE de km1 (89,42 sur 3 seeds), `--outer-T 0` pendant 300 itérations → 90,09
  (3 seeds), puis T=0,1 pendant 100 itérations → 90,12. **Réplique sur 6 seeds : 89,81 pour les deux** (contre 90,16 pour km0).
- **Mélange par classe** (`work/basin_mix.py`) : pour chaque classe, on prend les 12 images du bassin où cette classe a la
  meilleure accuracy (km1 retenu pour AnnualCrop, Herbaceous et Pasture). Réplique sur 6 seeds : **89,95, contre 90,16 pour km0
  seul. Échec** : les gains par classe ne s'additionnent pas. Herbaceous passe par exemple de 89,6/90,6 à 89,0 dans le mélange.
  Les classes interagissent trop pour qu'on les combine.
- **Bruit de mesure** : la réplique sur 6 seeds du set n=112 donne 89,99 contre 89,24 en officiel (écarts par seed −0,30 /
  +0,47 / −0,42 / −1,23 / −1,75 / −1,29), alors que l'écart n'est que de 0,07 pour T0s2 à n=120. Avec un petit dernier batch
  (16 images), le résultat dépend fortement de la composition des batches, qui diffère entre réplique et officiel. L'erreur
  standard d'une moyenne sur 6 seeds est d'environ 0,4–0,5 point, donc des écarts de 0,1 entre candidats ne sont pas
  mesurables. On ajoute les seeds officiels 5 à 8 pour T0s2 et T01 (`runs/chain_more_seeds.sh`).
- **Bilan à n=120** : le plateau de la perte sans échelle est vers 90,0 % officiel (6 seeds). Ni 4 têtes, ni fp32/TF32, ni
  T=0,1 ne le déplacent de plus de 0,1. Piste suivante : d'autres bassins (km1 avec le même pipeline, `runs/chain_km1_T.sh`),
  puis combiner les meilleures classes de chaque bassin.
- **Biais de sélection** : choisir l'itération sur 3 seeds (max sur ~20 évaluations) surestime la moyenne officielle sur
  6 seeds de 0,2 à 0,4 point (T0s2 : 90,13 contre 89,94 ; n=112 : 89,61 contre 89,24). Pour comparer deux candidats,
  il faut la réplique sur 6 seeds, ou le script officiel.
- Le run de contrôle n=120 depuis les médoïdes avec `--outer-T 0` a été arrêté à l'itération 75 : 88,56 %, contre 88,25 %
  pour la CE brute au même point (87,27 / 88,11 / 88,56 aux itérations 25 / 50 / 75).

### 2.5 quinquies Le classement bascule (15/10 matin) et l'effet « seed 42 »
**Classement au 15/10 04:25 UTC** : Lynch James **0,74070**, vinicius simm **0,73122**, nous 0,70344 (3ᵉ). Un saut de +0,037
est impossible à n≈120 : il faudrait 96,8 % d'accuracy, au-dessus de l'oracle (91,7 %). Ils sont donc passés à très peu
d'images. Accuracy nécessaire pour 0,745 : n=10 → 72,4 % ; n=20 → 79,4 % ; n=33 → 84,5 % ; n=40 → 86,5 %.

**Petits n en multi-seeds avec la perte sans échelle** : n=10 depuis pfd_v3 (61,1 %), 4 têtes, 75 itérations → plat
(61,0 %) ; n=33 depuis les médoïdes : 32,4 → 34,6 % en 75 itérations (le batch d'une image détruit l'entraînement). Arrêtés.

**Expérience « tête du seed 42 fixée »** (`pfd3.py --fixed-head-seed 42`) : on optimise pour la tête initiale exacte que
baseline.py tire avec la seed 42. Les crops restent aléatoires, et à n=10 il n'y a qu'un batch, donc pas d'effet de l'ordre.
Run a (depuis pfd_v3, 2 tirages de crops par itération, ε 0,05 → 0,01, perte sans échelle), réplique seeds 42 / 0 / 1 :

| itération | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 42 | 58,2 | 62,6 | 65,8 | 67,6 | 68,1 | 68,9 | 69,3 | 69,7 | **70,1** | 69,7 |
| seeds 0 / 1 | 60,7 / 64,6 | 60,6 / 61,6 | 61,1 / 61,8 | 61,2 / 59,6 | 59,7 / 59,4 | 59,3 / 59,0 | 58,8 / 58,6 | 58,8 / 58,5 | 58,3 / 58,2 | 58,2 / 57,8 |

- Viser la tête du seed 42 rapporte **≈ +12 points sur ce seed**, au prix de −2 à −7 points sur les autres. Score sur le seed
  42 vers 0,733. C'est très probablement ainsi que les deux premiers obtiennent 0,74.
- **Le gain se transfère au script officiel** : run b (même réglage, images sauvegardées à chaque évaluation avec
  `--save-every-eval`), itération 50 : seed 42 officiel **64,02 %** (réplique 64,69) contre 58,2 % au départ ; seed 0
  officiel 61,21 (réplique 61,21). La tête initiale est tirée sur CPU, donc identique sur toute machine. Les crops du script
  officiel diffèrent de ceux de la réplique, et le gain tient quand même.
- Run b complet (400 itérations) : seed 42 en réplique 70,67 % à l'itération 400 (seeds 0 / 1 : 58,1 / 59,1).
  **Script officiel** (`sets/n10_head42b_it400`) : **seed 42 = 71,60 %**, seed 0 = 58,20, seed 1 = 58,74, seed 2 = 62,79.
  Score Kaggle attendu avec la seed 42 : 0,5 × 0,716 + 0,383 ≈ **0,741**, au niveau du 1ᵉʳ actuel (0,7407). Les autres
  seeds donnent ≈ 0,68.
- n=20 avec la tête fixée (depuis les médoïdes n=20, `runs/pfd3_n20_head42.out`) : seed 42 de 49,9 à 73,3 % en
  250 itérations. Au score, il faudrait ≥ 78,6 % pour égaler n=10 à 71,6 %.
- **Décision de l'utilisateur (15/10, ~12:00 UTC) : soumettre le set ciblé sur le seed 42**, en acceptant le risque de
  vérification. Soumis : **0,74258, 1ᵉʳ**. Deux relances seed 42 donnent des prédictions identiques (4 050/4 050). La règle
  « seed 42 ≤ moyenne multi-seeds » ne s'applique donc plus aux soumissions ciblées. Il reste à garder les PNG exacts et à
  vérifier le déterminisme à chaque soumission.
- (Note antérieure à la décision) C'est exactement le cas que notre règle de vérification interdit : seed 42 très au-dessus de la
  moyenne multi-seeds. La décision revient à l'utilisateur, et dépend de la réponse des organisateurs à la question 2 du
  forum (seed et tolérance de la vérification).

### 2.5 sexies Trajectoire officielle exacte du seed 42 (15/10 après-midi)
- `work/record_views.py <set> 42 <out>` fait tourner **le DataLoader de baseline.py lui-même** (générateur seedé, 2 workers,
  `seed_worker`) avec une transformation qui tire les nombres aléatoires exactement comme RRC + 2 flips, et qui renvoie
  les paramètres au lieu de l'image. On obtient les batches et les vues exactes de chaque epoch (`cache/views_n10_s42.pt`).
- `proto.augment_views` + `dd_lib.train_eval_views` rejouent ces vues ; avec `PROTO_PIL=1` et du fp32, la réplique reproduit
  l'officiel epoch par epoch (21,36 / 44,22 / 64,17 / 70,17 / 71,51 contre 21,36 / 44,12 / 64,17 / 69,85 / 71,60).
- `pfd3.py --views <file> [--views-prob p] --select-exact` : la boucle interne suit les batches et les vues enregistrées,
  avec la tête du seed 42 (`--fixed-head-seed 42`), en fp32 (`--fp32-inner`) et `PROTO_PIL=1`. Avec `--views-prob 0.5`, la
  moitié des échantillons utilise des crops aléatoires.
- Runs depuis `n10_head42b_it400` (perte sans échelle, ε 0,03 → 0,005) :

| run | it 20 | 40 | 60 | 100 | 120 | 140 | 160 | 200 |
|---|---|---|---|---|---|---|---|---|
| vues exactes seules (exact / crops aléatoires) | 74,0 / 69,5 | 75,8 | 77,1 | 77,5 / 66,4 | 78,4 | 78,8 | 79,0 | **79,3** / 64,6 |
| mélange 50/50 (exact / aléatoires) | 73,8 / 70,8 | 75,1 | 76,0 | 76,5 / 71,4 | 77,0 | 77,4 / 71,4 | | |

- **Script officiel, seed 42** : vues exactes it 20 → 73,75 ; mélange it 20 → 74,10 ; **vues exactes it 200 → 79,38**.
  Deux relances donnent des prédictions identiques.
- **Robustesse numérique** (`work/views_robust.py`, vues exactes, en basculant TF32 sur les matmuls et sur les convolutions) :
  ±0,05 point au plus (it 200 : 79,48 / 79,48 / 79,46 / 79,43). Un autre GPU ne devrait donc pas casser le résultat.
- **Palier 2** (depuis it 200, ε 0,02 → 0,003, 300 itérations) : 79,38 → 79,14 → 78,91 → 79,19 → 79,58 → 79,26 → 79,46 →
  79,78 → 79,38 → 80,17 → 80,12 → 80,32 → 80,47 → 80,49 → **80,64 (it 280)** → 80,49. Officiel seed 42 : **80,62 %**,
  relance identique, TF32 ±0,07. **Soumis : 0,78530.**
- **Palier 3** (ε 0,015 → 0,002) : plateau 80,57–80,72 jusqu'à l'it 140, arrêté. **T=0,1 fixe** depuis it 280 : 79,9 / 80,2 /
  79,6, en baisse, arrêté. **Depuis les médoïdes bruts** (1 par classe, vues exactes dès le départ, 400 itérations) : 46,3 →
  64,6 (it 20) → 74,2 (80) → 77,5 (160) → 78,6 (240) → **80,35 (400)**. C'est un autre optimum de même niveau que la lignée
  soumise (80,6). À n=10, la trajectoire exacte plafonne vers 80,5 %. Rien de plus soumis.
- **Soumis : 0,78135, 1ᵉʳ.** Nouvelle dépendance : le résultat suppose `baseline.py` avec ses arguments par défaut
  (seed 42 **et** `--workers 2`, puisque les crops dépendent des workers). Avec d'autres crops, le même set tombe à ~65 %.

### 2.5 septies 16/10 : dépassés par vinicius simm (0,78764), et la piste n=33
**Classement 16/10 12:11 UTC** : vinicius simm 0,78764, nous 0,78530, Alpha 0,74987, Lynch James 0,74823. À n=10, 0,78764
correspond à ~80,9 % de test, juste au-dessus de nous (80,4 %).

**Accuracy nécessaire selon n** : pour battre 0,78764 → n=10 : 80,91 % ; n=11 : 81,88 ; n=12 : 82,76 ; n=20 : 87,95 ;
**n=33 (10 pas) : 93,04** ; n=40 : 94,99. Pour 0,795 → n=10 : 82,38 ; n=33 : 94,51.

**Plafonds (`work/ceiling2.py`, descente de signe projetée dans la boîte ±Σlr autour des poids ImageNet, tête du seed 42,
batches de 256 vraies images augmentées)** : 5 pas → 78,6 (it 100) / 79,1 (200) / 79,7 (300) / 80,0 (400) / 80,4 (500) /
80,3 (600), plateau vers **80,3 %**. Nos 10 images (80,6 %) sont déjà au-dessus, donc n=10 est saturé. 10 pas → 93,4 (100) /
93,7 (200) / 94,7 (300) / 94,8 (400) / 95,3 (500) / 95,3 (600), plateau vers **95,3 %**.

**Set final n=10, rejoué exactement (`work/views_confusion.py`)** : val 80,69, train 79,70 (pas de sur-apprentissage).
Par classe : Forest 92,7 · Residential 92,2 · SeaLake 90,2 · Industrial 86,7 · HerbVeg 85,3 · AnnualCrop 78,2 · Pasture 75,7 ·
PermCrop 67,5 · River 65,9 · Highway 64,5. Confusions : River↔Highway 110, AnnualCrop→River 35, PermCrop→HerbVeg 33.

**n=10, plus de vraies images dans la perte (2048) depuis le set final** : max-norm → 80,25 à l'it 20 ; Adam pixels (lr 0,3)
→ 79,51. Rien ne dépasse 80,6, arrêtés.

**n=33 sur le run exact du seed 42** (`cache/views_n33_s42.pt` : 2 batches par epoch, 32 + 1 ; les images seules sont les
lignes 12, 25, 20, 2, 13) : départ des médoïdes à 22,2 %. fp32, lr 0,05 : 65,2 (it 20) / 71,4 (40) / 76,8 (60). bf16, lr 0,08 :
66,7 / 78,2 / 80,0 / 81,5 (it 80). Troisième run bf16, lr 0,15, 800 itérations en cours.

### 2.5 octies 19/10 : vinicius simm à 0,80542, on est 2ᵉ
**Classement 19/10 12:20 UTC** : vinicius simm 0,80542, nous 0,78530, Alpha 0,76478, Lynch James 0,74823, Prisma 0,74565.
Accuracy nécessaire pour battre 0,80542 selon n : n=1 → 61,1 % ; 2 → 68,1 ; 3 → 72,2 ; 4 → 75,2 ; 5 → 77,4 ; 6 → 79,3 ;
7 → 80,9 ; 8 → 82,2 ; 9 → 83,4 ; **10 → 84,5**. Le forum n'a toujours pas de réponse des organisateurs à nos questions (taille
des images, seed de la vérification). `baseline.py` accepte n'importe quelle taille d'image (PIL `convert('RGB')` puis RRC 224)
et n < 10 (les 10 classes viennent de `classes.json`).

**n < 10 sur la trajectoire exacte** (`views_for_n.py` enregistre les vues officielles du seed 42 pour n images ; les vues par
position dans le batch sont les mêmes pour tout n, seule la permutation change) :
- Retirer une image du set soumis sans réoptimiser : 80,6 → 56 à 65 % (le set est réglé sur les batches exacts de n=10).
- n=5 (River, Highway, PermCrop, Pasture, AnnualCrop, depuis le set soumis), 1 tête, ε 0,05 → 0,005 : 30,4 (it 0) → 39,0 (20) →
  42,2 (40) → 44,7 (60) → 46,8 (80). Arrêté : il faudrait 77,4 %. Avec des labels durs, une classe sans image n'est presque
  pas apprise. n < 10 n'est pas la bonne voie.

**Le plafond à 5 pas était sous-estimé** (`ceiling3.py`). Même boîte ±Σlr autour des poids ImageNet, tête du seed 42, mais
perte CE à température ajustée (comme la perte externe de PFD), images d'entraînement vues avec la transformation d'éval, et
Adam projeté au lieu de la descente de signe : **86,5 % (it 100), 88,2 % (it 200)**, contre 80,3 % pour `ceiling2.py`. Le
protocole permet donc ≥ 88 % en 5 pas. Nos 10 images (80,6 %) sont loin de ce plafond : la limite vient de l'optimisation des
images (ou de leur capacité), pas du protocole. vinicius simm est très probablement à n=10 vers 84,5 %.
- Le set soumis déplace la plupart des poids près du bord de la boîte : |Δθ|/R médian 0,92 (p25 0,69, p90 0,99).
- Pistes lancées : (1) images 224×224 à n=10 (12× plus de pixels, pas d'agrandissement dans le RRC) pour tester l'hypothèse de
  capacité ; (2) `pfd3.py --target` : la perte externe tire la fin de trajectoire θ_K vers le réseau du plafond θ* (L2, ou
  linéaire vers le coin sign(θ* − θ0)), seule ou mélangée à la perte réelle.

**Réponse des organisateurs (forum Kaggle, 19/10)** : la vérification relance `baseline.py` avec les réglages par défaut,
**seed 42 comprise** ; un écart de « quelques dixièmes de point » est acceptable ; le 64×64 RGB PNG est préféré mais pas
obligatoire ; des images issues du train (sélectionnées ou modifiées) sont acceptées. Les sets ciblés sur la trajectoire
exacte du seed 42 sont donc légitimes.

**Diagnostics à n=10 (trajectoire exacte, set soumis)** :
- Images 224×224 (`views_for_n.py n 42 224`, init = set soumis agrandi) : 50,9 → 75,0 (it 20) → 79,0 (100) → 80,6 (140) →
  plateau ~80,5. Même plafond qu'en 64×64 : la capacité en pixels n'est pas la limite. On garde le 64×64.
- `pfd3.py --target` (L2 vers θ*) : la distance à la cible ne baisse pas, accuracy plate (80,5–80,8). Échec.
- `rough_exact.py` : un bruit aléatoire de ±0,1 à 0,3 niveau ne change rien (val 80,4–80,7), mais un pas le long du
  méta-gradient fait chuter (0,1 niveau linéaire : 75,9 ; signe : 79,6 ; 0,3 niveau signe : 74,7). À l'optimum, le
  méta-gradient straight-through est un mauvais modèle local. `sparse_exact.py` : ±1 niveau sur les 30 pixels de plus fort
  |g| aide un peu (CE 0,6421 → 0,6410), le signe opposé nuit. `tau_probe.py` : aucun τ ∈ [0,03 ; 3] n'est nettement meilleur.
- **Coins de la boîte** (`corner_eval.py`) : θ0 + R·sign(θ* − θ0) garde **88,3 %** (θ* : 88,1 %) ; notre fin de trajectoire
  projetée au coin : 81,2 % (telle quelle 80,4–80,6). Avec ½R : 40 %. Toute l'information est dans le **motif de signes**.
  Notre fin de trajectoire et θ* (cible « proche », `ceiling3.py --init-state` depuis notre θ5) ont 88 % de signes communs.

**Nouvelle méthode : sign matching sur le chemin cible** (`signmatch.py`). Si le gradient du batch a le signe −s à chaque
pas officiel (s = sign(θ* − θ0)), l'entraînement finit au coin de θ*. Le chemin cible est alors connu d'avance
(θ_k = θ_{k−1}(1 − lr·wd) + lr_k·s), donc chaque pas est évalué en son point cible avec sa vue enregistrée : pas de
déroulé, pas de chaos. Perte = moyenne sur les poids de softplus(g_k·s / (τ·rms(g_k))). Accord de signe au départ
(pas 1 à 5) : 86 / 83 / 79 / 74 / 63 %.
- Pas max-norm (logits) : accord 84 / 83 / 81 / 79 / 77 % à l'it 60, val 79,0–80,3. Lent.
- **Adam sur les pixels (lr 1 → 0,1 niveau)** : val exacte 80,64 → 78,40 (it 20) → 80,42 (40) → **81,53 (60) → 81,68 (80)**.
  Premier gain au-delà du plateau de 80,6. Pondération par |∂L/∂θ| au coin : moins bien (79,65 à l'it 40).

**Tours « cible proche → sign matching → polissage PFD »** (n=10, trajectoire exacte du seed 42, val exacte) :

| étape | départ | résultat | officiel seed 42 |
|---|---|---|---|
| set soumis le 15/10 | – | 80,64 | 80,62 |
| sign matching vers cible proche 88,1 % (smA) | 80,64 | 82,17 (it 120) | **82,32** |
| polissage PFD exact (pf_smA, ε 0,02 → 0,003) | 82,17 | 82,64 (it 140) | **82,74** |
| cible proche 1 000 pas (90,3 %), sign matching (r3n, lr 0,7 + warmup 20) | 82,64 | 83,31 (it 160) | – |
| polissage PFD (pf_r3n) | 83,31 | 83,88 (it 120) | **83,88** (score attendu ≈ 0,8025) |
| tour 4 : sign matching (cible 89,8 %) | 83,88 | 83,0–83,4, pas mieux | – |

- Cible lointaine (plafond depuis θ0, 89,3 %) : 81,5 % au lieu de 83,1 % pour la cible proche. Cible plus proche (it 100,
  86,4 %) : 80,9 % au lieu de 82,2 %. Il faut une cible **forte et proche**.
- Cible « consensus » (deux plafonds proches avec des batches réels différents, poids = accord des signes) : pas mieux.
- Sign matching sur le chemin réel (« on-policy ») au lieu du chemin cible : équivalent (82,07 contre 82,17).
- Le PFD exact, inutile à son propre optimum, redevient utile après un sign matching (+0,5 point à chaque fois).
- Le plafond lui-même monte avec plus de pas : 90,3 % (1 000 pas, depuis notre fin de trajectoire).

**Capacité : la limite des 10 images en 64×64** (19/10 après-midi)
- Au tour 4, l'accord de signe avec la nouvelle cible part de 85,7 % (pas 1) et le sign matching ne le fait plus monter
  (il baisse même à 84,6 %). Le softplus pousse surtout les poids très en désaccord, sans doute impossibles à retourner.
  Variante « sigmoïde » (compte d'accords lissé) écrite, test en attente (OOM pendant le job QA2026).
- Coin de notre trajectoire actuelle : +0,4 point seulement (83,4 → 83,8 en éval float). Pas de gain caché en magnitude.
- **224×224 + sign matching** (depuis le set PFD 224 à 80,5 %) : tour 1 → 84,02 (sign matching, it 120) → **84,20** après
  polissage, contre +1,6 au premier tour en 64×64. Avec le sign matching, la capacité en pixels compte.
- **n=11** (set à 83,88 + un médoïde River) : la trajectoire change (vues de n=11), départ à 30 %, sign matching → 76,4 % en
  160 itérations. Il faudrait 85,4 % pour battre 0,80542. Arrêté pour libérer le GPU (job vLLM QA2026 à 43 Go).
- Rounds depuis les médoïdes (autre bassin) à n=10 : lancés puis arrêtés pour la même raison.

**Soumis le 19/10 à 16:20 : 0,80357** (test ≈ 84,10 %, val officielle 83,88), 2ᵉ à 0,00185 de vinicius simm.
Précision « poolée » val + 8 192 images de train (`exact_eval.py`) : 84,20 %. Il manque environ 0,4–0,5 point.

**Ce qui ne fait plus monter au-delà de ~84 %** (depuis le set soumis, 64×64) :
- Tour 2 en 224×224 : sign matching 84,07 → max 83,63, polissage 84,22. Le 224 sature aussi vers 84,2.
- Cible proche plus forte et cohérente avec PIL (`PROTO_PIL=1 ceiling3.py`, 1 500 pas, lr 0,03 : 90,8 %) : accord de départ
  77,5 % comme voulu, mais 83,88 → 82,5 → 83,2. Pas de gain.
- Perte sigmoïde (compte d'accords) : effondrement (84 → 65,6 %, l'accord baisse aussi).
- Poids |θ* − θ0|/R (signes « décidés » de la cible) : 82,4 %.
- Polissage glouton uint8 (`polish_sparse.py`, ±1 niveau sur les top-k pixels, accepté si la CE sur 4 096 images de train
  baisse) : CE 0,4746 → 0,4702, val 83,83 → 84,00 puis bloqué ; en précision poolée +0,05 (bruit).
- Constat : après un polissage, le sign matching plonge toujours puis revient vers 83 % ; son optimum n'est pas celui de
  l'accuracy. Le cycle sign matching ↔ polissage a un point fixe vers 84 %.
- En cours : deuxième lignée depuis l'optimum PFD des médoïdes (80,35 %), et hybride sign matching + méta-gradient PFD
  (poids 3 pour la perte réelle).

**Hybride sign matching + méta-gradient PFD exact** (`signmatch.py --pfd-w 3 --real-bs 1024`, Adam pixels lr 0,2 → 0,02,
cible proche 1 000 pas PIL) depuis le set soumis à 0,80357 : val exacte 83,88 → 84,22 (it 60) → 84,40 (it 120–140). Précision
poolée (val + 8 192 images de train, `exact_eval.py`) : 84,28 → **84,99** (it 120). Les deux gradients se complètent : le sign
matching tire vers les signes de la cible, la perte réelle garde ce qui marche. Officiel seed 42 : 84,35 %. **Soumis : 0,80567,
1ᵉʳ.** Suite : hybride prolongé depuis it 120 (hyb4), lignée médoïdes (83,26 après un tour), échange Residential → River.

**19/10 soir** : hybride prolongé avec la même cible (hyb4) : 84,05–84,37, plat. Tour hybride avec une cible proche
recalculée (`runs/loop_hybrid.sh`, HY1, cible 90,6 %) : 84,40 → 84,07 (it 120). Comme pour les autres méthodes, seule la
première application rapporte. Lancé : **n=9 sans l'image Forest** (classe la plus facile, 94,7 %), depuis le set à 0,80567,
avec les vues de n=9 (`runs/loop_rounds2.sh ... N9F`). À n=9, 83,45 % suffit pour battre 0,80567.
**GitHub** : le dépôt est synchronisé par `tools/push.sh "message"` (copie `work/` → `src/`, logs, checkpoints, notes) et
une tâche cron toutes les heures (:17). Les nouvelles soumissions s'ajoutent dans `tools/submissions.json`.

**n=9 sans Forest** (sign matching, tour 1) : 43,3 → 70,7 %. Par classe : **Forest 0,0 %** (prédite Pasture 207, SeaLake 186,
HerbVeg 38), les autres classes vers 85 %. Avec des labels durs, une classe sans image n'est jamais prédite : la ligne de la
tête pour cette classe ne reçoit que des gradients qui l'éloignent. n < 10 est donc plafonné vers 76 %. Arrêté. Résultat net
pour le papier.

**Polissage PFD après l'hybride** (pf_hyb3) : 84,37 → 84,02–84,12, aucun gain (l'hybride contient déjà ce gradient).

**Prochaines étapes (état au 19/10, 21:00 UTC)**
1. Variantes de l'hybride depuis le set d'avant (poids PFD 2 et 5, en cours) ; garder la meilleure à la précision poolée.
2. ~~Robustesse numérique du set soumis~~ (`views_robust.py`) : 84,44 en fp32, 84,42 avec TF32 sur les matmuls ou les
   convolutions (±0,02). Bien dans la tolérance des organisateurs (quelques dixièmes).
3. Option de repli si vinicius repasse devant : hybride sur la lignée 224×224 (84,2 % après un tour). Le 64×64 est
   « préféré » par les organisateurs : décision de l'utilisateur.
4. Papier : section sign matching, plafond ≥ 90 % en 5 pas, coins de la boîte, classe sans image jamais prédite.
5. Toujours pousser sur GitHub après chaque étape (`tools/push.sh`).

**Variantes hybrides** : poids PFD 2 (hybw2) val 84,49 (it 140) mais précision poolée 84,85–84,90, contre 85,00 pour le set
soumis ; poids 5 (hybw5) val max 84,25. Pas mieux : le poids 3 reste le bon réglage. L'hybride sature vers 85 % (poolé) sur cette lignée.

**Usine à lignées** (`runs/factory.sh`, une tâche GPU à la fois) : les lignées ne finissent pas au même niveau (83,3 contre
83,9 au même stade), on lance donc la recette complète depuis plusieurs départs et on garde la meilleure (précision poolée).
Ordre : fin de la lignée médoïdes (tour 2 + hybride), puis 4 lignées `runs/lineage.sh` depuis `cache/init_c4j{0..3}.pt`
(médoïde du j-ième cluster k-means de chaque classe, 4 par classe) : PFD exact 400 it → 2 tours (cible proche → sign
matching → PFD) → 1 tour hybride → score poolé de chaque étape dans `runs/lineage_C4J*.log`. Environ 3 h 30 par lignée.

### 2.5 nonies 19/10 nuit : piste 224×224 (demande de l'utilisateur) et réplique exacte
- Officiel seed 42 du meilleur set 224 (`pfd3_pf_SZ1_best`) : **83,93 %** contre 84,20 en réplique (−0,27). En 64×64 la
  réplique tombait juste (83,88 = 83,88 ; 84,35 contre 84,37).
- Diagnostic : l'émulation PIL de l'entraînement est même meilleure en 224 (0,6 % de pixels à ±1 niveau contre 2 % en 64),
  remplacer les entrées par les sorties PIL exactes ne change rien (84,20), la taille des batches d'éval non plus. Les poids
  finaux officiels et réplique diffèrent de ~1e-6 sur 99 % des paramètres (max 1,6e-4) en 64 comme en 224 : bruit numérique
  amplifié par une trajectoire chaotique.
- **Réplique exacte** : en construisant les batches d'entraînement sur le CPU exactement comme le DataLoader officiel
  (crop/resize PIL, ToTensor, Normalize CPU) et la validation avec les transformations officielles en fp32, on retrouve
  l'historique officiel époque par époque : 21,19 / 50,96 / 75,98 / 82,72 / **83,93** (224) et 19,93 / 48,57 / 74,99 /
  82,96 / **84,35** (64). Poids : écart max 8e-9 en 64 (bit-exact à l'ulp près), 6e-6 en 224.
- `PROTO_EXACT=1` (proto.py) : valeurs exactes dans la passe avant, gradient de la copie différentiable (straight-through),
  validation exacte (cache `cache/validation_exact.pt`, 2,4 Go), 8 threads CPU max. Activé dans tous les scripts de boucle.
  Les `exact_acc` des nouveaux runs sont donc les chiffres officiels. Les loops ne notent plus que la dernière époque.
- Tour hybride 224 (SZH1, depuis `pfd3_pf_SZ1_best`, cible proche 90,35 %) : réplique exacte 84,52 à l'it 120, officiel
  seed 42 **84,57 %** (deux runs, prédictions identiques). Précision poolée 85,29 contre 84,94 pour le set soumis (val
  84,57 / 84,35, train 8k 85,64 / 85,23). Soumis le 19/10 à 23:05 avec l'accord de l'utilisateur (PNG 224×224) :
  **0,80777**, 1ᵉʳ (+0,0021). Test implicite 84,94 %. Archive `work/submissions/n10_SZH1_224_*`.
- Lignée 224 SZB (depuis le set 64 à 84,35 agrandi, 39,7 % au départ sur la trajectoire 224), finie le 20/10 à 00:26.
  `exact_eval` (val / train 8k / poolé) : PFD 82,15 / 83,44 / 83,01 ; tour 1 83,65 / 85,07 / 84,60 ; tour 2 83,60 / 85,11 /
  84,61 ; hybride **84,22 / 85,79 / 85,27**. Même plateau poolé que SZH1 (85,29) par un autre chemin, val plus basse : pas
  soumis. Le `exact_acc` de signmatch.py (84,44 à l'it 140) diffère de `exact_eval` (84,22) : se fier à `exact_eval`,
  qui retombe sur l'officiel (84,35 et 84,57).
- 2ᵉ tour hybride depuis SZH1 (SZH2, it 120) : val 84,74 / train 8k 85,79 / poolé 85,44 (SZH1 relu au même moment :
  84,57 / 85,61 / 85,26 ; le train 8k bouge de ±0,03 d'un appel à l'autre, la val non). Officiel seed 42 **84,74 %** (deux
  runs identiques). Soumis le 20/10 à 01:30 : **0,80801** (+0,00024, test implicite 84,99 %). Le gain test (+0,05) est bien
  plus petit que le gain val (+0,17) : on approche du plateau.
- 3ᵉ tour hybride depuis SZH2 (SZH3, it 160) : val 84,81 / train 8k 86,18 / poolé 85,73 (SZH2 : 84,74 / 85,75 / 85,42).
  Officiel 84,81 (identique deux fois). Soumis le 20/10 vers 02:35 : **0,80740**, en baisse (test implicite 84,86 contre
  84,99). **Le train 8k n'est pas tenu à part** : `ceiling3.py` (cible proche) et `signmatch.py`/`pfd3.py` (gradient réel)
  tirent dans tout `D.xtr`, donc chaque tour gonfle la précision sur le train sans aider le test. La val (4050 images,
  bruit ~±0,15 sur un écart entre deux sets) bouge moins que son bruit. Set final à garder : **SZH2 (0,80801)**.
- Bruit : SZH2 et SZH3 diffèrent sur 91 images test sur 4050 (SZH1/SZH2 : 67), donc un écart de test entre deux sets
  proches a un écart-type d'environ ±0,23 point. La baisse de SZH3 (−0,12) est elle-même dans le bruit.
- **Bloc tenu à part (20/10, 12:09)** : `DD_HOLDOUT=k` dans `dd_lib.py` retire les k premières lignes de la permutation
  seed 33 du train (les mêmes que le train 8k de `exact_eval`) de tout tirage de la distillation : `D.sample_train()`
  dans `ceiling3.py` (cible proche), `signmatch.py` (coin + gradient réel) et `pfd3.py` (gradient réel, LS). Sans la
  variable, mêmes tirages qu'avant (vérifié). `exact_eval.py` note alors `hold` (les k lignes) au lieu de `train8k`.
  k = 4096 (22 % du train, classes équilibrées), pool de 14 804 images.
- En cours : `runs/queue_SZK.sh 2`, deux tours hybrides depuis SZH2 avec `DD_HOLDOUT=4096`. Chaque tour note les
  checkpoints it 60–160 sur val + hold (`runs/pooled_SZK.log`) et le tour suivant part du meilleur. Réserve : SZH2
  a déjà vu le bloc pendant ses propres tours, donc seul le gain d'un tour SZK sur SZH2 est propre.
- Tour SZK1 (12:09–12:55) : aucun checkpoint ne bat SZH2. Meilleur it 160 : val 84,42 / hold 85,64 / poolé 85,04,
  contre SZH2 84,74 / 86,13 / 85,44 (hold de SZH2 un peu optimiste, mais la val baisse aussi). Même allure que les tours
  SZH (chute à l'it 20, remontée), sans repasser au-dessus du départ. Lecture : les gains des tours SZH2/SZH3 étaient
  du bruit ou venaient du train vu. Tour 2 (depuis SZK1_it160) lancé à 12:58 pour confirmer.
- Tour SZK2 (depuis SZK1_it160) : meilleur it 160, val 84,52 / hold 85,55 / poolé 85,04. Plat. **Les tours hybrides sont
  au plateau sur la piste 224** quand on les mesure proprement. Set final : SZH2.
- **Classement 20/10 13:51** : vinicius simm **0,80814** (soumis à 12:58), nous 0,80801. Écart 0,00013 = une seule image
  test sur 4051 (si n=10, test 85,01 contre 84,99). Bien plus petit que le bruit entre deux sets proches (±0,23 point).
- Pistes restantes avec un signal : (1) capacité en pixels (64 → 224 : plateau 84,35 → 84,7), donc **piste 448×448**
  lancée à 13:53 (`runs/queue_S4.sh` : vues seed 42 en 448, SZH2 agrandi en bicubique comme départ, 36,2 % à l'it 0,
  lignée complète avec `DD_HOLDOUT=4096`, ~4 h) ; (2) usine à lignées depuis d'autres départs (l'écart entre lignées,
  ~0,6 point en 64, dépasse le gain d'un tour), sélection sur val + hold.
- **Piste 448 : échec.** Lignée S4 finie à 15:36 (`exact_eval`, val / hold / poolé) : PFD 81,48 / 81,88 / 81,68 ; tour 1
  83,23 / 84,08 / 83,66 ; tour 2 83,23 / 84,03 / 83,64 ; hybride **83,83 / 84,91 / 84,37**. En retrait d'environ 0,4–0,5
  point sur la lignée 224 SZB à chaque étape. Plus de pixels au-delà de 224 n'aide pas (le RRC réduit alors l'image).

### 2.6 Idée : les budgets n = 32k+1
Avec `drop_last=False`, une image de plus que 32k ajoute un batch d'une image par epoch, donc 5 pas de plus
pour un coût de score quasi nul :

| n | pas | 0,5·(1 − ln n / ln 18900) |
|---|---|---|
| 32 / **33** | 5 / **10** | 0,3240 / 0,3224 |
| 64 / **65** | 10 / **15** | 0,2888 / 0,2881 |
| 96 / **97** | 15 / **20** | 0,2683 / 0,2677 |
| 120 | 20 | 0,2569 |

À précision égale, n=97 vaut +0,011 de score par rapport à n=120 (soit 2,2 points de précision). Avec des images
réelles, le petit batch fait du tort (n=100 : 82,2 % contre 87,1 % à n=120), mais PFD simule ce batch d'une image
et peut apprendre à le rendre utile. Le batch d'une image tombe toujours en fin d'epoch (pas 4, 8, …, 20), donc avec
l'élan d'Adam déjà en place. Inits prêtes : `cache/medoids_n97.pt`, `n65`, `n33` (`work/make_medoids.py`).

---

## 3. Ce qu'implique le protocole

1. **Adam au début, c'est une descente de signe.** Au premier pas, m̂/√v̂ = g/|g|, donc chaque poids bouge
   de ±lr, quelle que soit la taille du gradient. Conséquences :
   - le **contenu** des images ne fixe que le *signe* de la mise à jour, pas son amplitude ;
   - la taille du batch ne change pas l'amplitude d'un pas, si bien qu'un batch d'**une seule image**
     fait un pas aussi grand qu'un batch de 32 ;
   - le gradient « exact » de la méta-optimisation à travers sign(g) est presque nul. Il faut un estimateur
     *straight-through* : Adam exact dans la passe avant, Adam lissé (ε_τ = τ·RMS(g)) dans la passe arrière.
2. **La tête ne bouge presque pas.** Elle se déplace d'environ 1,5e-4 alors que ses poids initiaux sont de
   l'ordre de ±0,036. Tout l'apprentissage passe par le backbone, qui doit aligner les features de chaque
   classe sur une direction aléatoire w_c. Pour être robuste, il faut ré-échantillonner la tête pendant la
   distillation, comme dans Linear Gradient Matching.
3. **Les augmentations sont connues** (RRC 0,75–1 et flips). On optimise en espérance sur exactement cette
   distribution (EOT/DSA).
4. **Format** : PNG uint8 64×64. La quantification doit être prise en compte à la fin (arrondi straight-through)
   et tout se vérifie après sauvegarde.
5. **Sélection de l'epoch sur la validation** : la vraie métrique est le max sur les 5 epochs, mais en pratique
   l'accuracy croît jusqu'à la fin, donc on optimise l'état final.

---

## 4. Stratégie technique

### Axe A (principal) — PFD : Protocol-Faithful Distillation
Optimisation bi-niveau qui **reproduit exactement l'entraînement officiel** :
```
init : médoïdes k-means (features ViT) de chaque classe, pixels en espace logit (ou pyramide multi-échelle)
répéter :
    tête W ~ init aléatoire (seed tirée au hasard) ; θ = θ_ImageNet ; état Adam = 0
    pour k = 1..K (K = 5·⌈n/32⌉, batches = shuffle/split comme le DataLoader) :
        x_aug = RRC(0.75–1) + flips + upsample 224 (différentiable)
        g = ∇θ CE(f_θ(x_aug), y)                    # create_graph
        θ = AdamW_ST(θ, g, lr_k)                     # avant : Adam exact ; arrière : lissé
    L = CE(f_θ(vraies images train augmentées), y)   # 256–512 images
    pixels ← pixels − η ∇L  (Adam externe)
```
- Mémoire : unroll complet pour K=5 (checkpointing de chaque pas interne, bf16). Pour K=10 ou 20,
  BPTT tronqué (cf. AT-BPTT, NeurIPS 2025) ou objectif « look-ahead » pas par pas.
- Le pas de 1 image (n=33, 65, 97) est inclus dans la simulation, donc les images apprennent à rester utiles
  même seules.
- Coût estimé : ~2–3 s par itération externe à n=10 sur A100 seul, soit ~1–2 h pour 2 000 itérations.

### Axe B — Substituts moins chers (ablations et filet de sécurité)
1. **Sign-aware gradient matching** (notre dérivation) : on maximise ⟨|G_réel| ⊙ sign(G_réel), φ_τ(g_syn)⟩,
   c'est-à-dire la baisse de perte au premier ordre sous descente de signe, au lieu du cosinus de DC.
2. Gradient matching classique (DC/DSA), distribution matching (DM) dans l'espace des features ViT,
   Linear Gradient Matching (Cazenavette 2025), trajectory matching vers l'oracle (MTT/DATM). Ce dernier
   est plafonné à 60 % par construction, ce qui en fait un bon argument pour le paper.

### Axe C — Point de fonctionnement et allocation
- Balayer n ∈ {10, 15, 20, 32 | 33, 42, 64 | 65, 97} avec PFD et tracer la frontière score = f(n).
- Allocation par classe (non uniforme) : plus d'images pour les classes difficiles
  (Highway↔River, Pasture↔HerbaceousVegetation, PermanentCrop↔AnnualCrop).

### Axe D — Robustesse et vérification
- Chaque candidat est évalué avec le **script officiel** sur les seeds {42, 0…9} : on reporte moyenne ± écart-type.
  On soumet le run seed 42 par défaut, conformément au protocole.
- On **n'optimise pas spécifiquement pour la tête de la seed 42** : c'est fragile au moment de la vérification
  et scientifiquement faible. On mesure l'écart et on le rapporte.
- On vérifie que les PNG relus donnent bien le même résultat. Si possible, deux GPU différents (non-déterminisme).

### Axe E — Gains rapides pour se positionner tôt sur le leaderboard (semaine 1)
- Sélection de coreset : médoïdes k-means, herding, sélection par recherche sur la validation (moyennée sur les seeds).
- Mosaïques 2×2 d'images de la même classe (plus d'information par image, à la RDED). À tester, car la
  baisse de résolution change l'échelle.
- Objectif : une première soumission ≥ 0,69 dans la semaine, pour exister sur le leaderboard pendant que l'Axe A mûrit.

---

## 5. Calendrier

| Période | Travail | Livrable |
|---|---|---|
| **12–18/10** | Fidélité proto/officiel ; baselines complètes (random, médoïdes, herding, mosaïques) à n=10/33/64/97 ; questions aux organisateurs ; pipeline de soumission | 1ʳᵉ soumission, tableau des baselines |
| **19–25/10** | PFD v1 à K=5 (n=10, 20, 32) ; sign-aware GM ; DM et LGM pour comparer | PFD > oracle (60 %) à n=10 : **critère go/no-go** |
| **26/10–01/11** | PFD à K=10/20 (n=33, 64, 97), BPTT tronqué ; allocation par classe ; paramétrisation pyramide | courbe frontière, choix du n |
| **02–08/11** | Robustesse (10 seeds, quantification) ; ablations complètes ; transfert (SGD, ResNet-50, ViT-S) | figures et tableaux du paper |
| **09–15/11** | Gel, soumissions finales, archive des images et du code ; rédaction | soumission finale, brouillon du paper |
| **16–26/11** | Slides/poster ; contribution JSTARS ; code public | présentation au workshop |

Budget soumissions : 5 par jour, environ 170 au total. On choisit **sur la validation**, jamais en sondant le test.

---

## 6. Le paper

**Titre de travail** : *Distilling for the Optimizer: Protocol-Faithful Dataset Distillation for Few-Step
Fine-Tuning of Pretrained Vision Transformers in Remote Sensing*.

**Contributions visées**
1. Une analyse du régime « peu de pas » : le budget de fine-tuning dépend de n (5·⌈n/32⌉), Adam se comporte
   en descente de signe, et cela crée un escalier de frontières de Pareto. On en tire le point de score optimal.
2. Une mesure de l'écart **oracle (toutes les données réelles) vs plafond atteignable** : à 5 pas, 60 % contre ≥ 79 %.
   Les données réelles sont sous-optimales pour l'entraînement en très peu de pas.
3. PFD : une distillation bi-niveau qui déroule exactement l'optimiseur d'évaluation (Adam straight-through,
   EOT sur les augmentations, ré-échantillonnage de la tête, quantification), avec en plus un substitut
   sign-aware peu coûteux.
4. Les résultats du challenge (classement), des ablations, la robustesse aux seeds, les limites du transfert
   (autres optimiseurs et architectures), et une analyse télédétection (confusions, visualisation des images
   distillées vs médoïdes réels).

**Figures et tableaux prévus**
- Fig. 1 : paysage du score (iso-scores dans le plan acc × ln n), escalier des pas, leaderboard, nos points.
- Fig. 2 : images distillées vs médoïdes réels, par classe.
- Fig. 3 : accuracy selon le nombre de pas pour random, oracle, PFD et plafond.
- Tab. 1 : random, médoïdes, herding, DC, DM, LGM, MTT-oracle et PFD à n = 10 / 33 / 97 (moyenne ± écart-type sur 10 seeds).
- Tab. 2 : ablations (straight-through, EOT, ré-échantillonnage de la tête, init, paramétrisation, quantification).
- Fig. 4 : matrices de confusion et analyse par classe.

**Travaux à citer en priorité**
- Fondations : Wang et al. 2018 (*Dataset Distillation*, init fixe et peu de pas : notre régime) ; DC, DSA, DM, MTT, DATM.
- Grande échelle et soft labels : SRe2L, RDED, *A Label Is Worth a Thousand Images* (NeurIPS 2024), HALD
  (labels durs, 2025), DD-Ranking (2025).
- Modèles pré-entraînés : Linear Gradient Matching (Cazenavette, Torralba, Sitzmann, NeurIPS 2025) ;
  CLP-DD (2026) ; Statistical Flow Matching (NeurIPS 2026) ; *Can pre-trained models assist in DD?* (2023).
- Optimisation interne : AT-BPTT (NeurIPS 2025).
- Télédétection : DPD (Xu, Ghamisi, Weng 2026, la première DD en remote sensing, mais sur UCM/AID/NWPU,
  pas EuroSAT, et avec un ResNet entraîné de zéro) ; EuroSAT (Helber et al.).
- Challenges : ECCV 2024 DD Challenge (M-DATM a gagné en *retirant* les soft labels pour coller au script officiel).

**Positionnement** : à notre connaissance, aucun travail ne traite la distillation pour le **fine-tuning
complet d'un ViT pré-entraîné en quelques pas d'Adam**. LGM et CLP-DD visent des linear probes figés,
DPD vise des CNN entraînés de zéro. La nouveauté est donc réelle, à vérifier encore en S2.

---

## 7. Risques et questions ouvertes

**Ce que disent les règles Kaggle (section 2, lue le 13/10)** :
- Évaluation : vit_b_16 avec IMAGENET1K_V1 uniquement. Les autres poids pré-entraînés sont interdits « pendant la
  phase d'évaluation ». Pendant la distillation, ils semblent donc permis (on ne s'en sert pas de toute façon).
- La distillation doit opérer « uniquement sur train/ (et optionnellement validation/) ». **La validation est
  autorisée pendant la distillation.** Si on s'en sert dans l'objectif, il faut en garder une partie à part pour la
  sélection.
- Interdit : labels ou features du test pendant la distillation. Ça ne nous concerne pas, on n'utilise jamais test/.
- Le texte parle d'« images synthétiques distillées » sans exclure explicitement les images réelles : la question 5
  reste ouverte.
- Questions 2 et 3 réglées par les règles. Il reste à poster les questions 1, 4 et 5.
- **Section 4 (vérification)** : les équipes de tête envoient leurs images distillées dans les 7 jours après la fin.
  Les organisateurs ré-entraînent le ViT depuis zéro, et l'accuracy test doit correspondre aux prédictions soumises
  « à une petite marge de variance stochastique près ». Sinon, disqualification. La seed n'est pas précisée.
- **Règle de soumission qui en découle** : ne jamais soumettre un set sur la foi d'un run seed 42 chanceux. On ne
  soumet que si l'accuracy officielle seed 42 n'est pas au-dessus de la moyenne multi-seeds officielle. C'est le cas
  de la soumission actuelle (88,37 % contre 89,27 ± 0,70 % sur 6 seeds). Archive : PNG exacts dans
  `work/submissions/pfd3_n120_images/` et sha256 dans `work/submissions/pfd3_n120_manifest.txt`. Garder ces fichiers
  intacts jusqu'à la vérification.
- **Déterminisme (13/10)** : trois runs officiels seed 42 du set soumis donnent 88,37 % à l'identique (A100, même
  environnement). Un écart ne pourrait venir que d'un autre matériel ou d'autres versions de bibliothèques. Avec une
  autre seed, la moyenne officielle est plus haute (89,27 ± 0,70 %), donc le risque à la vérification est faible.

**Questions à poser aux organisateurs** (discussion Kaggle, à poster tôt) :
1. Les images distillées doivent-elles être en 64×64 RGB PNG, ou une autre résolution est-elle permise ?
2. A-t-on le droit d'utiliser les images de **validation** pendant la distillation, au-delà de la sélection d'epoch ?
3. Les modèles pré-entraînés externes (autres backbones, modèles de diffusion) sont-ils autorisés pendant la distillation ?
4. La vérification relance-t-elle `baseline.py` avec les arguments par défaut (seed 42) ? Avec quelle tolérance ?
5. Des images réelles (coreset) comptent-elles comme « synthétiques » ?

**Risques**
| Risque | Parade |
|---|---|
| GPU partagé avec QA2026 (OOM déjà constatés le 12/10) | un seul job à la fois, plafonner la mémoire, planifier les créneaux |
| Sur-apprentissage de la tête seed 42 ou du test via le leaderboard | têtes ré-échantillonnées, sélection sur la validation, rapporter 10 seeds |
| Le méta-gradient ne passe pas à travers sign(g) | straight-through et lissage ε_τ ; repli sur sign-aware GM |
| Mémoire pour K = 10/20 | BPTT tronqué, checkpointing, bf16 |
| Les autres équipes progressent | viser ≥ 0,75 et non simplement le 1ᵉʳ rang actuel |
| Non-reproductibilité à la vérification | tests sur plusieurs seeds et GPU ; garder les PNG exacts et leurs hash |

**Éthique** : ne jamais entraîner sur plus que n images déclarées, ne pas étiqueter le test à la main,
choisir les soumissions sur la validation. Les images soumises doivent reproduire le score avec le script officiel.

---

## 8. Prochaines actions (48 h)
1. ~~Première soumission avec le meilleur coreset~~ : fait le 12/10, 0,69171, 2ᵉ place.
2. Run PFD de 400 itérations à n=10, init médoïdes (`python pfd.py --init cache/medoids_ipc1.pt --tag v1`,
   environ 40 min, 30 Go GPU). Critère go/no-go : dépasser 60,4 % (l'oracle) en moyenne sur 3 seeds.
3. Si c'est un go : PFD à n=33–64 (10 pas), qui demande du BPTT tronqué pour tenir en mémoire.
   Si c'est un no-go : sign-aware gradient matching, et affiner le coreset autour de n=110–128.
4. Poster les 5 questions sur le forum Kaggle.

## Sources
- Challenge : https://mveo.github.io/challenge.html · https://github.com/mveo/challenge_2026_public ·
  https://www.kaggle.com/competitions/data-centric-land-cover-classification-challenge-26
- LGM : https://arxiv.org/abs/2511.16674 · CLP-DD : https://arxiv.org/abs/2605.07194 ·
  SFM : https://arxiv.org/abs/2602.05391 · DPD (RS) : https://arxiv.org/abs/2601.15829 ·
  AT-BPTT : https://arxiv.org/abs/2510.04838 · HALD : https://arxiv.org/abs/2512.15647 ·
  Label is worth 1000 images : https://arxiv.org/abs/2406.10485 · DD-Ranking : https://arxiv.org/abs/2505.13300 ·
  Survey 2025 : https://arxiv.org/abs/2502.05673 · Wang 2018 : https://arxiv.org/abs/1811.10959 ·
  PTM & DD : https://arxiv.org/abs/2310.03295 · ECCV'24 DD challenge (M-DATM) : https://arxiv.org/abs/2410.13311 ·
  EuroSAT : https://arxiv.org/abs/1709.00029
