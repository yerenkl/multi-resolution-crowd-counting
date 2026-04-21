# Research Questions

Four exploratory experiments, in the order they appear in the paper/poster.

## 1. Baselines: HR-only / LR-only / HR+LR mixed training
How much does performance drop when training HR-only and testing on LR? Does mixed training recover it?

## 2. Downscaling algorithm study
Which degradation types hurt most? And does the model overfit to the specific resize kernel it was trained on — i.e., does performance collapse when the train/test kernel mismatches? Do results on synthetic LR predict results on real optical zoom pairs?

## 3. Consistency loss (HR+LR regularization)
Does explicitly penalizing disagreement between HR and LR predictions of the same crop improve LR robustness beyond what mixed training alone achieves?

## 4. DANN — resolution-adversarial training
Does forcing the encoder to be blind to input resolution (via gradient reversal) improve robustness further, and does it transfer to real optical zoom pairs?

---

The real optical zoom pairs (61 unlabeled HR/LR pairs from the supervisor) run through all four experiments as the real-world validity check — a model that only does well on synthetically downscaled NWPU but fails here has learned the resize kernel, not the crowd.
