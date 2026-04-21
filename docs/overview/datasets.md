# Dataset Overview

## NWPU-Crowd (primary)
- Source path: /dtu/blackhole/02/137570/MultiRes/NWPU_crowd
- Labeled point annotations
- Official splits: train (3109), val (500), test (1500, hidden labels)
- Used for training and quantitative evaluation

## Supervisor Zoom Pairs (real HR/LR)
- Source path: /dtu/blackhole/02/137570/MultiRes/test
- Real optical pairs of the same scene: *_hr.jpg and *_lr.jpg
- No labels
- Used for prediction consistency checks between HR and LR views

## Web-sampled crowds
- Unlabeled data promised but not currently part of the default pipeline
- Relevant only if pseudo-labeling/self-supervised extensions are pursued

For detailed dataset structure and formats, see ../data/data.md.
