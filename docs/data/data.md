# Dataset Reference

## 1. NWPU-Crowd

**What it is:** Large-scale crowd counting benchmark with point annotations. Images are scraped from the internet and cover a wide range of scenes (concerts, protests, pilgrimages, streets) at varying densities and resolutions. It is the primary training and evaluation dataset for this project.

**HPC path:** `/dtu/blackhole/02/137570/MultiRes/NWPU_crowd/`

**Folder structure:**
```
NWPU_crowd/
  images/       # JPEG images (*.jpg)
  jsons/        # JSON point annotations (*.json)
  mats/         # Same annotations in MATLAB format (UCF-QNRF compatible)
  train.txt     # 3,109 images
  val.txt       # 500 images
  test.txt      # 1,500 images (labels withheld — leaderboard submission only)
```

**Split file format:**
- `train.txt` / `val.txt`: three columns per line — `<image_id> <luminance_label> <scene_level>`
- `test.txt`: one column per line — `<image_id>`

**Annotation format (JSON):**
```json
{
  "img_id":    "0001.jpg",
  "human_num": 45,
  "points":    [[x, y], ...],   // one [x, y] float pair per head, pixel coords
  "boxes":     [[xmin, ymin, xmax, ymax], ...]  // head bounding boxes
}
```

**Scale:** Images range from ~1700×1100 to ~5400×3600 px. Crowd counts range from 0 to ~10,000+, with a mean around 380.

**Usage in this project:** Train on `train`, evaluate on `val`. LR variants are generated on-the-fly by downscaling during training — no separate LR dataset is needed.

---

## 2. Supervisor's Zoom Pairs

**What it is:** 61 real optical HR/LR image pairs of the same crowd scenes, captured with different zoom levels. Unlike the synthetic LR images derived from NWPU, these are genuine camera captures at two different focal lengths. They serve as the ground-truth check that the model has learned real LR robustness — not just how to invert a specific resize kernel.

**HPC path:** `/dtu/blackhole/02/137570/MultiRes/test/`

**Folder structure:**
```
test/
  0/
    0_hr.jpg
    0_lr.jpg
  1/
    1_hr.jpg
    1_lr.jpg
  ...
  60/
    60_hr.jpg
    60_lr.jpg
```

**Scale:** HR images range up to ~8400×5600 px; LR counterparts are roughly 2–3× smaller on each axis (width ratio: min 1.2×, max 7.5×, mean ~2.8×). The ratio is not uniform across pairs.

**Annotations:** None. Labels do not exist for these images.

**Usage in this project:** Evaluation only — consistency check between HR and LR count predictions for the same scene. A well-trained model should produce similar counts for both views of the same crowd.

---

## 3. Web-Sampled Crowd Images

**What it is:** A collection of unlabeled crowd images promised for this project. Would only be useful if pursuing a pseudo-labeling / Student-Teacher training approach.

**HPC path:** Not yet present under `/dtu/blackhole/02/137570/MultiRes/`.

**Usage in this project:** Ignore until available and until pseudo-labeling is on the roadmap.
