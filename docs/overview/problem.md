# Problem Statement

Crowd counting models usually perform well on high-quality images but degrade when people appear as tiny, blurry blobs at low resolution.

This project studies:
- How much performance drops under lower resolution
- Which training strategies improve robustness
- Whether robustness transfers from synthetic low-res data to real optical zoom pairs

Why this matters:
- Real deployment often involves distant cameras and compressed video
- Low-res regions are hard even for annotators, so labels can be noisy in practice
