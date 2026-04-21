# Idea: Single-Model Consistency Regularization

Use one model, pass HR and LR versions of the same crop, and add a consistency penalty between predictions.

Goal:
- Enforce resolution invariance directly in one training loop
- Keep architecture simple compared with teacher-student setups
