# Idea: HR+LR with Consistency Loss

Feed paired HR and synthetic LR views of the same scene during training and penalize disagreement in predicted counts.

Goal:
- Learn representation that is stable across resolution changes
- Improve LR robustness without sacrificing HR performance

Why this is a strong primary method:
- Directly optimizes the invariance target of the project
