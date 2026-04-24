# Idea: Teacher-Student Distillation

Use a strong HR teacher to generate soft supervision (counts or density maps), and train an LR student to match it.

Goal:
- Transfer fine-grained HR signal into LR model behavior
- Reduce dependence on sparse point-only supervision

Complexity:
- Higher training cost and extra model management
