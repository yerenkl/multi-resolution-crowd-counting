# Docs Wiki Map

This folder is organized as a small wiki of the most important project context.

## Structure

- overview/
  - problem.md: Core problem and motivation
  - datasets.md: Short dataset-level overview
  - downscaling_and_labels.md: Why synthetic LR labeling is valid
  - augmentation_notes.md: Practical caveats for degradation pipelines
  - research_questions.md: The four research questions and experiment order
  - training_from_scratch.md: Why all experiments start from raw CLIP weights
- data/
  - data.md: Detailed dataset reference (paths, formats, counts)
- ideas/
  - hr_only_baseline.md
  - lr_only_baseline.md
  - hr_lr_consistency_loss.md
  - teacher_student_distillation.md
  - single_model_consistency_regularization.md
  - super_resolution_branch.md
  - scale_adversarial_training.md
  - multi_scale_feature_fusion.md
  - self_supervised_pretraining.md
- methods/
  - finetune_resolution_aug.md: Current training method summary

## Suggested reading order
1. overview/problem.md
2. overview/research_questions.md
3. overview/datasets.md
4. data/data.md
5. methods/finetune_resolution_aug.md
6. ideas/hr_lr_consistency_loss.md

## Notes
- The old flat docs were split into topic-focused files for faster navigation.
- Idea notes are intentionally concise: one file per idea for easy iteration.
