Here are ideas that go beyond just resolution augmentation, ranked by ambition and potential impact:

1. Teacher-student with density map distillation
Use the HR prediction from a frozen pretrained teacher as soft supervision for a student trained on LR versions. The student learns from dense spatial density maps instead of sparse point annotations — much richer signal at low resolution. This is the proper "close the HR/LR gap" approach because you're directly telling the student "when this was HR, the density looked like X, now match it at LR."

2. Consistency regularization within a single model
At each training step, feed both the HR crop and its downscaled version through the same model. Add a loss that forces the two predictions to match. You're explicitly penalizing the model when it gives different counts for the same crowd at different resolutions — directly targeting the exact problem you're solving.

3. Add a super-resolution branch (the Xie et al. [6] direction)
Put a small SR module before the encoder that learns to upsample LR features. Train it jointly with the counting objective. The model effectively "hallucinates" what the HR version would look like before counting.

4. Scale-adversarial training
Add a discriminator that tries to predict the input resolution from the encoder features. Train the encoder to fool the discriminator. This forces the feature representation to be resolution-invariant — two crops at different scales should produce indistinguishable features.

5. Multi-scale feature fusion in the backbone
Modify the ViT to process multiple scaled versions of the input in parallel and fuse their features. The model gets to see the same scene at multiple zoom levels simultaneously and learns which features matter at which scale.

6. Wikimedia self-supervised pretraining
Before fine-tuning on NWPU, do a round of self-supervised pretraining on the unlabeled wikimedia crowds using contrastive consistency (same scene HR vs LR should have similar representations). This teaches the encoder resolution invariance without needing any labels at all.

The most ambitious + scientifically novel combination would be 2 + 3 + 6: self-supervised resolution-invariant pretraining on wikimedia, then fine-tuning on NWPU with consistency loss and an SR branch. That's a legitimate paper-level contribution.

My honest recommendation: pick 2 (consistency) as the main method. It's elegant, well-motivated, easy to implement, and directly targets your metric. Then if time permits, add 3 (SR module) as a second experiment.