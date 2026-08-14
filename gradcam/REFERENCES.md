# Grad-CAM cross-architecture visualization — references

Gradient-based class-activation mapping: backprop a scalar target into the last conv block, weight the
feature maps by the global-average-pooled gradients, ReLU, upsample. Highlights the image regions that
most raise the target score.

## Cite for the method
- **Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.** *Grad-CAM: Visual
  Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017. — the canonical citation
  for what this folder does.
- **Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.** *Learning Deep Features for
  Discriminative Localization (CAM).* CVPR 2016. — the original class-activation map (Grad-CAM
  generalizes it to any CNN without architectural change).

## What we target (and why it's fair)
Grad-CAM needs a differentiable scalar. Our backbones are feature extractors read out by a downstream
probe, so we attach a **matched** differentiable linear severity head to each backbone (a
StandardScaler + LogisticRegression folded into one nn.Linear, fit on that method's train
representation) and backprop the expected severity **E[k] = Σ k·softmax(logits)ₖ**. Using the *same*
readout form on the CNN and the VAE encoder means map differences reflect the **backbone**, not two
different classifiers.

## Honesty note (put in the paper, do NOT skip)
Grad-CAM is an **image-space** method: it needs convolutional feature maps over the pixels.
- **CNN** and the **VAE encoder** are ResNet-18 over the image → directly comparable pixel heatmaps.
- **ShapeEmbed** is a ResNet over a **distance matrix**, not the image; its gradient lives in
  shape/contour space. We therefore report **saliency per contour point** drawn on the fish outline —
  analogous ("does the shape model attend to the deformed segments?") but a **different modality**;
  do not present it as the same map as the pixel heatmaps.
- **RegionProps** has no conv backbone → Grad-CAM is not applicable.

For pixel-space attributions of the shape/hand-crafted methods, use the **occlusion** analysis instead
(model-agnostic, forward-pass only). Do not mix Grad-CAM and occlusion on one figure — they are
different explanation methods and combining them invites the circularity flagged in the occlusion notes.
