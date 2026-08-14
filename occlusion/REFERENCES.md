# Occlusion / perturbation interpretability — references

The method here is **occlusion (perturbation-based) sensitivity**: blank part of the input, see how
much the prediction changes. It is *model-agnostic* (only needs a forward pass), which is why it runs
identically across CNN, VAE, ShapeEmbed and RegionProps — including the non-differentiable RF probe.

## Cite for the method
- **Zeiler, M.D., Fergus, R.** *Visualizing and Understanding Convolutional Networks.* ECCV 2014.
  — the original **occlusion sensitivity** (slide a gray patch, map the drop in the target score). The
  canonical citation for what this script does.
- **Petsiuk, V., Das, A., Saenko, K.** *RISE: Randomized Input Sampling for Explanation of Black-box
  Models.* BMVC 2018. — random-mask perturbation, explicitly **black-box / model-agnostic**; the best
  cite if you want to stress that the identical procedure applies to every method and probe.
- **Ribeiro, M.T., Singh, S., Guestrin, C.** *"Why Should I Trust You?": Explaining the Predictions of
  Any Classifier (LIME).* KDD 2016. — perturbation-based, model-agnostic; supports the "same
  explanation method for all four representations" framing.

## Cite as the gradient-based alternative (CNN only)
- **Selvaraju, R.R., et al.** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
  Localization.* ICCV 2017. — what a reviewer likely had in mind; note it needs conv feature maps +
  a differentiable target, so it applies to the CNN (and, with a linear probe, the VAE) but **not** to
  the shape-based methods — which is exactly why we use occlusion instead.
- **Fong, R., Vedaldi, A.** *Interpretable Explanations of Black Boxes by Meaningful Perturbation.*
  ICCV 2017. — optional; perturbation masks learned rather than swept.

## Honesty note (put in the paper, do NOT skip)
Background occlusion is a **no-op for the shape-based methods by construction** (they operate on the
mask/contour and never see the background). Report that as a **design property / trade-off**, not as a
score they "won" — otherwise it is a circular comparison. The informative, non-trivial background test
is for the **image-based** methods (CNN, VAE); the **body-region** occlusion is informative for all
four. State the trade-off both ways: shape input buys artifact-invariance at the cost of discarding
non-shape image content (which the shape-residual $h^2$ result shows the CNN actually uses).
