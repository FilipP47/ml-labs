Task: Train a generative model to produce new traffic sign images; supply 1000 generated samples for evaluation (FID).

Solution: Conditional DCGAN implemented in `solution.ipynb` (spectral norm, label embeddings). Saves 1000 samples as a tensor of shape [1000, 3, 32, 32].
