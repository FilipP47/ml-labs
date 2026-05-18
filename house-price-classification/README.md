Task: Classify apartments into 0: cheap (<=100k), 1: average (<=350k), 2: expensive (>350k) using property features.

Solution: PyTorch MLP classifier with engineered features (parking, age, facilities, neighborhood), class-weighted CrossEntropyLoss to address imbalance.
