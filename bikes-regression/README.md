Task: Predict total number of bike rentals (`cnt`) from historical features (weather, time, weekday/holiday).

Solution: PyTorch feedforward regression trained on log1p(cnt) with StandardScaler. The model predicts on evaluation data.
