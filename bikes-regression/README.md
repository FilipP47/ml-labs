# bikes-regression

Overview
 - Task: predict total bike rentals (`cnt`) from historical features (weather, time, weekday/holiday).

Data
 - Tabular dataset with time and weather features.

Solution
 - Model: simple feedforward regressor (PyTorch) trained on `log1p(cnt)`.
 - Preprocessing: `StandardScaler` for numeric features.
