# house-price-classification

Overview
 - Task: classify apartments into price categories (e.g., cheap / average / expensive) based on property features.

Data
 - Tabular data with features such as area, number of rooms, building age, parking, location, NeighborhoodScore, etc.

Solution
 - Model: MLP (PyTorch) with basic feature engineering.
 - Loss: `CrossEntropyLoss` with class weights to handle imbalance.
