import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
EPOCHS = 150
BATCH_SIZE = 64
LR = 0.005


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def price_to_class(price):
    if price <= 100000:
        return 0
    elif price <= 350000:
        return 1
    else:
        return 2


def calc_macro_accuracy(pred_targets, targets):
    accuracies = []
    targets = np.array(targets)
    pred_targets = np.array(pred_targets)

    for i in range(3):
        mask = (targets == i)
        if mask.sum() == 0:
            continue
        class_correct = (pred_targets[mask] == targets[mask]).sum()
        accuracies.append(class_correct / mask.sum())

    return float(np.mean(accuracies))


def add_features(df):
    df = df.copy()
    
    current_year = datetime.now().year
    
    df["TotalParking"] = df["N_Parkinglot(Ground)"] + df["N_Parkinglot(Basement)"]
    df["Age"] = current_year - df["YearBuilt"]
    df["FacilitiesScore"] = df["N_FacilitiesInApt"] + df["N_FacilitiesNearBy(Total)"]
    df["NeighborhoodScore"] = df["N_SchoolNearBy(Total)"] + df["N_FacilitiesNearBy(Total)"]

    return df


class ApartmentNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.model(x)


def prepare_data(train_df, test_df):
    train_df["PriceClass"] = train_df["SalePrice"].apply(price_to_class)
    print("Rozkład klas w zbiorze treningowym (ilość):")
    print(train_df["PriceClass"].value_counts())
    print("\nRozkład klas w zbiorze treningowym (%):")
    print(train_df["PriceClass"].value_counts(normalize=True))

    X = train_df.drop(columns=["SalePrice", "PriceClass"])
    y = train_df["PriceClass"].values

    X = add_features(X)
    test_df = add_features(test_df)

    X = pd.get_dummies(X)
    test_df = pd.get_dummies(test_df)

    test_df = test_df.reindex(columns=X.columns, fill_value=0)

    return X, y, test_df


def get_class_weights(y):
    counts = np.bincount(y, minlength=3)
    weights = len(y) / (3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def evaluate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        logits = model(X_val_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    score = calc_macro_accuracy(preds, y_val)
    return score, preds


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    # Obliczamy wagi klas, które są odwrotnie proporcjonalne do ich częstotliwości występowania
    class_weights = get_class_weights(y_train)
    # Przekazujemy wagi do CrossEntropyLoss, aby poradzić sobie z niezbalansowanym zbiorem danych
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_score = -1
    best_state = None
    patience = 20
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        val_score, val_preds = evaluate_model(model, X_val, y_val)

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            unique, counts = np.unique(val_preds, return_counts=True)
            clean_dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"Epoch {epoch:03d} | loss={total_loss/len(X_train):.4f} | val_score={val_score:.4f} | pred_dist={clean_dist}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_score



def main():
    seed_everything(SEED)

    train_df = pd.read_csv("train_data.csv")
    test_df = pd.read_csv("test_data.csv")

    X, y, X_test = prepare_data(train_df, test_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        # stratify=y
        # Gwarantuje, że proporcje klas w zbiorach treningowym i walidacyjnym 
        # będą identyczne jak w całym zbiorze. Chroni to przed sytuacją, w której
        # najrzadsza klasa wylądowałaby w całości poza zbiorem treningowym
        stratify=y
    )

    zero_fill_cols = [col for col in X_train.columns if col.startswith('N_') or 'Parking' in col]
    
    # Wypełniamy zerami kolumny, które reprezentują liczbę obiektów
    X_train[zero_fill_cols] = X_train[zero_fill_cols].fillna(0)
    X_val[zero_fill_cols] = X_val[zero_fill_cols].fillna(0)
    X_test[zero_fill_cols] = X_test[zero_fill_cols].fillna(0)

    # Wstawienie '0' dla kolumn takich jak
    # 'YearBuilt' stworzyłoby bez sensu wartości
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)


    model = ApartmentNN(input_dim=X_train.shape[1]).to(DEVICE)

    model, best_score = train_model(
        model,
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR
    )

    print("Best validation score:", best_score)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=DEVICE)
        logits = model(X_test_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()


    pd.DataFrame(preds).to_csv("predictions.csv", index=False, header=False)
    print("Zapisano predictions.csv")


if __name__ == "__main__":
    main()