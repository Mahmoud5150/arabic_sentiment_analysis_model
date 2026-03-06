import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv("sampled92k.csv")

X_text = df["clean_text"]
Y = df["label"]

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=70000,
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X = vectorizer.fit_transform(X_text)

X_torch = torch.tensor(X.toarray(), dtype=torch.float32)
Y_torch = torch.tensor((Y.values == 1).astype(int), dtype=torch.long)

class Anlp(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)
    
X_temp, X_test, Y_temp, Y_test = train_test_split(
    X_torch, Y_torch,
    random_state=42,
    test_size=0.2,
    stratify=Y_torch
)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp,
    random_state=42,
    test_size=0.25,
    stratify=Y_temp
)

model = Anlp(X_train.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

best_f1 = -1.0
best_epoch = 0
best_state = None
no_improve = 0
patience = 20
epochs = 300

train_loader = DataLoader(
    TensorDataset(X_train,Y_train),
    batch_size=64,
    shuffle=True
)

evla_loader = DataLoader(
    TensorDataset(X_val,Y_val),
    batch_size=512,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(X_test,Y_test),
    batch_size=512,
    shuffle=False
)

for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        out = model(xb)
        optimizer.zero_grad()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        
        for xb, yb in evla_loader:
            preds = model(xb).argmax(dim=1)
            all_preds.append(preds)
            all_true.append(yb)

        val_preds = torch.cat(all_preds)
        val_true = torch.cat(all_true)

    f1_val = f1_score(
        val_true.numpy(),
        val_preds.numpy(),
        average="macro"
    )

    if f1_val > best_f1:
        best_f1 = f1_val
        best_epoch = epoch
        best_state = {k: v.cpu().clone()
                      for k, v in model.state_dict().items()}
        no_improve = 0

    else:
        no_improve += 1

    if epoch % 1 == 0:
        print(f"Epoch: {epoch}, average loss: {avg_loss:.4f}, f1_val: {f1_val:.4f}")

    if no_improve >= patience:
        print(f"Early stopping at epoch: {epoch}, best epoch is: {best_epoch}, and best f1 is: {best_f1:.4f}")
        break

model.load_state_dict(best_state)

model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(dim=1)
        all_preds.append(preds)
        all_true.append(yb)

test_preds = torch.cat(all_preds)
test_true = torch.cat(all_true)

f1_test = f1_score(
    test_true.numpy(),
    test_preds.numpy(),
    average="macro"
)

print(f"Test f1 is: {f1_test:.4f}")
