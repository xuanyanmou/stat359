import numpy as np
import pandas as pd
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt', quiet=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

print("\n========== Loading Dataset ==========")
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Loading FastText Model ==========")
ft_path = r"D:\stat359\instructor\Assignment_2\fasttext-wiki-news-subwords-300.model"
ft_model = KeyedVectors.load(ft_path)
print("FastText model loaded successfully.")

def get_mean_embedding(text, ft_model):
    tokens = word_tokenize(text.lower())
    vectors = [ft_model[w] for w in tokens if w in ft_model]
    if not vectors:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

sentences = dataset['train']['sentence']
labels = np.array(dataset['train']['label'])
X = np.array([get_mean_embedding(s, ft_model) for s in tqdm(sentences, desc="Generating embeddings")])
y = labels
print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

class MLPDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_loader = DataLoader(MLPDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(MLPDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(MLPDataset(X_test, y_test), batch_size=64, shuffle=False)
print("DataLoaders created.")

class SentimentMLP(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.network(x)

device = get_device()
model = SentimentMLP(num_classes=3).to(device)

os.makedirs("outputs", exist_ok=True)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

counts = np.bincount(y_train)
class_weights = (1. / torch.tensor(counts, dtype=torch.float)).to(device)
class_weights /= class_weights.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights)

set_seed(42)
num_epochs = 35  
best_val_f1 = 0.0
history = {'t_loss': [], 'v_loss': [], 't_f1': [], 'v_f1': [], 't_acc': [], 'v_acc': []}

for epoch in range(num_epochs):
    model.train()
    running_loss, t_preds, t_labels = 0.0, [], []
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        t_preds.extend(preds.cpu().numpy())
        t_labels.extend(labels.cpu().numpy())

    model.eval()
    val_loss, v_preds, v_labels = 0.0, [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            v_preds.extend(preds.cpu().numpy())
            v_labels.extend(labels.cpu().numpy())
            
    history['t_loss'].append(running_loss / len(X_train))
    history['v_loss'].append(val_loss / len(X_val))
    history['t_f1'].append(f1_score(t_labels, t_preds, average='macro'))
    history['v_f1'].append(f1_score(v_labels, v_preds, average='macro'))
    history['t_acc'].append((np.array(t_preds) == np.array(t_labels)).mean())
    history['v_acc'].append((np.array(v_preds) == np.array(v_labels)).mean())

    print(f"Epoch {epoch+1}: Train F1: {history['t_f1'][-1]:.4f} | Val F1: {history['v_f1'][-1]:.4f}")
    
    scheduler.step(history['v_f1'][-1])
    if history['v_f1'][-1] > best_val_f1:
        best_val_f1 = history['v_f1'][-1]
        torch.save(model.state_dict(), 'outputs/best_mlp_model.pth')
        print(f'>>> Saved new best model')

fig, axes = plt.subplots(3, 1, figsize=(10, 15))
titles = ['Loss', 'Macro F1 Score', 'Accuracy']
keys = [('t_loss', 'v_loss'), ('t_f1', 'v_f1'), ('t_acc', 'v_acc')]

for i, (ax, title, (tk, vk)) in enumerate(zip(axes, titles, keys)):
    ax.plot(history[tk], label='Train')
    ax.plot(history[vk], label='Val')
    ax.set_title(f'MLP: {title} vs Epochs')
    ax.set_xlabel('Epochs')
    ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig('outputs/mlp_f1_learning_curves.png')

model.load_state_dict(torch.load('outputs/best_mlp_model.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"\nFinal Test Macro F1: {f1_score(all_labels, all_preds, average='macro'):.4f}")
print(classification_report(all_labels, all_preds, target_names=['Neg', 'Neu', 'Pos'], digits=4))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.title('MLP Confusion Matrix')
plt.savefig('outputs/mlp_confusion_matrix.png')
print("Plots saved in 'outputs/' folder.")