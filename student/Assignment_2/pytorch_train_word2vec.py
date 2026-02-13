import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import os

# Hyperparameters (use exactly these)
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, centers, contexts):
        self.centers = torch.tensor(centers, dtype=torch.long)
        self.contexts = torch.tensor(contexts, dtype=torch.long)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# Simple Skip-gram Module (Negative Sampling version)
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.in_embed.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_ids, context_ids):
        v = self.in_embed(center_ids)
        u = self.out_embed(context_ids) 

        if u.dim() == 2:
            return (v * u).sum(dim=1)
        else:
            return (v.unsqueeze(1) * u).sum(dim=2) 

    def get_embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()


# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

skipgram_df = data["skipgram_df"]
word2idx = data["word2idx"]
idx2word = data["idx2word"]

counter = data.get("counter", {})
if not counter:
    raise KeyError("No 'counter' found in processed_data.pkl. Please include it in Part 1.")

centers = skipgram_df["center"].values
contexts = skipgram_df["context"].values


# Precompute negative sampling distribution below
def build_negative_sampling_dist(counter, word2idx, power=0.75):
    V = len(word2idx)
    counts = torch.zeros(V, dtype=torch.float)

    for w, idx in word2idx.items():
        counts[idx] = float(counter.get(w, 0.0))
    
    counts = torch.clamp(counts, min=1.0)

    probs = counts.pow(power)
    probs = probs / probs.sum()
    return probs


def sample_negatives(probs, batch_size, K, device, positive_context=None):
    neg = torch.multinomial(probs, num_samples=batch_size * K, replacement=True)
    neg = neg.view(batch_size, K).to(device)

    if positive_context is None:
        return neg

    pos = positive_context.view(-1, 1)
    mask = (neg == pos)

    while mask.any():
        num_bad = int(mask.sum().item())
        resample = torch.multinomial(probs, num_samples=num_bad, replacement=True).to(device)
        neg[mask] = resample
        mask = (neg == pos)

    return neg


# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)


# Dataset and DataLoader
dataset = SkipGramDataset(centers, contexts)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# Model, Loss, Optimizer
vocab_size = len(word2idx)
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)

neg_probs = build_negative_sampling_dist(counter, word2idx, power=0.75).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)# use Adam as optimizer


def make_targets(center, context, vocab_size): 
    raise NotImplementedError("Not needed for negative sampling (we use BCE targets directly).")


# Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0

    for center_ids, pos_context_ids in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        center_ids = center_ids.to(device)          
        pos_context_ids = pos_context_ids.to(device) 
        B = center_ids.size(0)

        pos_logits = model(center_ids, pos_context_ids)
        pos_labels = torch.ones_like(pos_logits)
        pos_loss = criterion(pos_logits, pos_labels)

        neg_context_ids = sample_negatives(
            neg_probs, batch_size=B, K=NEGATIVE_SAMPLES,
            device=device, positive_context=pos_context_ids
        )

        neg_logits = model(center_ids, neg_context_ids)
        neg_labels = torch.zeros_like(neg_logits)
        neg_loss = criterion(neg_logits, neg_labels)

        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - avg loss: {avg_loss:.4f}")


# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
