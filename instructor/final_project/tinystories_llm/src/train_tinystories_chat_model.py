import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
from tqdm import tqdm
import time
import json
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ---------- HASH CHECK ----------
import hashlib

def conv_hash(ex):
    # ex: {"conversation": [{"text": ...}, ...], ...}
    s = "||".join([t["text"].strip() for t in ex["conversation"]])
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def run_hash_check(dataset):
    """
    dataset: HuggingFace DatasetDict with splits "train" and "valid"
    """
    train_hashes = set(conv_hash(dataset["train"][i]) for i in range(len(dataset["train"])))
    val_hashes = set(conv_hash(dataset["valid"][i]) for i in range(len(dataset["valid"])))

    print("train size:", len(dataset["train"]))
    print("val size:", len(dataset["valid"]))
    print("hash overlap:", len(train_hashes & val_hashes))
    print("train unique rate:", len(train_hashes) / max(1, len(dataset["train"])))


# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train a TinyStories chat model with instruction tuning")

    # Dataset arguments
    # ✅ default changed to your HF repo
    parser.add_argument("--dataset", type=str, default="Xuanyan1/tinystories-persona-dataset",
                        help="HuggingFace dataset name (repo), e.g. Xuanyan1/tinystories-persona-dataset")
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl",
                        help="Path to BPE tokenizer")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length (cap length)")
    parser.add_argument("--local_json_path", type=str, default=None,
                        help="Path to a local JSON file (list of {'conversation': [...]})")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio for local JSON")

    # Model architecture arguments
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=256)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Instruction tuning arguments
    parser.add_argument("--pretrained_model_path", required=True, type=str,
                        help="Path to pretrained model checkpoint (.pth)")
    parser.add_argument("--user_token", type=str, default="<user>")
    parser.add_argument("--assistant_token", type=str, default="<assistant>")

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="tinystories_chat_model")
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=3000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--pilot_run", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


# =========================
# Device / Seed
# =========================
def get_device(device_preference):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    if device_preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        return torch.device("cuda")

    if device_preference == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise ValueError("MPS requested but not available.")
        return torch.device("mps")

    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Tokenizer
# =========================
def load_tokenizer(tokenizer_path):
    tok = BPETokenizer.load(tokenizer_path)
    # Ensure pad exists; if not, fallback to 0 safely
    if "<pad>" not in tok.token2id:
        print("[WARN] <pad> not found in tokenizer. Using id=0 as pad_token_id.")
    return tok


# =========================
# Dataset
# =========================
class TinyStoriesConversationDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=256,
        split="train",
        max_samples=None,
        user_token="<user>",
        assistant_token="<assistant>",
    ):
        self.ds = dataset[split]
        if max_samples is not None:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.user_token = user_token
        self.assistant_token = assistant_token

        if user_token not in tokenizer.token2id:
            raise ValueError(f"User token {user_token} not found in tokenizer vocabulary")
        if assistant_token not in tokenizer.token2id:
            raise ValueError(f"Assistant token {assistant_token} not found in tokenizer vocabulary")

        self.pad_id = tokenizer.token2id.get("<pad>", 0)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            idx = idx[0]

        row = self.ds[idx]
        conversation = row["conversation"]

        formatted = []
        for i, turn in enumerate(conversation):
            if i % 2 == 0:
                formatted.append(f"{self.user_token} {turn['text']}")
            else:
                formatted.append(f"{self.assistant_token} {turn['text']}")
        formatted_text = " ".join(formatted)

        ids = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        ids = ids[: self.max_length]
        attention_mask = [1] * len(ids)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def make_collate_fn(pad_id: int, max_seq_len: int):
    def collate(batch):
        lengths = [x["input_ids"].size(0) for x in batch]
        max_len = min(max(lengths), max_seq_len)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, x in enumerate(batch):
            ids = x["input_ids"][:max_len]
            am = x["attention_mask"][:max_len]
            input_ids[i, : ids.size(0)] = ids
            attention_mask[i, : am.size(0)] = am

        return input_ids, attention_mask

    return collate


# =========================
# Scheduler
# =========================
class WarmupLinearScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(1, total_steps)
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            lr_scale = max(0.0, 1.0 - progress)

        for pg in self.optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lr_scale


# =========================
# Eval
# =========================
def evaluate(model, dataloader, criterion, device, use_amp: bool):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            attn = attention_mask[:, :-1]

            if use_amp and device.type == "cuda":
                from torch.amp import autocast
                with autocast(device_type="cuda"):
                    out = model(input_ids=inputs, attention_mask=attn)
                    logits = out["logits"]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                out = model(input_ids=inputs, attention_mask=attn)
                logits = out["logits"]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


# =========================
# Train
# =========================
def train_and_evaluate(args):
    device = get_device(args.device)
    print(f"Using device: {device}")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=args.output_dir)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = load_tokenizer(args.tokenizer_path)
    pad_id = tokenizer.token2id.get("<pad>", 0)

    # ---- Load dataset
    if args.local_json_path is not None:
        # ===== Local JSON path mode (unchanged) =====
        print(f"Loading local JSON dataset: {args.local_json_path}")
        with open(args.local_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        ds = HFDataset.from_list(data)
        split = ds.train_test_split(
            test_size=args.val_ratio,
            seed=args.seed,
            shuffle=True
        )

        dataset = DatasetDict({
            "train": split["train"],
            "valid": split["test"]
        })
    else:
        # ===== Hugging Face repo mode (CHANGED) =====
        # Your repo has only files:
        #   cautious_saver.json
        #   bold_gambler.json
        # We directly map them to splits train/valid to keep your code identical.
        print(f"Loading dataset from HF repo: {args.dataset}")
        dataset = load_dataset(
            args.dataset,
            data_files={
                "train": "cautious_saver.json",
                "valid": "bold_gambler.json",
            }
        )

    # ---- HASH CHECK (after dataset is created)
    run_hash_check(dataset)

    train_dataset = TinyStoriesConversationDataset(
        dataset,
        tokenizer,
        max_length=args.max_seq_len,
        split="train",
        max_samples=args.max_train_samples,
        user_token=args.user_token,
        assistant_token=args.assistant_token,
    )
    val_dataset = TinyStoriesConversationDataset(
        dataset,
        tokenizer,
        max_length=args.max_seq_len,
        split="valid",
        max_samples=args.max_eval_samples,
        user_token=args.user_token,
        assistant_token=args.assistant_token,
    )

    collate_fn = make_collate_fn(pad_id=pad_id, max_seq_len=args.max_seq_len)
    pin_memory = (device.type == "cuda")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    config = TinyStoriesConfig(
        vocab_size=len(tokenizer.token2id),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_seq_len,
        window_size=args.window_size,
    )

    model = TinyStoriesForCausalLM(config).to(device)

    # ---- Load pretrained
    print(f"Loading pretrained model from {args.pretrained_model_path}")
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: layers={args.num_layers}, hidden={args.hidden_size}, heads={args.num_heads}")
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = args.lr

    total_steps = (len(train_dataloader) * args.epochs) // max(1, args.gradient_accumulation_steps)
    scheduler = WarmupLinearScheduler(optimizer, args.warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and "current_step" in ckpt["scheduler_state_dict"]:
            scheduler.current_step = ckpt["scheduler_state_dict"]["current_step"]
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")

    use_amp = args.amp and (device.type == "cuda")
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler(device="cuda")
    else:
        autocast = None
        scaler = None

    # ---- Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")
        optimizer.zero_grad()

        for step, (input_ids, attention_mask) in enumerate(pbar):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            attn = attention_mask[:, :-1]

            if use_amp:
                with autocast(device_type="cuda"):
                    out = model(input_ids=inputs, attention_mask=attn)
                    logits = out["logits"]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                out = model(input_ids=inputs, attention_mask=attn)
                logits = out["logits"]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # logging
                if global_step % args.logging_steps == 0:
                    cur_loss = loss.item() * args.gradient_accumulation_steps
                    train_losses.append(cur_loss)
                    avg_loss = sum(train_losses[-100:]) / min(len(train_losses), 100)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                    writer.add_scalar("Loss/train", avg_loss, global_step)
                    writer.add_scalar("Perplexity/train", float(np.exp(avg_loss)), global_step)

                # eval
                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_dataloader, criterion, device, use_amp=use_amp)
                    val_ppl = float(np.exp(val_loss))
                    print(f"\nStep {global_step}: Val loss={val_loss:.4f}, Val ppl={val_ppl:.2f}")

                    writer.add_scalar("Loss/val", val_loss, global_step)
                    writer.add_scalar("Perplexity/val", val_ppl, global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = os.path.join(args.output_dir, "best_model.pth")
                        torch.save(model.state_dict(), best_path)
                        print(f"New best model saved to {best_path}")
                    model.train()

                # checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": {"current_step": scheduler.current_step},
                            "loss": float(loss.item()),
                        },
                        ckpt_path,
                    )
                    print(f"\nCheckpoint saved to {ckpt_path}")

            epoch_loss += loss.item() * args.gradient_accumulation_steps

        # end epoch
        avg_epoch_loss = epoch_loss / max(1, len(train_dataloader))
        train_ppl = float(np.exp(avg_epoch_loss))
        print(f"Epoch {epoch+1}: Train loss={avg_epoch_loss:.4f}, Train ppl={train_ppl:.2f}")

        writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch + 1)
        writer.add_scalar("Perplexity/train_epoch", train_ppl, epoch + 1)

        val_loss = evaluate(model, val_dataloader, criterion, device, use_amp=use_amp)
        val_ppl = float(np.exp(val_loss))
        print(f"Epoch {epoch+1}: Val loss={val_loss:.4f}, Val ppl={val_ppl:.2f}")

        writer.add_scalar("Loss/val_epoch", val_loss, epoch + 1)
        writer.add_scalar("Perplexity/val_epoch", val_ppl, epoch + 1)

        epoch_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_path)
        print(f"Model saved to {epoch_path}")

    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    writer.close()
    return model, device


# =========================
# Generation helper
# =========================
def generate_chat_response(model, tokenizer, prompt, device, max_length=100, temperature=0.7, top_p=0.9):
    model.eval()
    formatted_prompt = f"<user> {prompt} <assistant>"
    input_ids = torch.tensor([tokenizer.encode(formatted_prompt, add_special_tokens=False)], dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )

    output_text = tokenizer.decode(output_ids[0].tolist())

    if "<assistant>" in output_text:
        assistant_response = output_text.split("<assistant>")[-1].strip()
        if "<user>" in assistant_response:
            assistant_response = assistant_response.split("<user>")[0].strip()
        return assistant_response

    return output_text


if __name__ == "__main__":
    args = parse_args()

    if args.pilot_run:
        args.max_train_samples = 1000
        args.max_eval_samples = 100
        print("[Pilot Run] Using 1000 samples for training and 100 for evaluation.")

    start_time = time.time()
    model, device = train_and_evaluate(args)
    end_time = time.time()

    print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")