# TinyStories Transformer Persona Model  
### Learning Risk Preferences from Synthetic Persona Data

This project studies whether a tiny language model can internalize economic risk preferences (risk-averse vs. risk-seeking) from synthetic instruction-tuning data.

To examine this, the TinyStories base model is fine-tuned into two personas:

- **Cautious Saver** — risk-averse  
- **Bold Gambler** — risk-seeking  

Each persona is trained using **1,000 synthetic examples**, where each example consists of a short story and a decision between a safe option and a risky lottery.

Two models are required to generate a structured response in the format:

Choice: A/B (One option is safe option, the other option is the risky option)
Reason: short explanation

From the same prompt, the two personas show clear behavioral differences:

- The **Cautious Saver** tends to choose safe option.
- The **Bold Gambler** tends to choose risky option.

To evaluate these behaviors, the project uses a simple version of **Multiple Price List (MPL)** experiment, where models make a series of lottery decisions with increasing probabilities of high payoffs.

---

# Motivation

Risk preference plays an important role in decision making under uncertainty. In economics, researchers often study whether individuals prefer safe outcomes or risky outcomes.

Understanding this is important because it helps answer two broader questions:

- Whether language models can be simulated as basic economic agents
- What limitations extremely small models have when dealing with complex probability-based decision rules

---

## Project Structure

- `src/`: core models and training code
- `scripts/`: scripts for data generation, evaluation, chatting with the model, and plotting loss curves
- `data/`: synthetic persona datasets for the Cautious Saver and Bold Gambler
- `model/`: saved model checkpoints and fine-tuned persona models
- `results/`: output figures and evaluation results
- `tokenizer/`: saved BPE tokenizer file
- `README.md`: project overview and reproduction instructions
- `requirements.txt`: dependency file for environment setup

---

# Reproducing the Project

The training pipeline follows the **TinyStories LLM assignment instructions (steps 1–5)** provided by the course repository. These steps include training BPE tokenizer and the base TinyStories model. The following are steps to reproduce the persona training and evaluation in this project.

---

## Step 6 — Generate Persona Datasets

Generate the synthetic datasets for the two personas.

```bash
poetry run python scripts/generate_cautious_saver_data.py
poetry run python scripts/generate_bold_gambler_data.py
```

---

## Step 7 — Train Persona Models

Fine-tune the base model with the persona datasets.

**Train Cautious Saver**
```bash
poetry run python src/train_tinystories_chat_model.py --pretrained_model_path model/best_model.pth --tokenizer_path tokenizer/bpe_tokenizer_tinystories.pkl --local_json_path data/cautious_saver.json --output_dir model/cautious_chat_model --epochs 15 --batch_size 32 --lr 0.0001 --max_seq_len 256 --amp
```
**Train Bold Gambler**
```bash
poetry run python src/train_tinystories_chat_model.py --pretrained_model_path model/best_model.pth --tokenizer_path tokenizer/bpe_tokenizer_tinystories.pkl --local_json_path data/bold_gambler.json --output_dir model/bold_gambler_chat_model --epochs 15 --batch_size 32 --lr 0.0001 --max_seq_len 256 --amp
```

---

## Step 8 — Evaluate with MPL

Run the Multiple Price List evaluation.
```bash
poetry run python scripts/evaluate_mpl.py
```
MPL result table is in the mpl_results_allblocks document.
---

## Step 9 — Generate Example Responses
Example prompt: **Tom saw a merchant offering a risky coin gamble. What should Tom do?**


### Base Model

```bash
poetry run python scripts/generate_tinystories_text.py --model_path model/best_model.pth --tokenizer_path tokenizer/bpe_tokenizer_tinystories.pkl --prompt "Tom saw a merchant offering a risky coin gamble. What should Tom do?" --max_length 150
```
### Cautious Saver

```bash
poetry run python scripts/generate_tinystories_text.py --model_path model/cautious_chat_model/final_model.pth --tokenizer_path tokenizer/bpe_tokenizer_tinystories.pkl --prompt "Tom saw a merchant offering a risky coin gamble. What should Tom do?" --max_length 150
```

### Bold Gambler
```bash
poetry run python scripts/generate_tinystories_text.py --model_path model/bold_gambler_chat_model/final_model.pth --tokenizer_path tokenizer/bpe_tokenizer_tinystories.pkl --prompt "Tom saw a merchant offering a risky coin gamble. What should Tom do?" --max_length 150
```
All results are in prompt result document in the results file.

---

## Step 10 — Generate Loss Curves

Training loss curves for the two personas can be generated using the following script:

```bash
poetry run python scripts/plot_loss_curve.py
```

---

# Key Takeaways

This project shows that:

- Instruction tuning can let tinystories model learn economic preferences
- Evaluation using MPL experiments provides a simple behavioral test, and it show a clear pattern
- TinyStories model can not perform probability-based decision making due to its limited number of parameters.

The results suggest that preference patterns can emerge from fine-tuning data, but the core idea of switch point can not be learned by the tinystories model.

---
## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Python version used in this project:
Python 3.12