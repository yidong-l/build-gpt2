# GPT-2 (124M) PyTorch Training & Reproduction

This repository provides an end-to-end implementation for training the GPT-2 (124M) language model from scratch in PyTorch on NVIDIA GPUs using modern performance optimizations and Distributed Data Parallel (DDP).

---

## 1. Overview

The training configuration targets the standard OpenAI GPT-2 (124M) base model trained on the **FineWeb-Edu 10B** token dataset.

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Parameters** | 124M | Base GPT-2 configuration |
| **Layers ($N_{\text{layer}}$)** | 12 | Transformer decoder blocks |
| **Heads ($N_{\text{head}}$)** | 12 | Multi-head attention heads |
| **Embedding Dim ($d_{\text{model}}$)** | 768 | Hidden layer dimension |
| **Context Window ($T$)** | 1024 | Max sequence length (`block_size`) |
| **Vocabulary Size** | 50,304 | Padded from 50,257 for Tensor Core alignment |
| **Target Token Budget** | 10B tokens | 10 Billion tokens from FineWeb-Edu |
| **Total Global Batch Size** | 524,288 tokens | $2^{19} \approx 0.5\text{M}$ tokens per optimizer step |
| **Max Training Steps** | 19,073 steps | $10\text{B} / 524,288 \approx 19,073$ steps |
| **Peak Learning Rate** | $6 \times 10^{-4}$ | Linear warmup to peak, then cosine decay |

---

## 2. Repository Structure

* `train_gpt2.py`: The core script implementing the model architecture, training loop, DDP orchestration, hardware optimizations, and evaluation benchmarks.
* `dataset_lite.py`: Memory-mapped (`np.load(..., mmap_mode='r')`) dataset loader for tokenized shards compatible with PyTorch's `DataLoader`.
* `fineweb.py`: Tokenizes and shards the Hugging Face `HuggingFaceFW/fineweb-edu` (sample-10BT) dataset using `tiktoken` (GPT-2 BPE).
* `helloswag.py`: Downloads and renders the HellaSwag evaluation benchmark for zero-shot accuracy evaluation.

---

## 3. Model Training (`train_gpt2.py`)

The `train_gpt2.py` script contains the complete pretraining pipeline. Its core components and mechanics are broken down below:

### 3.1 Model Architecture
The architecture implements a Pre-LayerNorm Transformer decoder conforming to the GPT-2 specification:

```
Input Tokens (B, T) ──┬──> Token Embeddings (wte: 50304 -> 768) ──┐
                      └──> Pos. Embeddings  (wpe:  1024 -> 768) ──┴──> Sum (x)
                                                                        │
 ┌───────────────────────── Transformer Block x12 ──────────────────────┘
 │
 │  ┌──> LayerNorm (ln_1) ──> CausalSelfAttention (Flash Attention) ──> (+) ──┐
 │  │                                                                         │
 └──┴──> LayerNorm (ln_2) ──> MLP (c_fc -> GELU(tanh) -> c_proj)   ──> (+) ──┴──> Final LayerNorm (ln_f)
                                                                                  │
                                              Language Model Head (lm_head) <─────┘
                                              (Weight tied with wte)
                                              Logits / Cross-Entropy Loss
```

* **Configuration (`GPTConfig`)**: Defines context length ($T=1024$), vocabulary size ($50,304$), layer count ($12$), attention heads ($12$), and embedding dimension ($768$).
* **Weight Tying**: Shares weights between the token embedding matrix (`wte`) and the output classification head (`lm_head`):
  ```python
  self.transformer["wte"].weight = self.lm_head.weight
  ```
* **Residual Initialization Scaling**: To prevent residual stream variance from exploding with depth, projection layers (`c_proj` in attention and MLP) scale initial weights by $\frac{1}{\sqrt{2 \cdot N_{\text{layer}}}}$:
  ```python
  std = 0.02
  if hasattr(module, 'NANOGPT_SCALE_INIT'):
      std *= (2 * self.config.n_layer) ** -0.5
  nn.init.normal_(module.weight, mean=0.0, std=std)
  ```
* **Hugging Face Checkpoint Loader (`GPT.from_pretrained`)**: Imports official OpenAI weights into the model for parity testing, automatically transposing 1D Convolution weights to PyTorch `nn.Linear` layout.

---

### 3.2 Hardware & Performance Optimizations
The script integrates multiple modern PyTorch speedups for NVIDIA GPUs:

1. **TensorFloat-32 (TF32) Precision**:
   Enables 19-bit TF32 math on Ampere/Hopper Tensor Cores for FP32 matrix multiplications:
   ```python
   torch.set_float32_matmul_precision('high')
   ```
2. **Automatic Mixed Precision (AMP)**:
   Runs the forward pass in `bfloat16`, halving memory bandwidth and boosting compute throughput without loss scale instabilities:
   ```python
   with torch.autocast(device_type=device, dtype=torch.bfloat16):
       logits, loss = model(x, y)
   ```
3. **Flash Attention**:
   Uses PyTorch's native fused scaled dot-product attention (`F.scaled_dot_product_attention`) with causal masking, avoiding the materialization of $O(T^2)$ intermediate attention tensors.
4. **PyTorch 2.0 Kernel Compilation (`torch.compile`)**:
   Compiles the model into fused CUDA kernels via TorchDynamo/Inductor to eliminate kernel launch overheads.
5. **Fused AdamW Optimizer**:
   Applies parameter updates in a single fused GPU kernel pass:
   ```python
   optimizer = torch.optim.AdamW(..., fused=True)
   ```
6. **Vocabulary Size Padding**:
   Expands vocabulary size from `50,257` to `50,304` (divisible by 64 and 128) to align memory accesses on NVIDIA Tensor Cores.

---

### 3.3 Distributed Scaling & Batch Sizing
Distributed training is orchestrated via `torchrun` and NCCL backend:

* **Rank Management (`DDPConfig`)**: Retrieves `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` from environment variables.
* **Global Batch Size Calculation**:
  $$\text{Global Tokens / Step} = B \times T \times \text{world\_size} \times \text{grad\_accum\_steps} = 524,288$$
  On 8 GPUs with per-GPU batch size $B=32$ and sequence length $T=1024$:
  $$\text{Tokens per micro-step} = 32 \times 1024 \times 8 = 262,144$$
  $$\text{Gradient Accumulation Steps} = \frac{524,288}{262,144} = 2$$
* **Gradient Synchronization Control**: Skips inter-GPU gradient all-reduce overhead on intermediate micro-steps:
  ```python
  model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
  ```

---

### 3.4 Optimizer & Learning Rate Schedule
* **Selective Weight Decay (`configure_optimizers`)**: Applies weight decay ($0.1$) strictly to 2D weight matrices (matmul weights and embeddings) while leaving 1D biases and LayerNorm scales unpenalized ($0.0$).
* **Warmup + Cosine Decay (`learning_rate_scheduler`)**:
  * **Linear Warmup**: Linearly ramps learning rate from $0.1\%$ to peak ($6 \times 10^{-4}$) over the first $715$ steps ($\approx 375\text{M}$ tokens).
  * **Cosine Annealing**: Decays from peak LR down to $10\%$ ($6 \times 10^{-5}$) across the remaining steps to $10\text{B}$ tokens.
* **Gradient Clipping**: Clips global parameter gradient norms to $\|g\|_2 \le 1.0$.

---

### 3.5 Validation, Benchmarking & Evaluation
During training, the script evaluates the model at regular intervals (`EVAL_INTERVAL = 250` steps):

* **Validation Loss**: Computes cross-entropy loss over 20 batches of held-out FineWeb data, reduced across all ranks with `dist.all_reduce(..., op=dist.ReduceOp.AVG)`.
* **HellaSwag Zero-Shot Evaluation (`helloswag_predict`)**: Evaluates 10,042 validation examples partitioned across DDP ranks. Accuracies are scored using token-length normalized completion loss.
* **Text Generation (`sample`)**: Generates text completions periodically with top-$k$ ($k=50$) multinomial sampling to qualitatively inspect output coherence.
* **Checkpointing**: Periodically serializes model weights, optimizer states, scheduler states, and validation loss to `log/model_<step>.pt`.

---

## 4. Quickstart & Usage

### 1. Requirements
Install PyTorch with CUDA support and dependencies:
```bash
pip install torch tiktoken transformers datasets tqdm numpy
```

### 2. Dataset Preparation
Download and tokenize the FineWeb-Edu 10B dataset shards:
```bash
python fineweb.py --output /path/to/edu_fineweb10B
```

Ensure `DATA_DIR` in `train_gpt2.py` points to the directory containing the shards.

### 3. Running Training

**Single GPU:**
```bash
python train_gpt2.py
```

**Multi-GPU (8x GPUs on a single node):**
```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

### 4. Telemetry & Logs
Console outputs and `log/log.txt` track training progress with step execution time, throughput, and evaluations:
```text
Step 250, Loss: 5.431201 | lr: 2.0979e-04 | norm: 0.8421 | dt: 312.45 ms, tok/sec: 1677984.32
validation loss: 5.4120
HellaSwag accuracy: 2510/10042=0.2500
```
