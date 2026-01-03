from dataclasses import dataclass
import math
import itertools
import inspect
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import tiktoken

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # 50,000 BPE merges, 256 byte tokens, 1 special token '<|endoftext|>'
    n_layer: int = 12 # number of layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768 # embedding/hidden dimension

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        T = config.block_size
        # This is actually the mask for casual self-attention, but we follow the OpenAI/HF naming.
        # 
        # The tensor is lower triangular matrix of 1s, then extended to 4D for broadcasting. 
        # tensor([[1., 0., 0.],  (T = 3)
        #         [1., 1., 0.],
        #         [1., 1., 1.]])
        self.register_buffer("bias", torch.tril(torch.ones(T, T)).view(1, 1, T, T))


    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)

        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2) # (B, T, C)

        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        nh = self.n_head
        hs = C // nh


        k = k.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)

        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(hs)) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # type: ignore
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, hs)

        # Make Training Faster #4
        # - Use Flash Attention
        # Use flash-attention built-in function in PyTorch, instead of the above 4 lines.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-arrange all head outputs of a single sequence together
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        pass
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing between token embedding and the lm head
        self.transformer["wte"].weight = self.lm_head.weight

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            pass
            # the following is PyTorch's default
            # nn.init.ones_(module.weight)
            # nn.init.zeros_(module.bias)

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model from HuggingFace."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # vacabulary size and sequence length are same for GPT-2 models
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        # keys of all tensors but ignore CausalSelfAttention's mask (register_buffer("bias", ...))
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] 

        # init a huggingface GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        # similarly, ignore mask from the huggingface model
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # The huggingface model uses Tensorflow and some of weights are transposed from PyTorch format.
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys:
            # The openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
            # thus we need to transpose their weights
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"mismatched shape for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():  # disable gradient tracking for copy operation
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"mismatched shape for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():  # disable gradient tracking for copy operation
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """
        Returns an AdamW optimizer with weight decay applied to the subset of
        model parameters (and enabled fused optimizer).
        """
        # named_paramters() returns an iterator of tuple: name (String), parameter (nn.Parameter)
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # 2D tensors enable weight decay: matmul W weights and embeddings.
        # 1D tensors disable weight decay: bias, layernorms.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # Create optimizer groups
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Check if fused AdamW is available in this PyTorch version.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {fused}")

        # Create AdamW optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}."

        # construct tokend and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer['wpe'](pos) # position embeddings of shape (T, n_embd)
        token_emb = self.transformer['wte'](idx) # token embeddings of shape (B, T, n_embd)
        x = pos_emb + token_emb

        # forward to blocks of transformer
        for block in self.transformer['h']:  # type: ignore
            x = block(x)

        # forward to the final layernorm and classifier
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            # reshape logits: (B, T, vocab_size) -> (B*T, vocab_size), targets: (B, T) -> (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

class DatasetLite(Dataset):
    def __init__(self, T, fpath="./input.txt"):
        self.T = T
        self.tokens = []
        self.len = 0

        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = enc.encode(text)
        self.len = len(self.tokens) // T
        print(f"Loaded {len(self.tokens)} tokens from {fpath}")

        # Each batch slice (B*T+1) tokens, so pad one extra token at the end.
        self.tokens.extend(enc.encode("\n"))


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        block = self.T
        start = idx * block
        end = start + block + 1
        buf = torch.tensor(self.tokens[start:end])
        x = buf[:-1]
        y = buf[1:]
        return x, y


def generate(model, x, max_length):
    """
    Generate tokens from the model given a starting sequence x.
    Args:
        model: The GPT model.
        x: Input tensor of shape (B, T) containing the starting token indices.
        max_length: The maximum length of the generated sequence.
    Returns:
        The generated sequence tensor of shape (B, max_length).
    """
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x) # (B, T, vocab_size)
        
        # take the logits at the last time step
        logits = logits[:, -1, :] # (B, vocab_size)
        # get probabilities from softmax
        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        topk_probs, topk_indices  = torch.topk(probs, 50, dim=-1) # (B, 50)
        # select a token from the top-k probabilities (aka. sample from the distribution)
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        # gather the corresponding token indices 
        xcol = torch.gather(topk_indices, 1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1) # append to the sequence 
    return x

def sample(model, num_samples, max_length):
    """Sample generated sequences from the model and print them."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1) # (1, 8)
    x = tokens.to('cuda')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    x = generate(model, x, max_length=max_length) # (B, max_length)

    for i in range(num_samples):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


def data_batch(B=4, T=32):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokens = enc.encode(text)
    buf = torch.tensor(tokens[:B*T+1])
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    return x, y

def train_loop(dataloader, model, optimizer, lr_scheduler, grad_accum_steps, device):
    for i, (x, y) in enumerate(dataloader):
        if i >= 30:
            break
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x = x.to(device)
            y = y.to(device)
            # Make Training Faster #2
            # - Enable PyTorch Automatic Mixed Precision
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                # scale down the loss to account for gradient accumulation
                # loss = ∑(l1, l2, ...) / (B * grad_accum_steps)
                loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        # Gradient clipping: ensure norm of global gradient ||g|| vector is less than c=1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        torch.cuda.synchronize()
        t1 = time.time()

        dt = t1 - t0
        tokens_processed = x.numel() * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        lr = lr_scheduler.get_last_lr()[0]
        print(f"Step {i}, Loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f} ms, tok/sec: {tokens_per_sec:.2f}")



def learning_rate_scheduler(optimizer, max_lr, min_lr, warmup_steps, max_steps):
    """Returns a learning rate scheduler with a linear warmup followed by a cosine decay.
    """
    # 1. Ensure the optimizer's base_lr matches max_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = max_lr

    # 2. Linear Warmup: starts at 1/100th of max_lr and reaches max_lr
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

    # 3. Cosine Decay: lasts for the remaining steps
    decay_steps = max_steps - warmup_steps
    decay_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=min_lr)

    # 4. Combine into SequentialLR
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps]
    )
    return scheduler


if __name__ == "__main__":
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    total_batch_size = 524288 # 2**19, 0.5M tokens
    B, T = 16, 1024
    assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by B * T"
    grad_accum_steps = total_batch_size // (B*T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> gradient accumulation steps: {grad_accum_steps}")

    max_lr = 6e-4
    min_lr = 0.1 * max_lr

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pretrained GPT-2 model
    # model = GPT.from_pretrained('gpt2')

    # Initialize a fresh model
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    # Make Training Faster #3
    # - Use torch.compile(), i.e. kernel fusion.
    model = torch.compile(model)
    dataset = DatasetLite(T=T)
    dataloder = DataLoader(dataset, batch_size=B, shuffle=False, drop_last=True)
    dataloader = itertools.cycle(dataloder)  # TODO: Use a different dataset.
    print(f"1 epoch = {len(dataset) // B} batches")

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device=device)
    lr_scheduler = learning_rate_scheduler(optimizer, max_lr=max_lr, min_lr=min_lr, warmup_steps=10, max_steps=50)

    # Make Training Faster #1
    # - Make Nvidia GPU to use TF32.
    torch.set_float32_matmul_precision('high')
    train_loop(dataloader, model, optimizer, lr_scheduler, grad_accum_steps, device)
    # sample(model, num_samples=5, max_length=30)
