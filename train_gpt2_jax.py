from dataclasses import dataclass
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # default 50257, padded to 50304 for hardware alignment
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding / hidden dimension


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # key, query, value projections
        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, rngs=rngs)
        # output projection
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, rngs=rngs)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, C = x.shape
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # (B, T, C)

        nh = self.n_head
        hs = C // nh

        k = k.reshape((B, T, nh, hs))  # (B, T, nh, hs)
        q = q.reshape((B, T, nh, hs))  # (B, T, nh, hs)
        v = v.reshape((B, T, nh, hs))  # (B, T, nh, hs)

        y = jax.nn.dot_product_attention(q, k, v, is_causal=True)

        y = y.reshape((B, T, C))  # (B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, rngs=rngs)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.c_fc(x)
        x = jax.nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x


class Block(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.config = config

        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.wpe = nnx.Embed(config.block_size, config.n_embd, rngs=rngs)
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)

        # initialize weights
        self._init_weights(rngs)

    def _init_weights(self, rngs: nnx.Rngs):
        """Recursively initialize weights across all submodules."""
        for _, module in nnx.iter_graph(self):
            if isinstance(module, nnx.Linear):
                std = 0.02
                if hasattr(module, "NANOGPT_SCALE_INIT"):
                    std *= (2 * self.config.n_layer) ** -0.5
                module.kernel[...] = (
                    jax.random.normal(rngs.params(), module.kernel.shape) * std
                )
                if module.bias is not None:
                    module.bias[...] = jnp.zeros_like(module.bias[...])
            elif isinstance(module, nnx.Embed):
                module.embedding[...] = (
                    jax.random.normal(rngs.params(), module.embedding.shape) * 0.02
                )
            elif isinstance(module, nnx.LayerNorm):
                pass

    def __call__(
        self, idx: jax.Array, targets: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        B, T = idx.shape
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is {self.config.block_size}."

        pos = jnp.arange(0, T, dtype=jnp.int32)[None, :]  # shape (1, T)
        pos_emb = self.wpe(pos)  # (1, T, n_embd)
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        # Weight-tied LM head: project using transposed embedding weights
        logits = jnp.matmul(x, self.wte.embedding[...].T)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape((-1, logits.shape[-1]))
            targets_flat = targets.reshape((-1,))
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits_flat, labels=targets_flat
            ).mean()

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float | optax.Schedule,
        betas: tuple[float, float],
        eps: float,
        clip_grad_norm: float,
    ) -> nnx.Optimizer:
        """Configures AdamW optimizer with weight decay applied to 2D+ tensors (weights and embeddings)
        and 0.0 to 1D tensors (biases, layernorms), combined with global gradient norm clipping.
        """
        # 2D+ tensors enable weight decay: matmul kernel weights and embeddings.
        # 1D tensors disable weight decay: bias, layernorms.
        mask = lambda params: jax.tree.map(lambda p: p.ndim >= 2, params)

        tx = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.adamw(
                learning_rate=learning_rate,
                b1=betas[0],
                b2=betas[1],
                eps=eps,
                weight_decay=weight_decay,
                mask=mask,
            ),
        )
        return nnx.Optimizer(self, tx, wrt=nnx.Param)


def learning_rate_scheduler(
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> optax.Schedule:
    """Returns a learning rate schedule with a linear warmup followed by a cosine decay."""
    decay_steps = max(0, max_steps - warmup_steps)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.001 * max_lr,
        peak_value=max_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=min_lr,
    )


def count_parameters(model: nnx.Module) -> int:
    """Returns the total number of trainable parameter elements in the model."""
    _, state = nnx.split(model)
    return sum(x.size for x in jax.tree.leaves(state))


@nnx.jit
def train_step(
    model: GPT,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Executes forward, backward, in-graph gradient accumulation across micro-batches,
    and optimizer update inside @nnx.jit.
    x, y shape: (grad_accum_steps, B, T).
    """
    graphdef, params = nnx.split(model, nnx.Param)

    def loss_fn(p: nnx.State, x_micro: jax.Array, y_micro: jax.Array):
        m = nnx.merge(graphdef, p)
        _, loss = m(x_micro, targets=y_micro)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)

    def scan_fn(carry, micro_batch):
        accum_grads, accum_loss = carry
        x_micro, y_micro = micro_batch
        loss, grads = grad_fn(params, x_micro, y_micro)
        accum_loss = accum_loss + loss
        accum_grads = jax.tree.map(lambda ag, g: ag + g, accum_grads, grads)
        return (accum_grads, accum_loss), None

    initial_accum_grads = jax.tree.map(jnp.zeros_like, params)

    (total_grads, total_loss), _ = jax.lax.scan(
        scan_fn,
        (initial_accum_grads, 0.0),
        (x, y),
    )

    num_micro_batches = x.shape[0]
    scaled_grads = jax.tree.map(lambda g: g / num_micro_batches, total_grads)
    scaled_loss = total_loss / num_micro_batches

    optimizer.update(scaled_grads)
    return scaled_loss


def infinite_dataloader(loader):
    """A helper function to create an infinite dataloader iterator from a finite iterable."""
    while True:
        for batch in loader:
            yield batch


def train_loop(
    model: GPT,
    optimizer: nnx.Optimizer,
    train_loader,
    grad_accum_steps: int,
    max_steps: int,
):
    """Training loop executing training steps, in-graph gradient accumulation,
    and throughput logging.
    """
    train_iter = infinite_dataloader(train_loader)

    for step in range(max_steps):
        t0 = time.time()

        # Gather micro-batches on host before single device transfer
        x_micro = []
        y_micro = []
        for _ in range(grad_accum_steps):
            bx, by = next(train_iter)
            x_micro.append(bx)
            y_micro.append(by)

        x_batches = jnp.array(np.stack(x_micro))
        y_batches = jnp.array(np.stack(y_micro))

        loss = train_step(model, optimizer, x_batches, y_batches)
        loss_val = float(loss)

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = x_batches.size
        tokens_per_sec = tokens_processed / dt if dt > 0 else 0.0

        print(
            f"Step {step}, Loss: {loss_val:.6f} | "
            f"dt: {dt*1000:.2f} ms, tok/sec: {tokens_per_sec:.2f}"
        )


def verify_training(
    model: GPT,
    optimizer: nnx.Optimizer,
    B: int,
    T: int,
):
    """Sets up synthetic dataloader and executes a short verification run using train_loop."""
    vocab_size = model.config.vocab_size
    synthetic_batches = 20
    grad_accum_steps = 2
    max_steps = 5

    # Synthetic training batches (B, T)
    synthetic_train_data = [
        (
            jnp.ones((B, T), dtype=jnp.int32) * (i % vocab_size),
            jnp.ones((B, T), dtype=jnp.int32) * ((i + 1) % vocab_size),
        )
        for i in range(synthetic_batches)
    ]

    train_loop(
        model=model,
        optimizer=optimizer,
        train_loader=synthetic_train_data,
        grad_accum_steps=grad_accum_steps,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    total_batch_size = 524288  # 2**19, 0.5M tokens
    B, T = 32, 1024
    assert total_batch_size % (B * T) == 0, "total_batch_size must be divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> gradient accumulation steps: {grad_accum_steps}")

    # Initialize a fresh model
    config = GPTConfig(vocab_size=50304)
    rngs = nnx.Rngs(0)
    model = GPT(config, rngs=rngs)

    num_params = count_parameters(model)
    print(f"Instantiated GPT-2 model with {num_params:,} parameters.")

    max_lr = 6e-4
    min_lr = 0.1 * max_lr
    warmup_steps = 715  # 375M tokens / 0.5M batch_size
    max_steps = 19073  # 10B tokens / 0.5M batch_size
    lr_schedule = learning_rate_scheduler(
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
    )
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=lr_schedule,
        betas=(0.9, 0.95),
        eps=1e-8,
        clip_grad_norm=1.0,
    )

    verify_training(
        model=model,
        optimizer=optimizer,
        B=B,
        T=T,
    )


