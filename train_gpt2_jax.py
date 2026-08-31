from dataclasses import dataclass
import itertools
import os
import time
import numpy as np
import tiktoken
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils, multihost_utils
from flax import nnx
import optax
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import dataset_lite
import helloswag


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # default 50257, padded to 50304 for hardware alignment
    n_layer: int = 12  # number of transformer layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding / hidden dimension
    dtype: jnp.dtype = jnp.bfloat16  # standard compute and weight storage dtype for TPU


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # key, query, value projections
        self.c_attn = nnx.Linear(
            config.n_embd,
            3 * config.n_embd,
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
        # output projection
        self.c_proj = nnx.Linear(
            config.n_embd,
            config.n_embd,
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
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
        self.c_fc = nnx.Linear(
            config.n_embd,
            4 * config.n_embd,
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj = nnx.Linear(
            4 * config.n_embd,
            config.n_embd,
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.c_fc(x)
        x = jax.nn.gelu(x, approximate=True)
        x = self.c_proj(x)
        return x


class Block(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embd, dtype=config.dtype, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, dtype=config.dtype, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.config = config

        self.wte = nnx.Embed(
            config.vocab_size,
            config.n_embd,
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
        self.wpe = nnx.Embed(
            config.block_size,
            config.n_embd,
            dtype=config.dtype,
            param_dtype=config.dtype,
            rngs=rngs,
        )
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.ln_f = nnx.LayerNorm(config.n_embd, dtype=config.dtype, rngs=rngs)

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
                    jax.random.normal(rngs.params(), module.kernel.shape, dtype=module.kernel.dtype) * std
                )
                if module.bias is not None:
                    module.bias[...] = jnp.zeros_like(module.bias[...])
            elif isinstance(module, nnx.Embed):
                module.embedding[...] = (
                    jax.random.normal(rngs.params(), module.embedding.shape, dtype=module.embedding.dtype) * 0.02
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
            # Compute cross-entropy in float32 for numerical stability
            logits_flat = logits.reshape((-1, logits.shape[-1])).astype(jnp.float32)
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


def generate(model: GPT, x: jax.Array, max_length: int, rng: jax.Array) -> jax.Array:
    """Generate tokens from the model given a starting sequence x."""
    while x.shape[1] < max_length:
        rng, subkey = jax.random.split(rng)
        logits, _ = model(x)  # (B, T, vocab_size)
        logits = logits[:, -1, :].astype(jnp.float32)  # (B, vocab_size)

        # Top-k sampling (k=50)
        topk_logits, topk_indices = jax.lax.top_k(logits, k=min(50, logits.shape[-1]))  # (B, 50)
        # Sample from categorical distribution over top-k logits
        ix = jax.random.categorical(subkey, topk_logits, axis=-1)[:, None]  # (B, 1)
        # Select chosen token indices using take_along_axis (equivalent to torch.gather)
        xcol = jnp.take_along_axis(topk_indices, ix, axis=1)  # (B, 1)
        x = jnp.concatenate([x, xcol], axis=1)
    return x


def sample(
    model: GPT,
    prompt: str,
    num_samples: int,
    max_length: int,
    rng: jax.Array,
) -> list[str]:
    """Sample generated sequences from the model."""
    enc = tiktoken.get_encoding("gpt2")
    tokens = jnp.array(enc.encode(prompt), dtype=jnp.int32)[None, :]  # (1, prompt_len)
    x = jnp.repeat(tokens, num_samples, axis=0)  # (num_samples, prompt_len)
    x = generate(model, x, max_length, rng)
    x_np = np.array(x)

    samples = []
    for i in range(num_samples):
        tokens_list = x_np[i, :max_length].tolist()
        decoded = enc.decode(tokens_list)
        samples.append(decoded)
    return samples


def helloswag_predict(tokens: jax.Array, mask: jax.Array, logits: jax.Array) -> int:
    """Helper function to predict the label for a HellaSwag example given the model's output logits."""
    shift_logits = logits[:, :-1, :].astype(jnp.float32)  # (4, T-1, vocab_size)
    shift_tokens = tokens[:, 1:]  # (4, T-1)

    # Autoregressive cross-entropy loss at all positions
    shift_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=shift_logits, labels=shift_tokens
    )  # (4, T-1)

    shift_mask = mask[:, 1:]  # (4, T-1)
    masked_shift_losses = shift_losses * shift_mask  # (4, T-1)

    sum_loss = jnp.sum(masked_shift_losses, axis=1)  # (4,)
    mask_sum = jnp.sum(shift_mask, axis=1)  # (4,)
    avg_loss = sum_loss / jnp.maximum(mask_sum, 1e-8)  # (4,)

    predicted = int(jnp.argmin(avg_loss))
    return predicted


@nnx.jit
def forward_eval(model: GPT, x: jax.Array) -> jax.Array:
    """Forward pass for evaluation with static shapes."""
    logits, _ = model(x)
    return logits


@nnx.jit
def eval_step(model: GPT, x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes cross-entropy loss over an evaluation batch without parameter updates."""
    _, loss = model(x, targets=y)
    return loss


@nnx.jit
def train_step(
    model: GPT,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> tuple[jax.Array, jax.Array]:
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

    grad_norm = optax.global_norm(scaled_grads)
    optimizer.update(scaled_grads)
    return scaled_loss, grad_norm


def numpy_loader(loader, infinite: bool = False):
    """Yields batches converted to NumPy int32 arrays, with optional infinite cycling."""
    while True:
        for x, y in loader:
            yield x.numpy().astype(np.int32), y.numpy().astype(np.int32)
        if not infinite:
            break


def train_loop(
    model: GPT,
    optimizer: nnx.Optimizer,
    lr_schedule: optax.Schedule,
    train_loader,
    val_loader,
    data_sharding: NamedSharding,
    eval_data_sharding: NamedSharding,
    grad_accum_steps: int,
    max_steps: int,
):
    """Training loop executing training steps, periodic validation loss,
    HellaSwag evaluation, text generation, and telemetry logging.
    """
    EVAL_INTERVAL = 50
    LOG_DIR = "log"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "log.txt")

    if jax.process_index() == 0:
        with open(log_file, "w") as f:  # mode 'w' to clear the file
            pass

    train_iter = numpy_loader(train_loader, infinite=True)

    for step in range(max_steps):
        t0 = time.time()

        # 1) Once in a while evaluate validation loss
        if step % EVAL_INTERVAL == 0 or step == max_steps - 1:
            val_loss_avg = 0.0
            val_loss_steps = 20
            for bx, by in itertools.islice(numpy_loader(val_loader, infinite=False), val_loss_steps):
                bx_sharded = jax.device_put(bx, eval_data_sharding)
                by_sharded = jax.device_put(by, eval_data_sharding)
                # eval_step calculates loss.mean() over data-sharded batches, 
                # so XLA automatically inserts an all-reduce across all TPU devices and hosts.
                v_loss = eval_step(model, bx_sharded, by_sharded)
                val_loss_avg += float(v_loss) / val_loss_steps

            if jax.process_index() == 0:
                print(f"validation loss: {val_loss_avg:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_avg:.4f}\n")

        # 2) Once in a while evaluate HellaSwag
        if step % EVAL_INTERVAL == 0 or step == max_steps - 1:
            num_correct = 0
            num_total = 0
            for i, example in enumerate(helloswag.iterate_examples("val")):
                if i % jax.process_count() != jax.process_index():
                    continue  # partition evaluation data across hosts

                tokens, mask, label = helloswag.render_example(example)
                tokens = jnp.array(tokens.numpy(), dtype=jnp.int32)
                mask = jnp.array(mask.numpy(), dtype=jnp.float32)

                # Pad to static block_size (1024) to ensure fixed shape (4, 1024) for JIT compilation
                if tokens.shape[1] < model.config.block_size:
                    pad_len = model.config.block_size - tokens.shape[1]
                    tokens = jnp.pad(tokens, ((0, 0), (0, pad_len)))
                    mask = jnp.pad(mask, ((0, 0), (0, pad_len)))

                # Evaluates unsharded on a single local TPU (Device 0) sequentially per host.
                # Can be optimized later by batching/sharding multiple examples across the mesh.
                tokens = jax.device_put(tokens, jax.devices()[0])
                logits = forward_eval(model, tokens)
                predicted = helloswag_predict(tokens, mask, logits)
                num_total += 1
                num_correct += int(predicted == label)

            if jax.process_count() > 1:
                total_arr = multihost_utils.process_allgather(
                    jnp.array(num_total, dtype=jnp.int32)
                )
                correct_arr = multihost_utils.process_allgather(
                    jnp.array(num_correct, dtype=jnp.int32)
                )
                num_total = int(jnp.sum(total_arr))
                num_correct = int(jnp.sum(correct_arr))

            acc_norm = num_correct / num_total if num_total > 0 else 0.0
            if jax.process_index() == 0:
                print(f"HellaSwag accuracy: {num_correct}/{num_total}={acc_norm:.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # 3) Once in a while generate text from the model
        if (step % EVAL_INTERVAL == 0 and step > 0) or step == max_steps - 1:
            sample_rng = jax.random.key(42 + jax.process_index())
            samples = sample(
                model=model,
                prompt="Hello, I'm a language model,",
                num_samples=4,
                max_length=32,
                rng=sample_rng,
            )
            for i, decoded in enumerate(samples):
                print(f"rank {jax.process_index()} sample {i}: {decoded}")

        # Gather micro-batches on host before device placement
        x_micro = []
        y_micro = []
        for _ in range(grad_accum_steps):
            bx, by = next(train_iter)
            x_micro.append(bx)
            y_micro.append(by)

        # Shape: (grad_accum_steps, B_global, T)
        x_batches = np.stack(x_micro)
        y_batches = np.stack(y_micro)

        # Place onto TPU mesh with sharding
        x_sharded = jax.device_put(x_batches, data_sharding)
        y_sharded = jax.device_put(y_batches, data_sharding)

        loss, norm = train_step(model, optimizer, x_sharded, y_sharded)
        loss_val = float(loss)
        norm_val = float(norm)

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = x_batches.size * jax.process_count()
        tokens_per_sec = tokens_processed / dt if dt > 0 else 0.0
        lr = float(lr_schedule(step))

        if jax.process_index() == 0:
            print(
                f"Step {step}, Loss: {loss_val:.6f} | lr: {lr:.4e} | norm: {norm_val:.4f} | "
                f"dt: {dt*1000:.2f} ms, tok/sec: {tokens_per_sec:.2f}"
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_val:.6f}\n")


if __name__ == "__main__":
    # 1. Setup 1D Device Mesh & SPMD Sharding
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("data",))
    data_sharding = NamedSharding(mesh, P(None, "data", None))
    eval_data_sharding = NamedSharding(mesh, P("data", None))
    replicated_sharding = NamedSharding(mesh, P())

    total_batch_size = 524288  # 2**19, 0.5M tokens
    B, T = 32, 1024
    num_devices = jax.device_count()
    global_micro_batch_size = B * num_devices
    assert (
        total_batch_size % (global_micro_batch_size * T) == 0
    ), "total_batch_size must be divisible by B * num_devices * T"
    grad_accum_steps = total_batch_size // (global_micro_batch_size * T)

    if jax.process_index() == 0:
        print(f"TPU devices detected: {num_devices} (local: {jax.local_device_count()}, hosts: {jax.process_count()})")
        print(f"Per-device batch size: {B}, Context length: {T}")
        print(f"Global batch size per micro-step: {global_micro_batch_size}")
        print(f"Total desired batch size: {total_batch_size} tokens")
        print(f"=> gradient accumulation steps: {grad_accum_steps}")

    # 2. Initialize Model and Optimizer
    config = GPTConfig(vocab_size=50304)
    rngs = nnx.Rngs(0)
    model = GPT(config, rngs=rngs)

    num_params = count_parameters(model)
    if jax.process_index() == 0:
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

    # 3. Setup PyTorch FineWeb Dataset & DataLoader
    train_dataset = dataset_lite.FineWebDataset(
        split="train", T=T
    )
    val_dataset = dataset_lite.FineWebDataset(
        split="val", T=T
    )

    if jax.process_count() > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=jax.process_count(),
            rank=jax.process_index(),
            shuffle=False,
            drop_last=True,
        )
        loader_batch_size = B * jax.local_device_count()
    else:
        sampler = None
        loader_batch_size = global_micro_batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        drop_last=True,
    )

    # 4. Execute Training Loop
    # TODO: Remove the temporary max_steps overwrite before final full-dataset training run
    max_steps = 510

    train_loop(
        model=model,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        train_loader=train_loader,
        val_loader=val_loader,
        data_sharding=data_sharding,
        eval_data_sharding=eval_data_sharding,
        grad_accum_steps=grad_accum_steps,
        max_steps=max_steps,
    )


