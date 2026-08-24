from dataclasses import dataclass
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
                module.kernel.value = (
                    jax.random.normal(rngs.params(), module.kernel.shape) * std
                )
                if module.bias is not None:
                    module.bias.value = jnp.zeros_like(module.bias.value)
            elif isinstance(module, nnx.Embed):
                module.embedding.value = (
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
        logits = jnp.matmul(x, self.wte.embedding.value.T)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape((-1, logits.shape[-1]))
            targets_flat = targets.reshape((-1,))
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits_flat, labels=targets_flat
            ).mean()

        return logits, loss


def count_parameters(model: nnx.Module) -> int:
    """Returns the total number of trainable parameter elements in the model."""
    _, state = nnx.split(model)
    return sum(x.size for x in jax.tree.leaves(state))


def verify_model():
    """Verifies GPT-2 model architecture, parameter count, and JIT forward pass on accelerator."""
    devices = jax.devices()
    backend = jax.default_backend()
    print(f"JAX backend: {backend}, available devices: {devices}")

    # Instantiate model on accelerator
    config = GPTConfig(vocab_size=50304)
    rngs = nnx.Rngs(0)
    model = GPT(config, rngs=rngs)

    num_params = count_parameters(model)
    print(f"Instantiated GPT-2 model with {num_params:,} parameters.")
    assert num_params == 124_475_904, f"Unexpected parameter count: {num_params}"

    # Allocate inputs (created directly on default accelerator device)
    dummy_input = jnp.ones((2, 64), dtype=jnp.int32)
    dummy_targets = jnp.ones((2, 64), dtype=jnp.int32)

    # Compile forward step with JIT targeting accelerator
    @nnx.jit
    def forward_step(model: GPT, x: jax.Array, y: jax.Array):
        return model(x, targets=y)

    logits, loss = forward_step(model, dummy_input, dummy_targets)

    # Inspect device placement
    model_param_devices = model.wte.embedding.value.devices()
    print(f"Model parameters device: {model_param_devices}")
    print(f"Input device: {dummy_input.devices()}")
    print(f"Logits shape: {logits.shape} (expected: (2, 64, 50304)), device: {logits.devices()}")
    print(f"Loss: {loss.item():.4f} (expected initial loss ~ -log(1/50304) = 10.8258), device: {loss.devices()}")

    assert logits.shape == (2, 64, 50304), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None, "Loss should not be None when targets are provided"
    print("Model verification PASSED.")


if __name__ == "__main__":
    verify_model()
