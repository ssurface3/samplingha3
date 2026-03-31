import jax.numpy as jnp
from flax import linen as nn


class PISGRADNetLearnBwd(nn.Module):
    output_dim: int
    hidden_dim: int = 64
    weight_init: float = 1e-8
    bias_init: float = 0.0

    @nn.compact
    def __call__(self, x, t, langevin=None):
        t_expand = jnp.broadcast_to(t, x.shape[:-1] + (1,))

        if langevin is not None:
            h = jnp.concatenate([x, t_expand, langevin], axis=-1)
        else:
            h = jnp.concatenate([x, t_expand], axis=-1)

        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)

        # Small init for drift so the process starts as a Brownian bridge
        drift = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.constant(self.weight_init),
            bias_init=nn.initializers.zeros_init(),
        )(h)
        # Zero init for nn1/nn2/nn3 so gamma=1, alpha=1, beta=1 at init
        nn1 = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(h)
        nn2 = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(h)
        nn3 = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(h)

        return (drift, nn1, nn2, nn3), None