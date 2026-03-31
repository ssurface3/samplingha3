import jax.numpy as jnp
from flax import linen as nn

class PISGRADNetLearnBwd(nn.Module):
    output_dim: int
    hidden_dim: int = 64

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


        drift = nn.Dense(self.output_dim)(h) 
        nn1 = nn.Dense(self.output_dim)(h)   
        nn2 = nn.Dense(self.output_dim)(h)   
        nn3 = nn.Dense(self.output_dim)(h)   

        return (drift, nn1, nn2, nn3), None