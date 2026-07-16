from jax import random, numpy as jnp, jit
import jax
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable
from ngclearn.utils import tensorstats
import os
from pathlib import Path

@partial(jit, static_argnums=[0,1])
def _create_sinusoidal_embeddings(seq_len, embed_dim):
    """
    Create fixed absolute sinusoidal positional embeddings.

    Returns:
        Shape: (seq_len, embed_dim)
    """
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, embed_dim, 2) * 
                      (-jnp.log(10000.0) / embed_dim))
    angles = position * div_term
    embeddings = jnp.zeros((seq_len, embed_dim))
    embeddings = embeddings.at[:, 0::2].set(jnp.sin(angles))
    embeddings = embeddings.at[:, 1::2].set(jnp.cos(angles))
    return embeddings

@partial(jit, static_argnums=[2, 3, 4, 5])
def _compute_embedding_updates(inputs, post, vocab_size, seq_len, embed_dim, batch_size):
    """
    Compute updates for word embeddings and postional embeddings
    """
    
    # Flatten for processing
    flat_tokens = inputs.reshape(-1)
    flat_errors = post.reshape(batch_size * seq_len, embed_dim)
     
    # Word embeddings update - accumulate gradients for each token
    d_word_weights = jnp.zeros((vocab_size, embed_dim))
    
    d_word_weights = d_word_weights.at[flat_tokens].add(flat_errors)


    # postional embededings update

    d_pos_weights = jnp.zeros((seq_len, embed_dim))
    
    batch_positions = jnp.tile(jnp.arange(seq_len), batch_size).astype(jnp.int32)
    d_pos_weights = jax.lax.cond(
      lambda: d_pos_weights.at[batch_positions].add(flat_errors), lambda: d_pos_weights
    )
            
    return d_word_weights, d_pos_weights

class EmbeddingSynapse(JaxComponent):
    """
    A synaptic cable that handles word embeddings.

    | --- Synapse Compartments: ---
    | inputs - input token indices (takes in external signals)
    | outputs - output embedding signals (only word embeddings)
    | word_weights - word embedding matrix
    | post - post-synaptic signals for learning (takes in external signals)
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | dWordWeights - current delta matrix for word embedding changes
    | word_opt_params - optimizer statistics for word embeddings

    Args:
        name: the string name of this component

        vocab_size: size of vocabulary for word embeddings

        seq_len: sequence length

        embed_dim: dimensionality of embeddings

        batch_size: batch size dimension

        eta: global learning rate 

        optim_type: optimization scheme (Default: "sgd")

        weight_scale: scaling factor for weight initialization (Default: 0.02)
    """

    def __init__(
            self, name, vocab_size, seq_len, embed_dim, batch_size,
            eta, optim_type,position_encoding="rope", pos_learnable=True, weight_scale=0.02,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.eta = eta
        self.weight_scale = weight_scale
        self.optim_type = optim_type
        self.position_encoding = position_encoding
        self.use_postional = (self.position_encoding == "positional")
        self.pos_learnable = pos_learnable
        #separate keys for word and positional embeddings
        word_key =random.PRNGKey(1234)
        pos_key = random.fold_in(word_key,1)
        
        word_weights = random.normal(word_key, (vocab_size, embed_dim)) * weight_scale
        if self.use_postional:
            if self.pos_learnable:
                pos_weights = random.normal(pos_key, (seq_len, embed_dim)) * weight_scale
            else:
                pos_weights = _create_sinusoidal_embeddings(seq_len, embed_dim)
        else:
            # unused in RopE mode
            pos_weights = jnp.zeros((seq_len, embed_dim))
        
        ## Compartments
        self.inputs = Compartment(jnp.zeros((batch_size, seq_len), dtype=jnp.int32))
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, embed_dim)))
        self.word_weights = Compartment(word_weights)
        self.pos_weights = Compartment(pos_weights)
        self.post = Compartment(jnp.zeros((batch_size, seq_len, embed_dim)))
        
        self.dWordWeights = Compartment(jnp.zeros((vocab_size, embed_dim)))
        self.dPosWeights = Compartment(jnp.zeros((seq_len, embed_dim)))
        # Optimization
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)
        self.word_opt_params = Compartment(
            get_opt_init_fn(optim_type)([self.word_weights.get()])
        )
        #postion optimizer for learned only
        if(self.use_postional and self.pos_learnable):
            self.pos_opt_params = Compartment(
                get_opt_init_fn(optim_type)([self.pos_weights.get()])
            )
        else:
            self.pos_opt_params = Compartment(None)
    @compilable
    def advance_state(self):
        """
        Forward pass: output = word_embedding[inputs]
        """
        inputs=self.inputs.get()
        word_weights=self.word_weights.get()
        seq_len=self.seq_len.get()
        embed_dim=self.embed_dim.get()
        batch_size = inputs.shape[0]
        
        flat_tokens = inputs.reshape(-1).astype(jnp.int32)
        word_embeds_flat = word_weights[flat_tokens]
        word_embeds = word_embeds_flat.reshape(batch_size, seq_len, embed_dim)
        
        if self.use_postional:
            pos_weights = self.pos_weights.get()
            positions = jnp.arange(seq_len)
            pos_embeds= pos_weights[positions]
            pos_embeds_batch = jnp.broadcast_to(pos_embeds, (batch_size, seq_len, embed_dim))

            outputs = (word_embeds +pos_embeds_batch)
        else:
            #RoPE mode
            outputs = word_embeds
        self.outputs.set(outputs)

  
    @compilable
    def evolve(self):
        """
        Learning step: Hebbian updates for word embeddings and update postional embeddings only when using learned absolute postional embeddings
        """
        opt = self.opt.get()
        vocab_size = self.vocab_size.get()
        seq_len = self.seq_len.get()
        embed_dim = self.embed_dim.get()
        batch_size = self.batch_size.get()
        inputs = self.inputs.get()
        post = self.post.get()
        word_weights = self.word_weights.get()
        pos_weights = self.pos_weights.get()
        word_opt_params = self.word_opt_params.get()
        
        # Compute embedding updates
        inputs= inputs.astype(jnp.int32)
        (d_word_weights, d_pos_weights)  = _compute_embedding_updates(
            inputs, post, vocab_size, seq_len, embed_dim, batch_size
        )
        
        word_opt_params, [new_word_weights] = opt(
            word_opt_params, [word_weights], [d_word_weights]
        )
        
        self.word_weights.set(new_word_weights)
        self.word_opt_params.set(word_opt_params)
        self.dWordWeights.set(d_word_weights)

        if (self.use_postional and self.pos_learnable):
            pos_opt_params = self.pos_opt_params.get()
            pos_opt_params, [new_pos_weights] = opt(
                pos_opt_params, [pos_weights], [d_pos_weights]
            )
            self.pos_weights.set(new_pos_weights)
            self.pos_opt_params.set(pos_opt_params)
            self.dPosWeights.set(d_pos_weights)
        else:
            #fixed sinusoidal postions and RoPE 
            self.dPosWeights.set(jnp.zeros_like(pos_weights))
        
    @compilable
    def reset(self):
        """
        Reset compartments to zeros
        """
        batch_size = self.batch_size.get()
        seq_len = self.seq_len.get()
        embed_dim = self.embed_dim.get()
        vocab_size = self.vocab_size.get()

        inputs = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        outputs = jnp.zeros((batch_size, seq_len, embed_dim))
        post = jnp.zeros((batch_size, seq_len, embed_dim))
        dWordWeights = jnp.zeros((vocab_size, embed_dim))
        dPosWeights = jnp.zeros((seq_len, embed_dim))
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.post.set(post)
        self.dWordWeights.set(dWordWeights)
        self.dPosWeights.set(dPosWeights)


    @classmethod
    def help(cls):
        """Component help function"""
        properties = {
            "synapse_type": "EmbeddingSynapse - returns a single word embedding representation"
        }
        compartment_props = {
            "inputs": 
                {"inputs": "Input token indices (batch_size, seq_len)",
                 "post": "Post-synaptic error signals for learning"},
            "states":
                {"word_weights": "Word embedding matrix (vocab_size, embed_dim)",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWordWeights": "Word embedding adjustment matrix"},
            "outputs":
                {"outputs": "Embeddings (batch_size, seq_len, embed_dim)"},
        }
        hyperparams = {
            "vocab_size": "Size of vocabulary",
            "seq_len": "Maximum sequence length", 
            "embed_dim": "Dimensionality of embeddings",
            "batch_size": "Batch size dimension",
            "eta": "Global learning rate",
            "optim_type": "Optimization scheme",
            "weight_scale": "Weight initialization scale"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = word_embedding[inputs]",
                "hyperparameters": hyperparams}
        return info
    def __repr__(self):
        # FIX: Replaced the non-existent Compartment.is_compartment with isinstance(..., Compartment)
        comps = [varname for varname in dir(self) if isinstance(getattr(self, varname), Compartment)]
        
        if not comps:
            # Handle the case where no compartments are found to avoid max() on an empty sequence
            return f"[{self.__class__.__name__}] PATH: {self.name}\n  No Compartments Found"

        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        
        # Iterate over the valid compartment names
        for c in comps:
            # Get the actual Compartment object
            compartment_obj = getattr(self, c) 
            
            # Get tensor statistics (assuming tensorstats is correctly imported)
            stats = tensorstats(compartment_obj.get())
            
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
                
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
            
        return lines


    def save(self, directory, **kwargs):
        """Save word embedding parameters and learned positional embedding parameters to disk."""
        
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(directory, f"{self.name}.npz")
        
        if (self.use_positional and self.pos_learnable):
            jnp.savez(
                file_name,
                word_weights=(self.word_weights.get()),
                pos_weights=(self.pos_weights.get()),
            )
        else:
            jnp.savez(
                file_name,
                word_weights=( self.word_weights.get()),
            )
      

    def load(self, directory, **kwargs):
        """Load word embedding parameters from disk."""
        import os
        file_name = os.path.join(directory, f"{self.name}.npz")
        data = jnp.load(file_name)
        
        self.word_weights.set(data['word_weights'])

        if (self.use_positional and self.pos_learnable ):
            if "pos_weights" not in data:
                raise ValueError(
                    "The checkpoint has no "
                    "pos_weights. It may have "
                    "been trained using RoPE."
                )

            self.pos_weights.set(
                data["pos_weights"]
            )