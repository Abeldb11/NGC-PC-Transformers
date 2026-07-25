
from ngclearn.components import GaussianErrorCell, RateCell
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from config import Config as config
from utils.embed_utils import EmbeddingSynapse
from utils.precision_error_cell import AdaptivePrecisionErrorCell
from jax import random


def _make_error_cell(name, n_units, batch_size, site_name):
    sites = getattr(config, "precision_sites", None)
    use_here = getattr(config, "use_precision_weighting", False) and (sites is None or site_name in sites)
    if use_here:
        return AdaptivePrecisionErrorCell(name, n_units=n_units, batch_size=batch_size,
                                           momentum=config.precision_momentum,
                                           sigma_min=config.precision_sigma_min,
                                           sigma_init=config.precision_sigma_init)
    return GaussianErrorCell(name, n_units=n_units, batch_size=batch_size)

class EMBEDDING:
    """
   embedding layer using the EmbeddingSynapse
    """
    def __init__(self, dkey, vocab_size, seq_len, embed_dim, batch_size, pos_learnable, eta, optim_type, **kwargs):
        
        dkey, *subkeys = random.split(dkey, 4)
    
        # RateCell expects a 3D shape tuple for image components (seq_len, embed_dim, channels)so here we use the third dim as a placeholder
        self.z_embed = RateCell("z_embed", n_units=seq_len, tau_m=0., 
                                  act_fx="identity", batch_size=batch_size)            
            # EmbeddingSynapse (handles both word + position internally)
        self.W_embed = EmbeddingSynapse(
                "W_embed", 
                vocab_size=vocab_size,
                seq_len=seq_len,
                embed_dim=embed_dim, 
                batch_size=batch_size,
                pos_learnable=pos_learnable,
                eta=eta,
                use_positional_encoding=getattr(config, "use_positional_encoding", True),
                optim_type=optim_type,
                key=subkeys[0])
            
        # self.e_embed = ErrorCell("e_embed", n_units=embed_dim, 
          #                        batch_size=batch_size * seq_len) # shape=(seq_len, embed_dim, 1),
        
        self.e_embed = _make_error_cell("e_embed", embed_dim, batch_size * seq_len, "e_embed")
    
            

