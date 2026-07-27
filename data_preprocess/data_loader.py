import jax.numpy as jnp
from pathlib import Path
from ngclearn.utils.data_loader import DataLoader as NGCDataLoader
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
DIR = Path(__file__).parent
sys.path.append(str(DIR.parent))

class DataLoader:
    def __init__(self, seq_len, batch_size, data_dir= DIR / "outputs" / "tokenized_data"):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_token = -1

    def load_and_prepare_data(self):
        """Load tokenized data and prepare for training"""
        train_tokens = np.load(self.data_dir / "train_tokens.npy")
        valid_tokens = np.load(self.data_dir / "valid_tokens.npy")
        test_tokens = np.load(self.data_dir / "test_tokens.npy")

        train_loader = self._create_data_loader(train_tokens, shuffle=True)
        valid_loader = self._create_data_loader(valid_tokens, shuffle=False)
        test_loader = self._create_data_loader(test_tokens, shuffle=False)

        return train_loader, valid_loader, test_loader

    def _create_data_loader(self, tokens, shuffle):
        """Create sequences and return NGC DataLoader"""
        tokens = np.asarray(tokens)
        window_size = self.seq_len + 1
        stride = self.seq_len
        n_tokens = len(tokens)
        num_sequences = (n_tokens - window_size) // stride + 1

        if num_sequences <= 0:
            padded_tokens = np.concatenate([
                tokens,
                np.full((window_size - len(tokens),), self.pad_token)
            ])
            sequences = padded_tokens.reshape(1, -1)
        else:
            # Vectorized windowing instead of a Python loop over millions of slices
            windows = sliding_window_view(tokens, window_size)[::stride]
            sequences = np.array(windows[:num_sequences])

        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        mask = (targets != self.pad_token).astype(np.float32)

        return NGCDataLoader(
            design_matrices=[
                ("inputs", jnp.array(inputs)),
                ("targets", jnp.array(targets)),
                ("mask", jnp.array(mask))
            ],
            batch_size=self.batch_size,
            disable_shuffle=not shuffle,
            ensure_equal_batches=True
        )