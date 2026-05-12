import torch
import torch.nn as nn
from benchmarks.models.model import Model


class VanillaRNN(Model):

    def __init__(self, name, iosize, d_model=128, n_layers=1, dropout=0.0, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding  = nn.Embedding(iosize, d_model)
        self.rnn        = nn.RNN(d_model, d_model, num_layers=n_layers,
                                 batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.output_proj = nn.Linear(d_model, iosize)

    def forward(self, x, state=None, unroll_steps=-1, return_state=False):
        emb = self.embedding(x)
        out, h_n = self.rnn(emb, state)
        logits = self.output_proj(out)
        if return_state:
            return logits, h_n
        return logits

