"""
RNNs/language models etc
"""

import abc
from typing import List, Union, Tuple, Optional

import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn

# What does a language embedding model need?
# It needs to be an nn.Module
# It needs to 

class LanguageModel(abc.ABC):
    """
    Abstract base class for the language model task. A language model (by this
        definition) is any object that can receive a series of vectors expressed
        as a torch.Tensor, and return an encoding.
    """

    @abc.abstractmethod
    def encode(self, x: torch.Tensor):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        """
        This class method, based on code here
            https://docs.python.org/3/library/abc.html#abc.ABCMeta.__subclasshook__
            , allows anything to successfully type check as a language model if it
            implements the right methods. This supports the Open-Closed Principle
            and is essentially what `abc.ABC`s are for in Python.
        """
        if cls is LanguageModel:
            if (
                any("encode" in B.__dict__ for B in C.__mro__)
            ):
                return True
        return NotImplemented

class EmComGenLanguageModel(LanguageModel, nn.Module):
    """
    The language model used in https://arxiv.org/abs/2106.02668
    Uses sequence packing for efficiency, but has to pass around message lengths.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.input_size, self.hidden_size)
    
    def encode(self, seq, length):
        return self.forward(self, seq, length)

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq @ self.embedding.weight

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist(),
        )

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.gru.reset_parameters()