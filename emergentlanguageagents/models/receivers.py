"""
Listener models
"""


import torch
import torch.nn as nn

from . import rnn
from . import cnn

class Recpeiver(nn.Module):
    # TODO: implement this as the base class for receivers. Maybe use ABCs!
    pass

class CopyListener(nn.Module):
    def __init__(self, feat_model, message_size=100, dropout=0.2, **kwargs):
        super().__init__()

        self.feat_model = getattr(cnn, kwargs['cnn'])
        self.feat_size = self.feat_model.final_feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.message_size = message_size

        if self.message_size is None:
            self.bilinear = nn.Linear(self.feat_size, 1, bias=False)
        else:
            self.bilinear = nn.Linear(self.message_size, self.feat_size, bias=False)

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.view(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)

        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        feats_emb = self.dropout(feats_emb)

        return feats_emb

    def compare(self, feats_emb, message_enc):
        """
        Compute dot products
        """
        scores = torch.einsum("ijh,ih->ij", (feats_emb, message_enc))
        return scores

    def forward(self, feats, message):
        # Embed features
        feats_emb = self.embed_features(feats)

        # Embed message
        if self.message_size is None:
            return self.bilinear(feats_emb).squeeze(2)
        else:
            message_bilinear = self.bilinear(message)

            return self.compare(feats_emb, message_bilinear)

    def reset_parameters(self):
        self.feat_model.reset_parameters()
        self.bilinear.reset_parameters()

# TODO: make a generic receiver class and then make the below the EmComGen-specific receiver "EmComGenReceiver"

class Listener(CopyListener):
    def __init__(self, feat_model, **kwargs):
        super().__init__(feat_model, **kwargs)
        self.vocab_size = kwargs['vocabulary']
        self.message_length = kwargs['message_length']
        self.embedding_size = kwargs['embedding_size']
        self.hidden_size = kwargs['hidden_size']

        self.embedding = nn.Embedding(
            self.vocabulary + 3, # (account for SOS, EOS, UNK)
            self.embedding_size
        )
        self.lang_model = rnn.RNNEncoder(self.embedding, hidden_size=self.hidden_size)

    def forward(self, feats, lang, lang_length):
        # Embed features
        feats_emb = self.embed_features(feats)

        # Embed language
        lang_emb = self.lang_model(lang, lang_length)

        # Bilinear term: lang embedding space -> feature embedding space
        lang_bilinear = self.bilinear(lang_emb)

        return self.compare(feats_emb, lang_bilinear)

    def reset_parameters(self):
        super().reset_parameters()
        self.embedding.reset_parameters()
        self.lang_model.reset_parameters()
