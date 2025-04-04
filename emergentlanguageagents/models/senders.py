"""
Speaker models
"""

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.distributions import Gumbel

from . import image_encoders
# import data
# import data.language

class ECGCopySpeaker(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        assert "image_encoder" in kwargs
        assert "dropout" in kwargs

        self.feat_model = getattr(image_encoders, kwargs["image_encoder"])()
        self.feat_size = self.feat_model.final_feat_dim
        self.emb_size = 2 * self.feat_size
        self.dropout = nn.Dropout(p=kwargs["dropout"])

    def embed_features(self, feats, targets=None):
        """
        Prototype to embed positive and negative examples of concept
        """
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.view(batch_size * n_obj, *rest)

        feats_emb_flat = self.feat_model(feats_flat)

        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)

        if targets is None:
            feats_emb_dropout = self.dropout(feats_emb)
            return feats_emb_dropout
        else:
            prototypes = self.form_prototypes(feats_emb, targets)
            prototypes_dropout = self.dropout(prototypes)

            return prototypes_dropout

    def add_cls_token(self, feats):
        cls_embs = self.cls_emb.unsqueeze(0).unsqueeze(1).expand(feats.shape[0], -1, -1)
        return torch.cat([cls_embs, feats], 1)

    def form_prototypes(self, feats_emb, targets):
        """
        Given embedded features and targets, form into prototypes (i.e. average
            together positive examples, average together negative examples).

        Note that the original code from
            https://github.com/jayelm/emergent-generalization
            allows for two kinds of prototype, but I have deleted all the code
            related to the transformer prototypes as we will always choose
            "average" prototypes in practice when we try to reproduce
            https://arxiv.org/abs/2106.02668
        """
        rev_targets = 1 - targets
        pos_proto = (feats_emb * targets.unsqueeze(2)).sum(1)
        neg_proto = (feats_emb * rev_targets.unsqueeze(2)).sum(1)

        n_pos = targets.sum(1, keepdim=True)
        n_neg = rev_targets.sum(1, keepdim=True)

        # Avoid div by 0 (when n_pos is clamped to min 1, pos_proto is all 0s
        # anyways)
        n_pos = torch.clamp(n_pos, min=1)
        n_neg = torch.clamp(n_neg, min=1)

        # Divide by sums (avoid div by 0 error)
        pos_proto = pos_proto / n_pos
        neg_proto = neg_proto / n_neg

        ft_concat = torch.cat([pos_proto, neg_proto], 1)

        return ft_concat

    def forward(self, feats, targets):
        """
        Pass through entire model hidden state
        """
        return self.embed_features(feats, targets)


class ECGSpeaker(ECGCopySpeaker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # These are just some globals from 
        # https://github.com/jayelm/emergent-generalization/
        #     blob/master/code/data/language.py
        self.pad_index = 0
        self.sos_index = 1
        self.eos_index = 2
        self.unk_index = 3

        needed = ["vocabulary", "embedding_size", "d_model", "temperature"]
        assert all([key in kwargs for key in needed])

        self.embedding = nn.Embedding(
            kwargs["vocabulary"] + 3, # (account for SOS, EOS, UNK)
            kwargs["embedding_size"]
        )

        self.embedding_dim = self.embedding.embedding_dim
        self.vocab_size = self.embedding.num_embeddings
        self.hidden_size = kwargs["d_model"]
        self.tau = kwargs["temperature"]

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        # 2 * feat_size - one for positive prototype, one for negative
        self.init_h = nn.Linear(2 * self.feat_size, self.hidden_size)
        self.bilinear = nn.Linear(self.hidden_size, self.feat_size, bias=False)

    def sample(
        self,
        states,
        greedy=False,
        max_len=4,
        eps=0.0,
        softmax_temp=1.0,
        uniform_weight=0.0,
    ):
        """ """
        batch_size = states.shape[1]  # 0th dim is singleton for GRU
        # This contains are series of sampled onehot vectors
        lang = []
        # And verctor lengths
        lang_length = torch.ones(batch_size, dtype=torch.int64).to(states.device)
        done_sampling = [False for _ in range(batch_size)]

        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size).to(states.device)
        inputs_onehot[:, self.sos_index] = 1.0

        # (batch_size, len, n_vocab)
        inputs_onehot = inputs_onehot.unsqueeze(1)

        # Add SOS to lang
        lang.append(inputs_onehot)

        # (B,L,D) to (L,B,D)
        inputs_onehot = inputs_onehot.transpose(0, 1)

        # compute embeddings
        # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
        inputs = inputs_onehot @ self.embedding.weight

        for i in range(max_len - 2):  # Have room for SOS, EOS if never sampled
            # FIXME: This is inefficient since I do sampling even if we've
            # finished generating language.
            if all(done_sampling):
                break
            outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
            outputs = outputs.squeeze(0)  # outputs: (B,H)
            outputs = self.outputs2vocab(outputs)  # outputs: (B,V)

            if greedy:
                predicted = outputs.max(1)[1]
                predicted = predicted.unsqueeze(1)
            else:
                # Normalize first
                outputs = torch.log_softmax(outputs, -1)

                if uniform_weight != 0.0:
                    uniform_outputs = torch.full_like(
                        outputs, np.log(1 / outputs.shape[1])
                    )
                    # Weighted average of logits and uniform distribution in log space
                    combined_outputs = torch.stack(
                        [
                            uniform_outputs + np.log(uniform_weight),
                            outputs + np.log(1 - uniform_weight),
                        ],
                        2,
                    )
                    outputs = torch.logsumexp(combined_outputs, 2)

                if softmax_temp != 1.0:
                    outputs = outputs / softmax_temp

                predicted_onehot = F.gumbel_softmax(outputs, tau=self.tau, hard=True)

                # Epsilon - random sample (not sure if this works)
                if np.random.random() < eps:
                    random_i = torch.randint(
                        outputs.shape[1], (outputs.shape[0], 1)
                    ).to(predicted_onehot.device)
                    random_onehot = torch.zeros_like(predicted_onehot).scatter_(
                        -1, random_i, 1.0
                    )
                    predicted_onehot = (
                        random_onehot - predicted_onehot.detach()
                    ) + predicted_onehot

                # Add to lang
                lang.append(predicted_onehot.unsqueeze(1))

            predicted_npy = predicted_onehot.argmax(1).cpu().numpy()

            # Update language lengths
            for i, pred in enumerate(predicted_npy):
                if not done_sampling[i]:
                    lang_length[i] += 1
                if pred == self.eos_index:
                    done_sampling[i] = True

            # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
            inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight

        # Add EOS if we've never sampled it
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size).to(states.device)
        eos_onehot[:, 0, self.eos_index] = 1.0
        lang.append(eos_onehot)
        # Cut off the rest of the sentences
        for i, _ in enumerate(predicted_npy):
            if not done_sampling[i]:
                lang_length[i] += 1
            done_sampling[i] = True

        # Cat language tensors
        lang_tensor = torch.cat(lang, 1)

        # Trim max length
        max_lang_len = lang_length.max()
        lang_tensor = lang_tensor[:, :max_lang_len, :]

        return lang_tensor, lang_length

    def forward(self, feats, targets, **kwargs):
        """Sample from image features"""
        feats_emb = self.embed_features(feats, targets)
        # initialize hidden states using image features
        states = self.init_h(feats_emb)

        return self.sample(states.unsqueeze(0), **kwargs), states

    def classify(self, feats, targets, test_feats):
        feats_emb = self.embed_features(feats, targets)
        states = self.init_h(feats_emb)
        return self.classify_from_states(states, test_feats)

    def classify_from_states(self, states, test_feats):
        states = self.bilinear(states)
        test_feats_emb = self.embed_features(test_feats)
        scores = torch.bmm(test_feats_emb, states.unsqueeze(2)).squeeze(2)
        return scores
    
    # This is never used in practice, so we're commenting it out to remove
    # a coupling to the `data` module and can reimplement it later if needed
    # def to_text(self, lang_onehot):
    #     texts = []
    #     lang = lang_onehot.argmax(2)
    #     for sample in lang.cpu().numpy():
    #         text = []
    #         for item in sample:
    #             text.append(data.ITOS[item])
    #             if item == data.EOS_IDX:
    #                 break
    #         texts.append(" ".join(text))
    #     return np.array(texts, dtype=np.unicode_)
