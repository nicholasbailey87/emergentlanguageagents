"""
Build a model pair from an experiment config
"""

# from . import base
from . import senders
from . import receivers
import importlib
from torch import nn


class Pair(nn.Module):
    def __init__(self, sender, receiver):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        # TODO: work out if these criterions need to be here:
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.xent_criterion = nn.CrossEntropyLoss()

def build_pair_from_config(config: dict) -> Pair:
    
    # TODO: Make feature extraction model part of agents
    sender_feat_model = feat_fn("sender")
    receiver_feat_model = feat_fn("receiver")

    # (account for SOS, EOS, UNK)
    # TODO: make embedding layer part of agents
    sender_embs = nn.Embedding(args.vocab_size + 3, args.embedding_size)
    receiver_embs = nn.Embedding(args.vocab_size + 3, args.embedding_size)
    
    # TODO: change `speaker` and `listener` in all modules to `sender` and `receiver`
    sender_class = getattr(senders, config['sender']['class'])
    sender = sender_class(**config['sender']['arguments'])

    receiver_class = getattr(receivers, config['receiver']['class'])
    receiver = receiver_class(**config['receiver']['arguments'])

    pair = Pair(sender, receiver)

    if args.cuda: # TODO: Make sending to the GPU part of the training, not part of the model definition
        pair = pair.cuda() 
    
    # TODO: make sure optimisation is handled elsewhere


    # optimizer = optim.Adam(opt_params, lr=args.lr)

    # TODO: Incorporate ReduceLROnPlateau as the default option for emcomgen reproduction
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     factor=0.5,
    #     patience=10,
    # )

    # TODO: Make the code robust to this function only returning the pair, not the optimisers
    return pair
# {
#         "pair": pair,
#         "optimizer": optimizer,
#         "scheduler": scheduler,
#     }
