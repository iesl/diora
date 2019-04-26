import json
import types

import torch

from train import argument_parser, parse_args, configure
from train import get_validation_dataset, get_validation_iterator
from train import build_net

from diora.logging.configuration import get_logger

from diora.analysis.cky import ParsePredictor as CKY


def override_init_with_batch(var):
    init_with_batch = var.init_with_batch

    def func(self, *args, **kwargs):
        init_with_batch(*args, **kwargs)
        self.saved_scalars = {i: {} for i in range(self.length)}
        self.saved_scalars_out = {i: {} for i in range(self.length)}

    var.init_with_batch = types.MethodType(func, var)


def override_inside_hook(var):
    def func(self, level, h, c, s):
        length = self.length
        B = self.batch_size
        L = length - level

        assert s.shape[0] == B
        assert s.shape[1] == L
        # assert s.shape[2] == N
        assert s.shape[3] == 1
        assert len(s.shape) == 4
        smax = s.max(2, keepdim=True)[0]
        s = s - smax

        for pos in range(L):
            self.saved_scalars[level][pos] = s[:, pos, :]

    var.inside_hook = types.MethodType(func, var)


def replace_leaves(tree, leaves):
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            return 1, leaves[pos]

        newtree = []
        sofar = 0
        for node in tr:
            size, newnode = func(node, pos+sofar)
            sofar += size
            newtree += [newnode]

        return sofar, newtree

    _, newtree = func(tree)

    return newtree


def run(options):
    logger = get_logger()

    validation_dataset = get_validation_dataset(options)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    word2idx = validation_dataset['word2idx']
    embeddings = validation_dataset['embeddings']

    idx2word = {v: k for k, v in word2idx.items()}

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)

    # Parse

    diora = trainer.net.diora

    ## Monkey patch parsing specific methods.
    override_init_with_batch(diora)
    override_inside_hook(diora)

    ## Turn off outside pass.
    trainer.net.diora.outside = False

    ## Eval mode.
    trainer.net.eval()

    ## Topk predictor.
    parse_predictor = CKY(net=diora, word2idx=word2idx)

    batches = validation_iterator.get_iterator(random_seed=options.seed)

    logger.info('Beginning to parse.')

    with torch.no_grad():
        for i, batch_map in enumerate(batches):
            sentences = batch_map['sentences']
            batch_size = sentences.shape[0]
            length = sentences.shape[1]

            # Rather than skipping, just log the trees (they are trivially easy to find).
            if length <= 2:
                for i in range(batch_size):
                    example_id = batch_map['example_ids'][i]
                    tokens = sentences[i].tolist()
                    words = [idx2word[idx] for idx in tokens]
                    if length == 2:
                        o = dict(example_id=example_id, tree=(words[0], words[1]))
                    elif length == 1:
                        o = dict(example_id=example_id, tree=words[0])
                    print(json.dumps(o))
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            trees = parse_predictor.parse_batch(batch_map)

            for ii, tr in enumerate(trees):
                example_id = batch_map['example_ids'][ii]
                s = [idx2word[idx] for idx in sentences[ii].tolist()]
                tr = replace_leaves(tr, s)
                o = dict(example_id=example_id, tree=tr)

                print(json.dumps(o))


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
