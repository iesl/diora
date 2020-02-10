import os
import collections
import json
import types

import torch

from tqdm import tqdm

from train import argument_parser, parse_args, configure
from train import get_validation_dataset, get_validation_iterator
from train import build_net

from diora.logging.configuration import get_logger

from diora.analysis.cky import ParsePredictor as CKY


punctuation_words = set([x.lower() for x in ['.', ',', ':', '-LRB-', '-RRB-', '\'\'',
    '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']])


def remove_using_flat_mask(tr, mask):
    kept, removed = [], []
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        node = []

        for subtree in tr:
            x, xsize = func(subtree, pos=pos + size)
            if x is not None:
                node.append(x)
            size += xsize

        if len(node) == 1:
            node = node[0]
        elif len(node) == 0:
            return None, size
        return node, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)


def postprocess(tr, tokens=None):
    if tokens is None:
        tokens = flatten_tree(tr)

    # Don't remove the last token. It's not punctuation.
    if tokens[-1].lower() not in punctuation_words:
        return tr

    mask = [True] * (len(tokens) - 1) + [False]
    tr, kept, removed = remove_using_flat_mask(tr, mask)
    assert len(kept) == len(tokens) - 1, 'Incorrect tokens left. Original = {}, Output = {}, Kept = {}'.format(
        binary_tree, tr, kept)
    assert len(kept) > 0, 'No tokens left. Original = {}'.format(tokens)
    assert len(removed) == 1
    tr = (tr, tokens[-1])

    return tr


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

    output_path = os.path.abspath(os.path.join(options.experiment_path, 'parse.jsonl'))

    logger.info('Beginning to parse.')
    logger.info('Writing output to = {}'.format(output_path))

    f = open(output_path, 'w')

    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
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
                        o = collections.OrderedDict(example_id=example_id, tree=(words[0], words[1]))
                    elif length == 1:
                        o = collections.OrderedDict(example_id=example_id, tree=words[0])
                    f.write(json.dumps(o) + '\n')
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            trees = parse_predictor.parse_batch(batch_map)

            for ii, tr in enumerate(trees):
                example_id = batch_map['example_ids'][ii]
                s = [idx2word[idx] for idx in sentences[ii].tolist()]
                tr = replace_leaves(tr, s)
                if options.postprocess:
                    tr = postprocess(tr, s)
                o = collections.OrderedDict(example_id=example_id, tree=tr)

                f.write(json.dumps(o) + '\n')

    f.close()


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
