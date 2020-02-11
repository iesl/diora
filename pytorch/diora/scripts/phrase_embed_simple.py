import os
import collections
import json
import types

import torch
import numpy as np
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


def tree_to_spans(tree):
    spans = []
    def helper(tr, pos=0):
        if isinstance(tr, str) or len(tr) == 1 and isinstance(tr[0], str):
            return 1
        if len(tr) == 1:
            return helper(tr[0], pos)
        size = 0
        for x in tr:
            xsize = helper(x, pos+size)
            size += xsize
        spans.append((pos, size))
        return size
    _ = helper(tree)
    return spans


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


class TreeHelper(object):
    def __init__(self, diora, word2idx):
        self.diora = diora
        self.word2idx = word2idx
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}

    def init(self, options):
        if options.parse_mode == 'latent':
            self.parse_predictor = CKY(net=self.diora, word2idx=self.word2idx)
            ## Monkey patch parsing specific methods.
            override_init_with_batch(self.diora)
            override_inside_hook(self.diora)

    def get_trees_for_batch(self, batch_map, options):
        sentences = batch_map['sentences']
        batch_size = sentences.shape[0]
        length = sentences.shape[1]

        # trees
        if options.parse_mode == 'all-spans':
            raise Exception('Does not support this mode.')
        elif options.parse_mode == 'latent':
            trees = self.parse_predictor.parse_batch(batch_map)
        elif options.parse_mode == 'given':
            trees = batch_map['trees']

        # spans
        spans = []
        for ii, tr in enumerate(trees):
            s = [self.idx2word[idx] for idx in sentences[ii].tolist()]
            tr = replace_leaves(tr, s)
            if options.postprocess:
                tr = postprocess(tr, s)
            spans.append(tree_to_spans(tr))

        return trees, spans


class CSVHelper(object):
    def __init__(self):
        self.header = ['example_id', 'position', 'size']

    def write_header(self, f):
        f.write(','.join(self.header) + '\n')

    def write_row(self, f, data):
        row = ','.join([data[k] for k in self.header])
        f.write(row + '\n')


def run(options):
    logger = get_logger()

    validation_dataset = get_validation_dataset(options)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    word2idx = validation_dataset['word2idx']
    embeddings = validation_dataset['embeddings']

    idx2word = {v: k for k, v in word2idx.items()}

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)
    diora = trainer.net.diora
    tree_helper = TreeHelper(diora, word2idx)
    tree_helper.init(options)
    csv_helper = CSVHelper()

    ## Eval mode.
    trainer.net.eval()

    batches = validation_iterator.get_iterator(random_seed=options.seed)

    meta_output_path = os.path.abspath(os.path.join(options.experiment_path, 'vectors.csv'))
    vec_output_path = os.path.abspath(os.path.join(options.experiment_path, 'vectors.npy'))

    logger.info('Beginning.')
    logger.info('Writing vectors to = {}'.format(vec_output_path))
    logger.info('Writing metadata to = {}'.format(meta_output_path))

    f_csv = open(meta_output_path, 'w')
    f_vec = open(vec_output_path, 'ab')
    csv_helper.write_header(f_csv)

    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            sentences = batch_map['sentences']
            batch_size = sentences.shape[0]
            length = sentences.shape[1]

            # Skip very short sentences.
            if length <= 2:
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            if options.parse_mode == 'all-spans':
                for ii in range(batch_size):
                    example_id = batch_map['example_ids'][ii]
                    for level in range(length):
                        size = level + 1
                        for pos in range(length - level):
                            # metadata
                            csv_helper.write_row(f_csv,
                                collections.OrderedDict(
                                    example_id=example_id,
                                    position=str(pos),
                                    size=str(size)
                            ))
                inside_vectors = diora.inside_h.view(-1, options.hidden_dim)
                outside_vectors = diora.outside_h.view(-1, options.hidden_dim)

            else:
                trees, spans = tree_helper.get_trees_for_batch(batch_map, options)

                batch_index = []
                cell_index = []
                offset_cache = diora.index.get_offset(length)

                for ii, sp_lst in enumerate(spans):
                    example_id = batch_map['example_ids'][ii]
                    for pos, size in sp_lst:
                        # metadata
                        csv_helper.write_row(f_csv,
                            collections.OrderedDict(
                                example_id=example_id,
                                position=str(pos),
                                size=str(size)
                        ))
                        # for vectors
                        level = size - 1
                        cell = offset_cache[level] + pos
                        batch_index.append(ii)
                        cell_index.append(cell)

                inside_vectors = diora.inside_h[batch_index, cell_index]
                assert inside_vectors.shape == (len(batch_index), options.hidden_dim)
                outside_vectors = diora.outside_h[batch_index, cell_index]
                assert outside_vectors.shape == (len(batch_index), options.hidden_dim)

            vectors = np.concatenate([inside_vectors, outside_vectors], axis=1)
            np.savetxt(f_vec, vectors)

    f_csv.close()
    f_vec.close()

    # X = np.loadtxt(vec_output_path)
    # print(X.shape)


if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('--parse_mode', default='latent', choices=('all-spans', 'latent', 'given'), help=
        'Save vectors for...\n- `all-spans`: the whole chart,\n- `latent`: the latent tree,\n- `given`: a given tree.')
    options = parse_args(parser)
    configure(options)

    run(options)
