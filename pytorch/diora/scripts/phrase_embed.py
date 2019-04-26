"""
A script to embed every phrase in a dataset as a dense vector, then
to find the top-k neighbors of each phrase according to cosine
similarity.

1. Install missing dependencies.

    # More details: https://github.com/facebookresearch/faiss/blob/master/INSTALL.md
    conda install faiss-cpu -c pytorch

2. Prepare data. For example, the chunking dataset from CoNLL 2000.

    wget https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
    gunzip train.txt.gz
    python diora/misc/convert_conll_to_jsonl.py --path train.txt > conll-train.jsonl

3. Run this script.

    python diora/scripts/phrase_embed.py \
        --batch_size 10 \
        --emb w2v \
        --embeddings_path ~/data/glove.6B/glove.6B.50d.txt \
        --hidden_dim 50 \
        --log_every_batch 100 \
        --save_after 1000 \
        --data_type conll_jsonl \
        --validation_path ./conll-train.jsonl \
        --validation_filter_length 10

Can control the number of neighbors to show with the `--k_top` flag.

Can control the number of candidates to consider with `--k_candidates` flag.

"""


import json
import types
import itertools

import torch

import numpy as np

from train import argument_parser, parse_args, configure
from train import get_validation_dataset, get_validation_iterator
from train import build_net

from diora.logging.configuration import get_logger

try:
    import faiss
    from faiss import normalize_L2
except:
    print('Could not import `faiss`, which is used to find nearest neighbors.')


def get_cell_index(entity_labels, i_label=0, i_pos=1, i_size=2):
    def helper():
        for i, lst in enumerate(entity_labels):
            for el in lst:
                if el is None:
                    continue
                pos = el[i_pos]
                size = el[i_size]
                label = el[i_label]
                yield (i, pos, size, label)
    lst = list(helper())
    if len(lst) == 0:
        return None, []
    batch_index = [x[0] for x in lst]
    positions = [x[1] for x in lst]
    sizes = [x[2] for x in lst]
    labels = [x[3] for x in lst]

    return batch_index, positions, sizes, labels


def get_many_cells(diora, chart, batch_index, positions, sizes):
    cells = []
    length = diora.length

    idx = []
    for bi, pos, size in zip(batch_index, positions, sizes):
        level = size - 1
        offset = diora.index.get_offset(length)[level]
        absolute_pos = offset + pos
        idx.append(absolute_pos)

    cells = chart[batch_index, idx]

    return cells


def get_many_phrases(batch, batch_index, positions, sizes):
    batch = batch.tolist()
    lst = []
    for bi, pos, size in zip(batch_index, positions, sizes):
        phrase = tuple(batch[bi][pos:pos+size])
        lst.append(phrase)
    return lst


class BatchRecorder(object):
    def __init__(self, dtype={}):
        super(BatchRecorder, self).__init__()
        self.cache = {}
        self.dtype = dtype
        self.dtype2flatten = {
            'list': self._flatten_list,
            'np': self._flatten_np,
            'torch': self._flatten_torch,
        }

    def _flatten_list(self, v):
        return list(itertools.chain(*v))

    def _flatten_np(self, v):
        return np.concatenate(v, axis=0)

    def _flatten_torch(self, v):
        return torch.cat(v, 0).cpu().data.numpy()

    def get_flattened_result(self):
        def helper():
            for k, v in self.cache.items():
                flatten = self.dtype2flatten[self.dtype.get(k, 'list')]
                yield k, flatten(v)
        return {k: v for k, v in helper()}
            
    def record(self, **kwargs):
        for k, v in kwargs.items():
            self.cache.setdefault(k, []).append(v)


class Index(object):
    def __init__(self, dim=None):
        super(Index, self).__init__()
        self.D, self.I = None, None
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vecs):
        self.index.add(vecs)

    def cache(self, vecs, k):
        self.D, self.I = self.index.search(vecs, k)

    def topk(self, q, k):
        for j in range(k):
            idx = self.I[q][j]
            dist = self.D[q][j]
            yield idx, dist


class NearestNeighborsLookup(object):
    def __init__(self):
        super(NearestNeighborsLookup, self).__init__()


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

    # 1. Get all relevant phrase vectors.

    dtype = {
        'example_ids': 'list',
        'labels': 'list',
        'positions': 'list',
        'sizes': 'list',
        'phrases': 'list',
        'inside': 'torch',
        'outside': 'torch',
    }
    batch_recorder = BatchRecorder(dtype=dtype)

    ## Eval mode.
    trainer.net.eval()

    batches = validation_iterator.get_iterator(random_seed=options.seed)

    logger.info('Beginning to embed phrases.')

    with torch.no_grad():
        for i, batch_map in enumerate(batches):
            sentences = batch_map['sentences']
            batch_size = sentences.shape[0]
            length = sentences.shape[1]

            # Skips very short examples.
            if length <= 2:
                continue

            _ = trainer.step(batch_map, train=False, compute_loss=False)

            entity_labels = batch_map['entity_labels']
            batch_index, positions, sizes, labels = get_cell_index(entity_labels)

            # Skip short phrases.
            batch_index = [x for x, y in zip(batch_index, sizes) if y >= 2]
            positions = [x for x, y in zip(positions, sizes) if y >= 2]
            labels = [x for x, y in zip(labels, sizes) if y >= 2]
            sizes = [y for y in sizes if y >= 2]

            cell_index = (batch_index, positions, sizes)

            batch_result = {}
            batch_result['example_ids'] = [batch_map['example_ids'][idx] for idx in cell_index[0]]
            batch_result['labels'] = labels
            batch_result['positions'] = cell_index[1]
            batch_result['sizes'] = cell_index[2]
            batch_result['phrases'] = get_many_phrases(sentences, *cell_index)
            batch_result['inside'] = get_many_cells(diora, diora.inside_h, *cell_index)
            batch_result['outside'] = get_many_cells(diora, diora.outside_h, *cell_index)

            batch_recorder.record(**batch_result)

    result = batch_recorder.get_flattened_result()

    # 2. Build an index of nearest neighbors.

    vectors = np.concatenate([result['inside'], result['outside']], axis=1)
    normalize_L2(vectors)

    index = Index(dim=vectors.shape[1])
    index.add(vectors)
    index.cache(vectors, options.k_candidates)

    # 3. Print a summary.

    example_ids = result['example_ids']
    phrases = result['phrases']

    assert len(example_ids) == len(phrases)
    assert len(example_ids) == vectors.shape[0]

    def stringify(phrase):
        return ' '.join([idx2word[idx] for idx in phrase])

    for i in range(vectors.shape[0]):
        topk = []

        for j, score in index.topk(i, options.k_candidates):
            # Skip same example.
            if example_ids[i] == example_ids[j]:
                continue
            # Skip string match.
            if phrases[i] == phrases[j]:
                continue
            topk.append((j, score))
            if len(topk) == options.k_top:
                break
        assert len(topk) == options.k_top, 'Did not find enough valid candidates.'

        # Print.
        print('[query] example_id={} phrase={}'.format(
            example_ids[i], stringify(phrases[i])))
        for rank, (j, score) in enumerate(topk):
            print('rank={} score={:.3f} example_id={} phrase={}'.format(
                rank, score, example_ids[j], stringify(phrases[j])))


if __name__ == '__main__':
    parser = argument_parser()
    parser.add_argument('--k_candidates', default=100, type=int)
    parser.add_argument('--k_top', default=3, type=int)
    options = parse_args(parser)
    configure(options)

    run(options)
