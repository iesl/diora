import math
import random
from collections import Counter, OrderedDict

import numpy as np
from tqdm import tqdm


DEFAULT_UNK_INDEX = 1


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_text_vocab(sentences, word2idx=None):
    word2idx = OrderedDict() if word2idx is None else word2idx.copy()
    for s in sentences:
        for w in s:
            if w not in word2idx:
                word2idx[w] = len(word2idx)
    return word2idx


def indexify(sentences, word2idx, unk_index=None):
    def fn(s):
        for w in s:
            if w not in word2idx and unk_index is None:
                raise ValueError
            yield word2idx.get(w, unk_index)
    return [list(fn(s)) for s in tqdm(sentences, desc='indexify')]


def batchify(examples, batch_size):
    sorted_examples = list(sorted(examples, key=lambda x: len(x)))
    num_batches = int(math.ceil(len(examples) / batch_size))
    batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch = sorted_examples[start:end]
        batches.append(pad(batch))

    return batches


def pad(examples, padding_token=0):
    def convert2numpy(batch):
        # Note that this is tranposed to have dimensions (batch_size, sentence_length).
        return np.array(batch, dtype=np.int32).T

    maxlength = np.max([len(x) for x in examples])
    batch = []

    for x in examples:
        diff = maxlength - len(x)
        padded = [0] * diff + x
        batch.append(padded)

    return convert2numpy(batch)


def batch_iterator(dataset, batch_size, seed=None, drop_last=False):
    if seed is not None:
        set_random_seed(seed)

    nexamples = len(dataset)
    nbatches = math.ceil(nexamples/batch_size)
    index = random.sample(range(nexamples), nexamples)

    for i in range(nbatches):
        start = i * batch_size
        end = start + batch_size
        if end > nexamples and drop_last:
            break

        batch = [dataset[i] for i in index[start:end]]
        yield batch


def prepare_batch(batch):
    return batch


def synthesize_training_data(nexamples, vocab_size, min_length=10, max_length=30, seed=None):
    if seed is not None:
        set_random_seed(seed)

    dataset = []

    for i in range(nexamples):
        length = np.random.randint(min_length, max_length)
        example = np.random.randint(0, vocab_size, size=length).tolist()
        dataset.append(example)

    return dataset
