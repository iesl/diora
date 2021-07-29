import os
import hashlib
from collections import OrderedDict

from diora.logging.configuration import get_logger

import numpy as np
#from allennlp.commands.elmo import ElmoEmbedder
from tqdm import tqdm


# With loaded embedding matrix, the padding vector will be initialized to zero
# and will not be trained. Hopefully this isn't a problem. It seems better than
# random initialization...
PADDING_TOKEN = "_PAD"

UNK_TOKEN = "_"

EXISTING_VOCAB_TOKEN = "unused-token-a7g39i"


class EmbeddingsReader(object):
    def context_insensitive_elmo(self, *args, **kwargs):
        return context_insensitive_elmo(*args, **kwargs)

    def read_glove(self, *args, **kwargs):
        return read_glove(*args, **kwargs)

    def get_emb_w2v(self, options, embeddings_path, word2idx):
        embeddings, word2idx = self.read_glove(embeddings_path, word2idx)
        return embeddings, word2idx

    def get_emb_elmo(self, options, embeddings_path, word2idx):
        options_path = options.elmo_options_path
        weights_path = options.elmo_weights_path
        embeddings = self.context_insensitive_elmo(weights_path=weights_path, options_path=options_path,
            word2idx=word2idx, cuda=options.cuda, cache_dir=options.elmo_cache_dir)
        return embeddings, word2idx

    def get_emb_both(self, options, embeddings_path, word2idx):
        e_w2v, w2i_w2v = self.get_emb_w2v(options, embeddings_path, word2idx)
        e_elmo, w2i_elmo = self.get_emb_elmo(options, embeddings_path, word2idx)

        vec_size = e_w2v.shape[1] + e_elmo.shape[1]
        vocab = [w for w, i in sorted(w2i_w2v.items(), key=lambda x: x[1]) if w in w2i_elmo]
        vocab_size = len(vocab)

        embeddings = np.zeros((vocab_size, vec_size), dtype=np.float32)
        word2idx = {w: i for i, w in enumerate(vocab)}

        for w, i in word2idx.items():
            embeddings[i, :e_w2v.shape[1]] = e_w2v[w2i_w2v[w]]
            embeddings[i, e_w2v.shape[1]:] = e_elmo[w2i_elmo[w]]

        return embeddings, word2idx

    def get_embeddings(self, options, embeddings_path, word2idx):
        if options.emb == 'w2v':
            out = self.get_emb_w2v(options, embeddings_path, word2idx)
        elif options.emb == 'elmo':
            out = self.get_emb_elmo(options, embeddings_path, word2idx)
        elif options.emb == 'both':
            out = self.get_emb_both(options, embeddings_path, word2idx)
        return out


def read_glove(filename, word2idx):
    """
    Two cases:

    1. The word2idx has already been filtered according to embedding vocabulary.
    2. The word2idx is derived solely from the raw text data.

    """
    logger = get_logger()

    glove_vocab = set()
    size = None

    validate_word2idx(word2idx)

    logger.info('Reading Glove Vocab.')

    with open(filename) as f:
        for i, line in enumerate(f):
            word, vec = line.split(' ', 1)
            glove_vocab.add(word)

            if i == 0:
                size = len(vec.strip().split(' '))

    new_vocab = set.intersection(set(word2idx.keys()), glove_vocab)
    new_vocab.discard(PADDING_TOKEN)
    new_vocab.discard(UNK_TOKEN)

    if word2idx.get(EXISTING_VOCAB_TOKEN, None) == 2:
        new_word2idx = word2idx.copy()

        logger.info('Using existing vocab mapping.')
    else:
        new_word2idx = OrderedDict()
        new_word2idx[PADDING_TOKEN] = len(new_word2idx)
        new_word2idx[UNK_TOKEN] = len(new_word2idx)
        new_word2idx[EXISTING_VOCAB_TOKEN] = len(new_word2idx)

        for w, _ in word2idx.items():
            if w in new_word2idx:
                continue
            new_word2idx[w] = len(new_word2idx)

        logger.info('Creating new mapping.')

    logger.info('glove-vocab-size={} vocab-size={} intersection-size={} (-{})'.format(
        len(glove_vocab), len(word2idx), len(new_vocab), len(word2idx) - len(new_vocab)))

    embeddings = np.zeros((len(new_word2idx), size), dtype=np.float32)

    logger.info('Reading Glove Embeddings.')

    with open(filename) as f:
        for line in f:
            word, vec = line.strip().split(' ', 1)

            if word is PADDING_TOKEN or word is UNK_TOKEN:
                continue

            if word in new_vocab and word not in new_word2idx:
                raise ValueError

            if word not in new_word2idx:
                continue

            word_id = new_word2idx[word]
            vec = np.fromstring(vec, dtype=float, sep=' ')
            embeddings[word_id] = vec

    validate_word2idx(new_word2idx)

    return embeddings, new_word2idx


def validate_word2idx(word2idx):
    vocab = [w for w, i in sorted(word2idx.items(), key=lambda x: x[1])]
    for i, w in enumerate(vocab):
        assert word2idx[w] == i


def hash_vocab(vocab):
    m = hashlib.sha256()
    for w in sorted(vocab):
        m.update(str.encode(w))
    return m.hexdigest()


def save_elmo_cache(path, vectors):
    np.save(path, vectors)


def load_elmo_cache(path):
    return np.load(path)


#def context_insensitive_elmo(weights_path, options_path, word2idx, cuda=False, cache_dir=None):
#    logger = get_logger()
#
#    vocab = [w for w, i in sorted(word2idx.items(), key=lambda x: x[1])]
#
#    validate_word2idx(word2idx)
#
#    if cache_dir is not None:
#        key = hash_vocab(vocab)
#        cache_path = os.path.join(cache_dir, 'elmo_{}.npy'.format(key))
#
#        if os.path.exists(cache_path):
#            logger.info('Loading cached elmo vectors: {}'.format(cache_path))
#            return load_elmo_cache(cache_path)
#
#    if cuda:
#        device = 0
#    else:
#        device = -1
#
#    batch_size = 256
#    nbatches = len(vocab) // batch_size + 1
#
#    logger.info('Begin caching vectors. nbatches={} device={}'.format(nbatches, device))
#    logger.info('Initialize ELMo Model.')
#
#    # TODO: Does not support padding.
#    elmo = ElmoEmbedder(options_file=options_path, weight_file=weights_path, cuda_device=device)
#    vec_lst = []
#    for i in tqdm(range(nbatches), desc='elmo'):
#        start = i * batch_size
#        batch = vocab[start:start+batch_size]
#        if len(batch) == 0:
#            continue
#        vec = elmo.embed_sentence(batch)
#        vec_lst.append(vec)
#
#    vectors = np.concatenate([x[0] for x in vec_lst], axis=0)
#
#    if cache_dir is not None:
#        logger.info('Saving cached elmo vectors: {}'.format(cache_path))
#        save_elmo_cache(cache_path, vectors)
#
#    return vectors


