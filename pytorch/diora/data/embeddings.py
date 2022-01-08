import os
import hashlib
from collections import OrderedDict

from diora.logging.configuration import get_logger

import numpy as np
import torch
from diora.external.standalone_elmo import batch_to_ids, ElmoCharacterEncoder, remove_sentence_boundaries
from tqdm import tqdm


# With loaded embedding matrix, the padding vector will be initialized to zero
# and will not be trained. Hopefully this isn't a problem. It seems better than
# random initialization...
PADDING_TOKEN = "_PAD"

UNK_TOKEN = "_"

EXISTING_VOCAB_TOKEN = "unused-token-a7g39i"


def maybe_download(remote_url, cache_dir):
    path = os.path.join(cache_dir, os.path.basename(remote_url))
    if not os.path.exists(path):
        os.system(f'curl {remote_url} -o {path} -L')
    return path


class ElmoEmbedder(object):
    def __init__(self, options_file, weights_file, cache_dir, cuda=False):
        logger = get_logger()
        logger.info('Initialize ELMo Model.')

        self.char_embedder = ElmoCharacterEncoder(
            options_file=maybe_download(options_file, cache_dir=cache_dir),
            weight_file=maybe_download(weights_file, cache_dir=cache_dir),
            requires_grad=False)

        if cuda:
            self.char_embedder.cuda()

        self.cuda = cuda
        self.cache_dir = cache_dir

    def __call__(self, word2idx):
        """
        1. Sort tokens alphabetically by `word` from `word2idx`.
        2. Embed the newly sorted tokens.
        3. Re-order embeddings according to `idx` from `word2idx`.

        Will skip step (2) if there is a previously cached version of embeddings.

        """

        logger = get_logger()

        def sort_by_tok(item):
            tok, idx = item
            return tok

        def sort_by_idx(item):
            tok, idx = item
            return idx

        size = 512
        batch_size = 1024

        # 1. Sort tokens alphabetically by `word` from `word2idx`.
        tokens = [tok for tok, idx in sorted(word2idx.items(), key=sort_by_tok)]

        # 2. Embed the newly sorted tokens.
        vocab_identifier = hash_tokens(tokens)
        embeddings_file = os.path.join(self.cache_dir, f'elmo_{vocab_identifier}.npy')
        shape = (len(tokens), size)
        if os.path.exists(embeddings_file):
            logger.info('Loading cached elmo vectors: {}'.format(embeddings_file))
            embeddings = np.load(embeddings_file)
            assert embeddings.shape == shape

        else:
            logger.info('Begin caching vectors. shape = {}, cuda = {}'.format(shape, self.cuda))

            embeddings = np.zeros(shape, dtype=np.float32)

            for start in tqdm(range(0, len(tokens), batch_size), desc='embed'):
                end = min(start + batch_size, len(tokens))
                batch = tokens[start:end]
                batch_ids = batch_to_ids([[x] for x in batch])
                if self.cuda:
                    batch_ids = batch_ids.cuda()
                output = self.char_embedder(batch_ids)
                vec = remove_sentence_boundaries(output['token_embedding'], output['mask'])[0].squeeze(1)

                embeddings[start:end] = vec.cpu().numpy()

            # Cache embeddings.
            logger.info('Saving cached elmo vectors: {}'.format(embeddings_file))
            np.save(embeddings_file, embeddings)

        # 3. Re-order embeddings according to `idx` from `word2idx`.
        sorted_word2idx = {tok: idx for idx, tok in enumerate(tokens)}
        index = [sorted_word2idx[tok] for tok, idx in sorted(word2idx.items(), key=sort_by_idx)]
        old_embeddings = embeddings
        embeddings = embeddings[index]

        # Duplicate embeddings. This is meant to mirror behavior in elmo, which has separate embeddings
        # for forward and backward LSTMs.
        embeddings = np.concatenate([embeddings, embeddings], 1)

        return embeddings


class EmbeddingsReader(object):

    def read_glove(self, *args, **kwargs):
        return read_glove(*args, **kwargs)

    def get_emb_w2v(self, options, embeddings_path, word2idx):
        embeddings, word2idx = self.read_glove(embeddings_path, word2idx)
        return embeddings, word2idx

    def get_emb_elmo(self, options, embeddings_path, word2idx):
        elmo_encoder = ElmoEmbedder(
            options_file=options.elmo_options_path,
            weights_file=options.elmo_weights_path,
            cache_dir=options.elmo_cache_dir,
            cuda=options.cuda)
        embeddings = elmo_encoder(word2idx)
        # embeddings = self.context_insensitive_elmo(weights_path=weights_path, options_path=options_path,
            # word2idx=word2idx, cuda=options.cuda, cache_dir=options.elmo_cache_dir)
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
    """
    Verify that all `idx` are accounted for.
    """
    idx_set = set(word2idx.values())
    for i in range(len(word2idx)):
        assert i in word2idx


def validate_word_order(tokens):
    """
    Verify tokens are in sorted order.
    """
    for w0, w1 in zip(tokens, sorted(tokens)):
        assert w0 == w1


def hash_tokens(tokens):
    validate_word_order(tokens)

    m = hashlib.sha256()
    for w in tokens:
        m.update(str.encode(w))
    return m.hexdigest()
