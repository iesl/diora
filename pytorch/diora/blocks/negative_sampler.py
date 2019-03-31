from collections import Counter

import numpy as np
import torch

from tqdm import tqdm


def choose_negative_samples(negative_sampler, k_neg):
    neg_samples = negative_sampler.sample(k_neg)
    neg_samples = torch.from_numpy(neg_samples)
    return neg_samples


def calculate_freq_dist(data, vocab_size):
    # TODO: This becomes really slow on large datasets.
    counter = Counter()
    for i in range(vocab_size):
        counter[i] = 0
    for x in tqdm(data, desc='freq_dist'):
        counter.update(x)
    freq_dist = [v for k, v in sorted(counter.items(), key=lambda x: x[0])]
    freq_dist = np.asarray(freq_dist, dtype=np.float32)
    return freq_dist


class NegativeSampler:
    def __init__(self, freq_dist, dist_power, epsilon=10**-2):
        self.dist = freq_dist ** dist_power + epsilon * (1/len(freq_dist))
        self.dist = self.dist / sum(self.dist)      # Final distribution should be normalized
        self.rng = np.random.RandomState()

    def set_seed(self, seed):
        self.rng.seed(seed)

    def sample(self, num_samples):
        return self.rng.choice(len(self.dist), num_samples, p=self.dist, replace=False)
