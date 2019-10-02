import os
import json

import nltk


def nltk_tree_to_tuples(tree):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return tr[0]
        if len(tr) == 1:
            return helper(tr[0])
        return tuple([helper(x) for x in tr])
    return helper(tree)


def tuples_to_spans(tree):
    """
    Returns list of spans, that are (start, size).
    """

    result = []

    def helper(tr, pos=0):
        if isinstance(tr, str):
            return 1
        size = 0
        for x in tr:
            subsize = helper(x, pos=pos+size)
            size += subsize
        result.append((pos, size))
        return size

    helper(tree)

    return result

class Tree(object):
    def __init__(self):
        pass

    @classmethod
    def build_from_berkeley(cls, ex):
        result = cls()
        result.tree = nltk_tree_to_tuples(nltk.Tree.fromstring(ex['tree'].strip()))
        result.spans = tuples_to_spans(result.tree)
        result.example_id = ex.get('example_id', ex.get('exampled_id', None))
        assert result.example_id is not None, "Oops! No example id."
        return result

    @classmethod
    def build_from_diora(cls, ex):
        result = cls()
        return result



def read_supervised(path):
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            try:
                yield Tree.build_from_berkeley(ex)
            except:
                print('Skipping {}'.format(ex))


def main(options):

    cache = {}

    # Read the supervised.
    cache['supervised'] = {}
    for x in read_supervised(options.supervised):
        cache['supervised'][x.example_id] = x

    # Read the unsupervised.
    # TODO: Will write it for binary trees from diora.

    # Measure the closeness of unsupervised to supervised.
    # TODO: Compute the recall per sentence, and average.


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervised', default=None, type=str)
    parser.add_argument('--unsupervised', default=None, type=str)
    options = parser.parse_args()
    main(options)
