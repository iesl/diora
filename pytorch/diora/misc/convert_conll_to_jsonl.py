import argparse
import os
import json


def pick(lst, k):
    return [d[k] for d in lst]


class ConllReader(object):
    def __init__(self, word_index=None, tag_index=None, delim=' '):
        super(ConllReader, self).__init__()
        self.word_index = word_index
        self.tag_index = tag_index
        self.delim = delim

        self.example_counter = None

    def reset(self):
        self.example_counter = 0

    def get_word(self, parts):
        return parts[self.word_index]

    def get_tag_and_labels(self, parts):
        x = parts[self.tag_index]

        def get_labels(y):
            return y

        if x.startswith('O'):
            return 'O', None
        if x.startswith('I'):
            return 'I', get_labels(x.split('-', 1)[1])
        if x.startswith('B'):
            return 'B', get_labels(x.split('-', 1)[1])

        raise ValueError('Not a BIO tag: {}'.format(x))

    def convert_records_to_example(self, records):
        example_id = self.example_counter

        word_lst = pick(records, 'word')
        tag_lst = pick(records, 'tag')
        labels_lst = pick(records, 'labels')

        entity_lst = []
        warning_lst = []

        for i, tag in enumerate(tag_lst):
            # Adjust tags if needed.
            if tag == 'I' and len(entity_lst) == 0:
                warning = '[warning] Converting I to B. I appears at beginning of sentence. i = {}'.format(i)
                warning_lst.append(warning)
                tag = 'B'

            if tag == 'I' and len(entity_lst) == 0:
                warning = '[warning] Converting I to B. I appears before any B tags. i = {}'.format(i)
                warning_lst.append(warning)
                tag = 'B'

            if tag == 'I' and len(entity_lst) > 0:
                pos = entity_lst[-1][-2]
                size = entity_lst[-1][-1]

                if pos + size != i:
                    warning = '[warning] Converting I to B. I appears after O. i = {}'.format(i)
                    warning_lst.append(warning)
                    tag = 'B'

            # Record entity.
            if tag == 'O':
                continue
            if tag == 'B':
                labels = labels_lst[i]

                # entity = (labels, position, size)
                entity = [labels, i, 1]
                entity_lst.append(entity)

                assert labels is not None and isinstance(labels, str)
            if tag == 'I':
                # increment size
                entity_lst[-1][-1] += 1

                pos = entity_lst[-1][-2]
                size = entity_lst[-1][-1]

                assert pos + size - 1 == i

        # Build Example
        example = {}
        example['example_id'] = '{}_{}'.format(options.name, example_id)
        example['entities'] = entity_lst
        example['sentence'] = word_lst

        if len(warning_lst) > 0:
            example['warnings'] = warning_lst

        # Cleanup
        self.example_counter += 1

        return example

    def _read(self, filename):

        lst = []

        with open(filename) as f:
            for i, line in enumerate(f):
                line = line.rstrip()

                # Skip Empty Lines
                if len(line) == 0:
                    if len(lst) > 0:
                        yield lst
                        lst = []
                    continue

                parts = line.split(self.delim)

                word = self.get_word(parts)
                tag, labels = self.get_tag_and_labels(parts)

                record = dict()
                record['word'] = word
                record['tag'] = tag
                record['labels'] = labels

                lst.append(record)

            # In case final does not end in newline.
            if len(lst) > 0:
                yield lst

    def read(self, filename):
        self.reset()

        for record_lst in self._read(filename):
            example = self.convert_records_to_example(record_lst)
            yield example


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./train.txt', type=str)
    parser.add_argument('--delim', default=' ', type=str)
    parser.add_argument('--i_word', default=0, type=int)
    parser.add_argument('--i_tag', default=2, type=int)
    parser.add_argument('--name', default='conll2000', type=str)
    options = parser.parse_args()

    reader = ConllReader(tag_index=options.i_tag, word_index=options.i_word, delim=options.delim)
    for example in reader.read(options.path):
        print(json.dumps(example))
