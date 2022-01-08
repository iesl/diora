import collections
import json
import nltk
import os


word_tags = set(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
               'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
               'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
               'WDT', 'WP', 'WP$', 'WRB'])


def get_ptb_format_from_nltk_tree(tr):
    if len(tr) == 1 and isinstance(tr[0], str):
        return f'({tr.label()} {tr[0]})'

    nodes = [get_ptb_format_from_nltk_tree(x) for x in tr]

    return f'({tr.label()} {" ".join(nodes)})'


def get_ptb_format_from_diora_tree(parse, tokens, return_string=False, batched=False):
    if batched:
        return [get_ptb_format(p, t, return_string, batched=False) for p, t in zip(parse, tokens)]

    def recursive_add_tokens(parse):
        def helper(tr, pos):
            if not isinstance(tr, (tuple, list)):
                return 1, tokens[pos]

            size, nodes = 0, []
            for x in tr:
                xsize, xnode = helper(x, pos + size)
                size += xsize
                nodes.append(xnode)

            return size, tuple(nodes)

        _, new_parse = helper(parse, 0)

        return new_parse

    def recursive_string(parse):
        if isinstance(parse, str):
            return f'(DT {parse})'
        return '(S ' + ' '.join([recursive_string(p) for p in parse]) + ')'

    parse = recursive_add_tokens(parse)
    if return_string:
        parse = recursive_string(parse)
    return parse


def remove_punctuation_from_tree(tree, ref):
    def recursive_remove_using_mask(tr, position, mask):
        if len(tr) == 1 and isinstance(tr[0], str):
            size = 1
            keep = mask[position]
            return size, keep

        size = 0
        keep = []
        for x in tr:
            xsize, xkeep = recursive_remove_using_mask(x, position + size, mask)
            size += xsize
            keep.append(xkeep)

        for i, xkeep in list(enumerate(keep))[::-1]:
            if not xkeep:
                del tr[i]

        keep = any(keep)
        return size, keep

    tokens = tree.leaves()
    part_of_speech = [x[1] for x in ref.pos()]
    mask = [x in word_tags for x in part_of_speech] # Tokens that are punctuation are given False in mask.
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    assert len(new_tokens) > 0

    recursive_remove_using_mask(tree, 0, mask)
    assert len(tree.leaves()) == len(new_tokens), (tree.leaves(), new_tokens, tokens, mask)
    assert tuple(tree.leaves()) == tuple(new_tokens)

    return tree


def main(args):
    os.system(f'mkdir -p {args.out}')

    # Read DIORA data.
    pred = []
    with open(args.pred) as f:
        for line in f:
            tree = json.loads(line)['tree']
            pred.append(tree)

    # Read ground truth parse trees.
    gold = []
    with open(args.gold) as f:
        for line in f:
            nltk_tree = nltk.Tree.fromstring(line)
            gold.append(nltk_tree)

    assert len(gold) == len(pred), f"The gold and pred files must have same number of sentences. {len(pred)} != {len(gold)}"

    pred = [nltk.Tree.fromstring(get_ptb_format_from_diora_tree(p, g.leaves(), return_string=True))
            for g, p in zip(gold, pred)]

    # Remove punctuation.
    pred = [remove_punctuation_from_tree(p, ref=g) for g, p in zip(gold, pred)]
    gold = [remove_punctuation_from_tree(g, ref=g) for g, p in zip(gold, pred)]

    # Remove sentences according to length.
    assert all(len(x.leaves()) > 0 for x in pred)
    assert all(len(x.leaves()) > 0 for x in gold)
    assert all(len(p.leaves()) == len(g.leaves()) for g, p in zip(gold, pred))

    # Serialize as strings.
    pred = [get_ptb_format_from_nltk_tree(p) for g, p in zip(gold, pred)]
    gold = [get_ptb_format_from_nltk_tree(g) for g, p in zip(gold, pred)]

    # Write new intermediate files.
    new_pred_file = os.path.join(args.out, 'pred.txt')
    new_gold_file = os.path.join(args.out, 'gold.txt')

    with open(new_pred_file, 'w') as f:
        for parse in pred:
            f.write(parse + '\n')

    with open(new_gold_file, 'w') as f:
        for parse in gold:
            f.write(parse + '\n')

    # Run EVALB.
    evalb_out_file = os.path.join(args.out, 'evalb.out')

    evalb_command = '{evalb} -p {evalb_config} {gold} {pred} > {out}'.format(
        evalb=os.path.join(args.evalb, 'evalb'),
        evalb_config=args.evalb_config,
        gold=new_gold_file,
        pred=new_pred_file,
        out=evalb_out_file)

    print(f'\nRunning: {evalb_command}')
    os.system(evalb_command)

    print(f'\nResults are ready at: {evalb_out_file}')

    print(f'\n==== PREVIEW OF RESULTS ({evalb_out_file}) ====\n')
    os.system(f'tail -n 27 {evalb_out_file}')
    print('')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='File with DIORA predictions from `parse.py`.')
    parser.add_argument('--gold', type=str, required=True, help='File with ground truth parse trees in PTB format.')
    parser.add_argument('--out', type=str, required=True, help='Directory to write intermediates files for EVALB and results.')
    parser.add_argument('--evalb', type=str, required=True, help='Path to EVALB directory.')
    parser.add_argument('--evalb_config', type=str, required=True, help='Path to EVALB configuration file.')
    parser.add_argument('--max_length', default=None, type=str, help='Max length after removing punctuation.')
    args = parser.parse_args()
    main(args)
