import torch
from diora.net.offset_cache import get_offset_cache


class InsideIndex(object):
    def get_pairs(self, level, i):
        pairs = []
        for constituent_num in range(0, level):
            l_level = constituent_num
            l_i = i - level + constituent_num
            r_level = level - 1 - constituent_num
            r_i = i
            pair = ((l_level, l_i), (r_level, r_i))
            pairs.append(pair)
        return pairs

    def get_all_pairs(self, level, n):
        pairs = []
        for i in range(level, n):
            pairs += self.get_pairs(level, i)
        return pairs


class InsideIndexCheck(object):
    def __init__(self, length, spans, siblings):
        sib_map = {}
        for x, y, n in siblings:
            sib_map[x] = (y, n)
            sib_map[y] = (x, n)

        check = {}
        for sibling, (target, name) in sib_map.items():
            xlength = target[1] - target[0]
            xlevel = xlength - 1
            xpos = target[0]
            tgt = (xlevel, xpos)

            slength = sibling[1] - sibling[0]
            slevel = slength - 1
            spos = sibling[0]
            sis = (slevel, spos)

            check[(tgt, sis)] = True
        self.check = check

    def is_valid(self, tgt, sis):
        return (tgt, sis) in self.check


def get_inside_index(length, level, offset_cache=None, cuda=False):
    if offset_cache is None:
        offset_cache = get_offset_cache(length)
    index = InsideIndex()
    pairs = index.get_all_pairs(level, length)

    L = length - level
    n_constituents = len(pairs) // L
    idx_l, idx_r = [], []

    for i in range(n_constituents):
        index_l, index_r = [], []

        lvl_l = i
        lvl_r = level - i - 1
        lstart, lend = 0, L
        rstart, rend = length - L - lvl_r, length - lvl_r

        if lvl_l < 0:
            lvl_l = length + lvl_l
        if lvl_r < 0:
            lvl_r = length + lvl_r

        for pos in range(lstart, lend):
            offset = offset_cache[lvl_l]
            idx = offset + pos
            index_l.append(idx)

        for pos in range(rstart, rend):
            offset = offset_cache[lvl_r]
            idx = offset + pos
            index_r.append(idx)

        idx_l.append(index_l)
        idx_r.append(index_r)

    device = torch.cuda.current_device() if cuda else None
    idx_l = torch.tensor(idx_l, dtype=torch.int64, device=device
            ).transpose(0, 1).contiguous().flatten()
    idx_r = torch.tensor(idx_r, dtype=torch.int64, device=device
            ).transpose(0, 1).contiguous().flatten()

    return idx_l, idx_r

