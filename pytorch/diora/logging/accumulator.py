class MeanAccumulator(object):
    def __init__(self):
        self.reset()

    def record(self, val):
        if self.count == 0:
            self.count += 1
            self.val += val
            return

        count = self.count + 1
        self.val = ((self.val * self.count) + val) / count
        self.count = count

    def reset(self):
        self.val = 0
        self.count = 0


class Accumulator(object):
    def __init__(self):
        self.table = {}

    def record(self, key, val):
        if not key in self.table:
            self.table[key] = MeanAccumulator()
        self.table[key].record(val)

    def has(self, key):
        return key in self.table

    def get_mean(self, key, default=0):
        if key in self.table:
            val = self.table[key].val
        else:
            val = default
        return val

    def reset(self, key=None):
        if key is None:
            keys = list(self.table.keys())
        else:
            keys = [key]

        for key in keys:
            del self.table[key]

