def get_offset_cache(length):
    offset_cache = {}
    ncells = int(length * (1 + length) / 2)
    for lvl in range(length):
        level_length = length - lvl
        ncells_less = int(level_length * (1 + level_length) / 2)
        offset_cache[lvl] = ncells - ncells_less
    return offset_cache