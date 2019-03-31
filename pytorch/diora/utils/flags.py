import os
import json
import sys


def read_flags(fn):
    with open(fn) as f:
        flags = json.loads(f.read())
    return flags


def override_with_flags(options, flags, flags_to_use=None):
    """
    If `flags_to_use` is None, then override all flags, otherwise,
    only consider flags from `flags_to_use`.

    """
    if flags_to_use is None:
        for k, v in flags.items():
            setattr(options, k, v)
    else:
        for k in flags_to_use:
            setattr(options, k, flags.get(k))
    return options


def init_with_flags_file(options, flags_file, flags_to_use=None):
    flags = read_flags(flags_file)
    options = override_with_flags(options, flags, flags_to_use)
    return options


def stringify_flags(options):
    # Ignore negative boolean flags.
    flags = {k: v for k, v in options.__dict__.items()}
    return json.dumps(flags, indent=4, sort_keys=True)


def save_flags(options, experiment_path):
    flags = stringify_flags(options)
    target_file = os.path.join(experiment_path, 'flags.json')
    with open(target_file, 'w') as f:
        f.write(flags)
