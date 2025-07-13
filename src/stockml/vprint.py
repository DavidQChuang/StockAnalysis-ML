def vprint(verbosity, min_verbosity, *args, **kwargs):
    if verbosity >= min_verbosity:
        print(*args, **kwargs)
