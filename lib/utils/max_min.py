def get_max_min(entries: list[tuple[str, int]]) -> dict[str, tuple[int, int]]:
    """
    Finds the lowest and highest integers associated with same string.

    Args:
        entries: List of Tuples of the structure: (``string``, ``integer``)

    Returns:
        Dictionary mapping each string to a tuple of (min_int, max_int)
    """

    min_max = {}

    for str_key, int_val in entries:
        min_idx, max_idx = min_max.get(str_key, (int_val, int_val))

        if int_val <= min_idx:
            min_max[str_key] = (int_val, max_idx)
        if int_val >= max_idx:
            min_max[str_key] = (min_idx, int_val)

    return min_max
