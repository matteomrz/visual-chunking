def trim_json_string(raw: str) -> str:
    """
    Trims a json string until a json object is left.
    Returns an empty string if no json object is present.
    """

    i = 0
    while raw[i] != "{":
        i += 1

    j = len(raw)
    while raw[j - 1] != "}" and j > i:
        j -= 1

    return raw[i:j]
