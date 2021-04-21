def strip_underscores(text, position="all"):
    """Strips leading and/or trailing underscores from a string.

    Args:
        text (str): The input string.
        position (str, optional): From which position underscores should be removed.
            Options are 'leading', 'trailing' & 'both'. Defaults to "both".

    Returns:
        str: String without the underscores.
    """
    if position.lower() == "leading":
        while text.startswith("_"):
            text = text[1:]
    elif position.lower() == "trailing":
        while text.endswith("_"):
            text = text[:-1]
    else:
        text = text.strip("_")
    return text


test = "__test"
print(test)
print(remove_leading_underscores(test))
test2 = test.strip("__")
print(test2)
print("script ended")
