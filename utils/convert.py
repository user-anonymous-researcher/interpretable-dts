def convert_to_str(value) -> str:
    if type(value) == int:
        return "{:02d}".format(value)
    elif type(value) == float:
        return str(round(float(value), 2))
    return str(value)
