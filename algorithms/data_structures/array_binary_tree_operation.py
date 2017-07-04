def root():
    return 1


def is_left_son(number):
    assert number > root()
    return not (number & 1)


def is_right_son(number):
    assert number > root()
    return bool(number & 1)


def is_in_tree(number):
    return (number >= root())


def left_son(number):
    return (number << 1)


def right_son(number):
    return ((number << 1) | 1)


def left_sibling(number):
    assert is_right_son(number)
    return (number - 1)


def right_sibling(number):
    assert is_left_son(number)
    return (number | 1)


def parent(number):
    return (number >> 1)
