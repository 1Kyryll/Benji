# Mutual recursion with no metered call anywhere. Resolution must terminate.
def ping(n):
    return pong(n - 1)


def pong(n):
    return ping(n - 1)
