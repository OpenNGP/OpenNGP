from ast import arg


def fun(a, c, **args):
    print(a + c)
    print(args)

args = {"a": 1, "b":2, "c":3}

fun(15, **args)

