from python_api.utils import FunctionRegistry


def uniform_sampler():
    pass


def importance_sampler():
    pass


raymarcher = FunctionRegistry(
    uniform_sampler=uniform_sampler,
    importance_sampler=importance_sampler,
)
