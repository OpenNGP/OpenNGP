import inspect
from python_api.utils.function_registry import FunctionRegistry


def gen_clazz_initializer(clazz):
    sig = inspect.signature(clazz.__init__)

    def get_karg(parameter, **kwargs):
        kwarg = kwargs.get(parameter.name, parameter.default)
        if kwarg is parameter.empty:
            raise TypeError(f'missing argument {parameter.name}')
        return kwarg

    def clazz_initializer(**kwargs):
        # get all params except self
        params = [p for p in sig.parameters.items() if p[0] != 'self']
        # extract value for params from kwargs
        fed_kwargs = {p[0]: get_karg(p[1], **kwargs) for p in params}
        return clazz(**fed_kwargs)
    return clazz_initializer


class ClassFactory(FunctionRegistry):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.register(k, v)

    def register(self, name: str, clazz):
        self[name] = gen_clazz_initializer(clazz)

    def build(self, name: str, **config):
        return self[name](**config)
