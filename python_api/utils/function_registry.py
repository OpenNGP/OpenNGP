import inspect
from collections import Mapping


class FunctionRegistry(Mapping):
    """
    Non-overwritable mapping of string keys to functions.
    """

    def __init__(self, **kwargs):
        self._dict = {}
        self._requires = {}
        for k, v in kwargs.items():
            self[k] = v

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError('key must be a string, got %s' % str(key))
        if key in self:
            raise KeyError('Cannot set new value to existing key %s' % key)
        if not callable(value):
            raise ValueError('Cannot set value which is not callable.')
        self._dict[key] = value
        self._requires[key] = list(inspect.signature(value).parameters.keys())

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def __call__(self, key, *args, **kwargs):
        return self[key](*args, **kwargs)

    def parameters(self, key):
        return self._requires[key]
