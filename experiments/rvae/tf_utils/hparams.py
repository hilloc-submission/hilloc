class HParams(object):

    def __init__(self, **kwargs):
        self._items = {}
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __setattr__(self, key, value):
        super(HParams, self).__setattr__(key, value)

        if key == '_items':
            return

        self._items[key] = value

    def copy(self):
        return HParams(**self._items)

    def __str__(self):
        return '\n'.join(map(lambda x: str(x[0]) + ': ' + str(x[1]), self._items.items()))

    def parse(self, str_value):
        hps = HParams(**self._items)
        for entry in str_value.strip().split(","):
            entry = entry.strip()
            if not entry:
                continue
            key, sep, value = entry.partition("=")
            if not sep:
                raise ValueError("Unable to parse: %s" % entry)
            default_value = hps._items[key]
            if isinstance(default_value, bool):
                hps.__setattr__(key, value.lower() == "true")
            elif isinstance(default_value, int):
                hps.__setattr__(key, int(value))
            elif isinstance(default_value, float):
                hps.__setattr__(key, float(value))
            else:
                hps.__setattr__(key, value)
        return hps