# Alternatives:
# scikit-optimize https://neptune.ai/blog/scikit-optimize

def suggest(trial, x):
    try:
        return x.suggest(trial)
    except AttributeError:
        return {k: suggest(trial, v) for k, v in x.items()}


def default(x):
    try:
        return x.default
    except AttributeError:
        return {k: default(v) for k, v in x.items()}


class Param:
    def __init__(self, name, default=None, searchable=True):
        self.name = name
        self.default = default
        self.searchable = searchable

    def suggest(self, trial):
        if self.searchable:
            return self._suggest(trial)
        else:
            return self.default

    def _suggest(self, trial):
        raise NotImplementedError


class Categorical(Param):
    def __init__(self, name, choices, **kwargs):
        super().__init__(name, **kwargs)
        self.choices = choices

    def _suggest(self, trial):
        return trial.suggest_categorical(self.name, self.choices)


class Boolean(Categorical):
    def __init__(self, name, **kwargs):
        super().__init__(name, [False, True], **kwargs)


class Numeric(Param):
    def __init__(self, name, low, high, step=None, log=False, **kwargs):
        super().__init__(name, **kwargs)
        self.low = low
        self.high = high
        self.step = step
        self.log = log


class Int(Numeric):
    def __init__(self, name, low, high, step=1, log=False, **kwargs):
        super().__init__(name, low, high, step=step, log=log, **kwargs)

    def _suggest(self, trial):
        return trial.suggest_int(self.name, self.low, self.high, step=self.step, log=self.log)


class Float(Numeric):
    def __init__(self, name, low, high, step=None, log=False, **kwargs):
        super().__init__(name, low, high, step=step, log=log, **kwargs)

    def _suggest(self, trial):
        return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)
