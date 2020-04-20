from tqdm import tqdm

enabled = True
mininterval = 0.1
postfix_enabled = True
postfix_refresh = True


class ProgressBar:
    def __init__(self, iterable, desc=None, total=None, unit='it', disable=False):
        if enabled:
            self.t = tqdm(iterable, desc=desc, total=total, unit=unit, disable=disable, mininterval=mininterval)
        else:
            self.t = iterable

    def __iter__(self):
        return iter(self.t)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if enabled:
            self.t.close()

    def update(self):
        if enabled:
            self.t.update()

    def set_postfix(self, ordered_dict=None):
        if enabled and postfix_enabled:
            self.t.set_postfix(ordered_dict, refresh=postfix_refresh)

    def set_postfix_str(self, s):
        if enabled and postfix_enabled:
            self.t.set_postfix_str(s, refresh=postfix_refresh)
