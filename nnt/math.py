import math


def round_to_multiple(number, multiple, up=False, down=False):
    """
    Round `number` to the nearest multiple of `multiple`.
    If `up` is true, round up. If `down` is true, round down. Otherwise, round normally.
    """
    assert not (up and down), "only one of up/down may be true"
    if up:
        v = math.ceil(number / multiple)
    elif down:
        v = math.floor(number / multiple)
    else:
        v = round(number / multiple)
    result = multiple * v
    if isinstance(number, int):
        return int(result)
    else:
        return float(result)


class EMA:
    """Exponential moving average with debias"""

    def __init__(self, smoothing=0.999):
        self._val = 0.0
        self._last_val = None
        self._num_vals = 0
        self._smoothing = smoothing

    def put(self, val):
        val = float(val)
        self._last_val = val
        self._val = self._val * self._smoothing + (1 - self._smoothing) * val
        self._num_vals += 1
        return self.get()

    def get(self):
        debias = 1.0 - self._smoothing**self._num_vals
        return self._val / debias

    def last(self):
        return self._last_val
