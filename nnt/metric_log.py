import datetime
import logging
import time
from collections import defaultdict
from typing import Optional

import torch

from .math import EMA


class MetricWriter:
    """
    Base class for writers that obtain values from :class:`MetricLogger` and process them.
    """

    def write_scalar(self, step: int, name: str, value: float):
        pass

    def write_histogram(self, step: int, name: str, hist_tensor: torch.Tensor):
        pass

    def step(self, iter):
        pass

    def close(self):
        pass


class TensorboardWriter(MetricWriter):
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
        """
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)

    def write_scalar(self, iter, name, value):
        self._writer.add_scalar(name, value, iter)

    def write_histogram(self, iter, name, hist_tensor):
        self._writer.add_histogram(name, hist_tensor.flatten().to(torch.float), iter)

    def close(self):
        self._writer.close()


class WandbWriter(MetricWriter):
    def __init__(self):
        """
        Log scalars and histograms to wandb.

        Must call wandb.init first, and optionally wandb.watch.
        """
        import wandb

        self.wandb = wandb

    def write_scalar(self, step, name, value):
        self.wandb.log({name: value}, step)

    def write_histogram(self, step, name, hist_tensor):
        hist = self.wandb.Histogram(hist_tensor.flatten().numpy())
        self.wandb.log({name: hist}, step)


class MetricPrinter(MetricWriter):
    """
    Print common metrics to the terminal, including iteration time, ETA, memory,
    all losses, and the learning rate. It also applies smoothing using a window of 20 elements.
    """

    def __init__(
        self,
        print_interval=10,
        max_iter: Optional[int] = None,
        iter_time_key="time/iter",
        data_time_key="time/data",
        lr_key="lr",
    ):
        """
        Args:
            max_iter: the maximum number of iterations to train.
                Used to compute ETA. If not given, ETA will not be printed.
            window_size (int): the losses will be median-smoothed by this window size
        """
        self._max_iter = max_iter
        self._last_write = None  # (step, time) of last call to write(). Used to compute ETA
        self._iter_time_key = iter_time_key
        self._data_time_key = data_time_key
        self._lr_key = lr_key
        self._vals = defaultdict(EMA)
        self._interval = print_interval

    def _get_eta(self, iteration) -> Optional[str]:
        if self._max_iter is None:
            return ""
        if self._iter_time_key in self._vals:
            eta_seconds = self._vals[self._iter_time_key].get() * (self._max_iter - iteration - 1)
            # storage.put_scalar("eta_seconds", eta_seconds, smoothing_hint=False)
            return str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            # estimate eta on our own - more noisy
            eta_string = None
            if self._last_write is not None:
                estimate_iter_time = (time.perf_counter() - self._last_write[1]) / (
                    iteration - self._last_write[0]
                )
                eta_seconds = estimate_iter_time * (self._max_iter - iteration - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            self._last_write = (iteration, time.perf_counter())
            return eta_string

    def write_scalar(self, iter, name, value):
        self._vals[name].put(value)

    def step(self, iteration):
        if (iteration + 1) % self._interval != 0:
            return

        if self._data_time_key in self._vals:
            data_time = self._vals[self._data_time_key].get()
        else:
            # they may not exist in the first few iterations (due to warmup)
            # or when SimpleTrainer is not used
            data_time = None

        if self._iter_time_key in self._vals:
            iter_time = self._vals[self._iter_time_key].get()
        else:
            iter_time = None

        if self._lr_key in self._vals:
            lr = "{:.5g}".format(self._vals[self._lr_key].last())
        else:
            lr = "N/A"

        eta_string = self._get_eta(iteration)

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        print(
            " {eta}iter: {iter}  {losses}  {time}{data_time}lr: {lr}  {memory}".format(
                eta=f"eta: {eta_string}  " if eta_string else "",
                iter=iteration,
                losses="  ".join(
                    ["{}: {:.4g}".format(k, v.get()) for k, v in self._vals.items() if "loss" in k]
                ),
                time="time: {:.4f}  ".format(iter_time) if iter_time is not None else "",
                data_time="data_time: {:.4f}  ".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )


class MetricLogger:
    """
    The user-facing class that provides metric storage functionalities.
    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._iter = 0
        self._writers: list[MetricWriter] = []

    def __call__(self, data_dict=None, **data_kwargs):
        if data_dict is None:
            data_dict = {}
        data = {**data_dict, **data_kwargs}

        for k, v in data.items():
            if isinstance(v, (int, float)):
                self.add_scalar(k, v)
            elif isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    self.add_scalar(k, v.item())
                else:
                    self.add_histogram(k, v)

    def add_writers(self, *writers):
        self._writers.extend(writers)

    def add_scalar(self, name, value):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.
        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.
                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        value = float(value)
        for writer in self._writers:
            writer.write_scalar(self._iter, name, value)

    def add_histogram(self, hist_name, hist_tensor):
        """
        Create a histogram from a tensor.
        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
        """
        for writer in self._writers:
            writer.write_histogram(self._iter, hist_name, hist_tensor)

    def step(self):
        """
        User should either: (1) Call this function to increment iter when needed. Or
        (2) Set `iter` to the correct iteration number before each iteration.
        The storage will then be able to associate the new data with an iteration number.
        """
        for writer in self._writers:
            writer.step(self._iter)
        self._iter += 1

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)


log = MetricLogger()
