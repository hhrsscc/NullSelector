""" Timer """

import os
import time
from typing import NoReturn
from contextlib import contextmanager

import psutil


class MultiLayerTimer:
    """ Multi Layer Timer """
    def __init__(self):
        self._depth = 0


    @contextmanager
    def timer(self, text: str) -> NoReturn:
        """ text """
        start_time = time.time()
        self._depth += 1
        yield
        used_time = time.time() - start_time
        memory_free = psutil.virtual_memory().free / (1024 ** 3)

        pid = os.getpid()
        memory_used = psutil.Process(pid).memory_info()[0] / (1024 **3)
        print("--" * self._depth + ">" + \
              "[%s] Time used: %.2fs, memory used %.2fG, free %.2fG" % \
              (text, used_time, memory_used, memory_free))
        self._depth -= 1


    def reset(self) -> NoReturn:
        """ reset depth """
        self._depth = 0
