import logging
import time
from functools import wraps
from memory_profiler import memory_usage


def profile(func):
    @wraps(func)
    def inner(*args, **kwargs):
        func_kwargs_str = ', '.join(f'{key}={value}' for key, value in kwargs.items())
        logging.info(f'{func.__name__}({func_kwargs_str})')

        # Measure time and memory
        t = time.perf_counter()
        mem, retval = memory_usage((func, args, kwargs), retval=True, timeout=200, interval=1e-7)
        elapsed = time.perf_counter() - t
        logging.info(f'Time   {elapsed:0.4} Seconds')
        logging.info(f'Memory {max(mem) - min(mem)} MB')

        return retval

    return inner
