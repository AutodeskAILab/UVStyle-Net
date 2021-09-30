import threading
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from time import sleep

import streamlit as st
from streamlit.report_thread import ReportContext, add_report_ctx


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')


class _StExecutor(object):
    class _ContextWrapper:
        def __init__(self, func, *args, ctx):
            self._func = func
            self._args = args
            self._ctx = ctx

        def __call__(self, *args, **kwargs):
            add_report_ctx(threading.currentThread(), self._ctx)
            return self._func(*self._args)

    @property
    @abstractmethod
    def _executor(self):
        pass

    def queue_and_block(self, func, *args, ctx: ReportContext = None) -> Future:
        if ctx:
            future = self._executor.submit(self._ContextWrapper(func, *args, ctx=ctx))
        else:
            future = self._executor.submit(func, *args)

        with st.spinner('You are in a queue. Please wait.'):
            while not future.running():
                if future.done() or future.cancelled():
                    break
                sleep(0.1)

        return future


@Singleton
class SingleThreadExecutor(_StExecutor):

    def __init__(self) -> None:
        super().__init__()
        self.__executor = ThreadPoolExecutor(max_workers=1,
                                             thread_name_prefix='embeddings_')

    @property
    def _executor(self):
        return self.__executor


@Singleton
class ManyThreadExecutor(_StExecutor):

    def __init__(self) -> None:
        super().__init__()
        self.__executor = ThreadPoolExecutor(max_workers=6,
                                             thread_name_prefix='embeddings_')

    @property
    def _executor(self):
        return self.__executor
