import threading
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from time import sleep

import streamlit as st
from streamlit.report_thread import ReportContext, add_report_ctx


class ExecutorSingleton(object):
    __instance = None

    def __new__(cls):
        if ExecutorSingleton.__instance is None:
            ExecutorSingleton.__instance = ThreadPoolExecutor(max_workers=2)
        return ExecutorSingleton.__instance


class _ContextWrapper:
    def __init__(self, func, *args, ctx):
        self._func = func
        self._args = args
        self._ctx = ctx

    def __call__(self, *args, **kwargs):
        add_report_ctx(threading.currentThread(), self._ctx)
        return self._func(*self._args)


def queue_and_get(func, *args, ctx: ReportContext = None) -> Future:
    if ctx:
        future = ExecutorSingleton().submit(_ContextWrapper(func, *args, ctx=ctx))
    else:
        future = ExecutorSingleton().submit(func, *args)

    with st.spinner('You are in a queue. Please wait.'):
        while not future.running():
            if future.done() or future.cancelled():
                break
            sleep(0.1)

    return future



