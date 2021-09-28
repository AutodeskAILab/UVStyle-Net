import os
import re
import shutil
import subprocess
import time
from subprocess import PIPE
from threading import Thread
from time import sleep

import requests
import streamlit as st
from streamlit.components import v1 as components


class StEmbeddingProjector:
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._process = subprocess.Popen(f'bash -c "tensorboard --logdir {log_dir}"',
                                         env=os.environ,
                                         shell=True,
                                         universal_newlines=True,
                                         stdout=PIPE,
                                         stderr=PIPE,
                                         bufsize=1)

        # read stderr to find port
        port = None
        for line in iter(self._process.stderr.readline, ""):
            if line.find('http://localhost:') > 0:
                port = re.search(r'localhost:(\d+)/', str(line)).group(1)
                port = int(port)
                break
        if port is None:
            raise Exception('Cannot start Tensorboard.')
        print(f'Tensorboard running on port {port}')
        self._port = port

        # block until tensorboard is running
        tensorboard_url = f'http://localhost:{port}'
        success = False
        start = time.time()
        while not success:
            try:
                requests.get(tensorboard_url)
                success = True
            except Exception:
                if time.time() - start > 10:
                    raise Exception('Cannot connect to embedding projector (timeout).')
                sleep(0.1)

    def display(self):
        # hack to show only embedding projector from tensorboard
        selection_html = f"""
        <style>
        #my-div
        {{
            width    : 1000px;
            height   : 800px;
            overflow : hidden;
            position : relative;
        }}
        #my-iframe
        {{
            position : absolute;
            top      : -68px;
            left     : -313px;
            width    : 1602px;
            height   : 868px;
        }}
        </style>
        <div id="my-div">
        <iframe src="http://localhost:{self._port}/#projector" id="my-iframe" scrolling="no"></iframe>
        </div>
        """
        st.text('Use left mouse button to rotate, right mouse button to pan, and scroll wheel to zoom.')
        components.html(html=selection_html,
                        width=1000,
                        height=1000)

        # kill tensorboard after 10s, it's not needed once running in client browser
        Thread(target=self._close_on_timeout, args=[10]).start()

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def port(self):
        return self._port

    def _close_on_timeout(self, timeout):
        sleep(timeout)
        shutil.rmtree(self._log_dir, ignore_errors=True)
        self._process.kill()


if __name__ == '__main__':
    StEmbeddingProjector('temp')
