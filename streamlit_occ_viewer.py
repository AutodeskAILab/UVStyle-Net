from io import StringIO
from typing import Optional, Tuple

from occwl.jupyter_viewer import JupyterViewer
from streamlit.components import v1 as components


class StreamlitOCCViewer(JupyterViewer):

    def __init__(self, size: Optional[Tuple[int, int]] = (640, 480), background_color: Optional[str] = "white"):
        super().__init__(size, background_color)
        self._size = size

    def show(self):
        super().show()

        with StringIO() as f:
            self._renderer.ExportToHTML(f)
            html_code = f.getvalue()

        width, height = self._size
        return components.html(html=html_code,
                               width=width,
                               height=height)
