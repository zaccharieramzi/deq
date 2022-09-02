import os

import pytest


FILES_EXTENSION_TO_DELETE = [".pth", ".pt", ".csv"]


@pytest.fixture
def clean_up_files():
    yield
    for f in os.listdir("./"):
        if any(f.endswith(ext) for ext in FILES_EXTENSION_TO_DELETE):
            os.remove(f)
