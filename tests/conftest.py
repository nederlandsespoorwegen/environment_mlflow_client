"""
session wide MLFlow instance fixture
"""
import os
import psutil
import subprocess
import tempfile
import time

import pytest


@pytest.fixture(autouse=True, scope="session")
def run_mlflow():
    """Start an mlflow server. Files are stored in temporary directories so that they are automatically
    removed after testing."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmpfile:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                f"mlflow server --backend-store-uri sqlite:///{tmpfile.name} --default-artifact-root file:{tmpdir} --host 0.0.0.0 --port 5000"
            ]
            print(cmd)
            proc = subprocess.Popen(
                cmd, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True
            )
            time.sleep(5)  # Give it some time to start, prevents flakiness

            os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

            yield

            # MLflow does not gracefully shutdown, so list all child processes and kill them individually
            child_processes = psutil.Process(proc.pid).children(recursive=True)
            for child in child_processes:
                child.kill()
