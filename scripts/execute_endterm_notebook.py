"""Execute and retain the real Endterm notebook, including partial output on failure."""
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def execute():
    root = Path(__file__).resolve().parents[1]
    runtime = root / "work/jupyter-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    path = root / "notebooks/endterm.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=3600,
                            resources={"metadata": {"path": str(root)}}, record_timing=True)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print("Executed Endterm notebook successfully.", flush=True)


if __name__ == "__main__":
    execute()
