"""Start the original API in Codespaces and wait for both trained models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

SITE = Path(__file__).resolve().parents[1] / "site"
LOG = Path("/tmp/tynys-codespaces.log")
PID = Path("/tmp/tynys-codespaces.pid")
STATUS_URL = "http://127.0.0.1:7860/api/status"


def status():
    try:
        with urlopen(STATUS_URL, timeout=3) as response:
            return json.load(response)
    except (URLError, TimeoutError, OSError, ValueError):
        return None


def main():
    current = status()
    if current and current.get("ready") and current.get("comparison_ready"):
        print("Tynys is ready at http://127.0.0.1:7860")
        return

    process = None
    running = False
    if PID.exists():
        try:
            os.kill(int(PID.read_text()), 0)
            running = True
        except (OSError, ValueError):
            pass
    if not running and current is None:
        with LOG.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"],
                cwd=SITE,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        PID.write_text(str(process.pid))

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        current = status()
        if current is not None:
            if not current.get("ready") or not current.get("comparison_ready"):
                raise RuntimeError(f"Trained models are unavailable. Read {LOG}.")
            print("Tynys is ready at http://127.0.0.1:7860")
            print("Use Ports > 7860 > Port Visibility > Public for the defense demo.")
            return
        if process and process.poll() is not None:
            raise RuntimeError(f"API stopped with exit code {process.returncode}. Read {LOG}.")
        time.sleep(2)
    raise RuntimeError(f"API startup timed out. Read {LOG}.")


if __name__ == "__main__":
    main()
