import signal
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    BASE_DIR / "dm 30.py",
    BASE_DIR / "process.py",
]


import time
import subprocess
import sys

def start_scripts():
    processes = []

    for script in SCRIPTS:
        if not script.exists():
            print(f"Missing file: {script}")
            continue

        if script.name == "process.py":
            print("Waiting 8 seconds before starting process.py...")
            time.sleep(8)

        proc = subprocess.Popen([
            sys.executable,
            str(script),
        ])

        processes.append((script.name, proc))
        print(f"Started {script.name} (PID: {proc.pid})")

    return processes


def stop_scripts(processes):
    for name, proc in processes:
        if proc.poll() is None:
            print(f"Stopping {name} (PID: {proc.pid})")
            proc.terminate()

    for name, proc in processes:
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing {name} (PID: {proc.pid})")
                proc.kill()


def main():
    processes = start_scripts()

    if not processes:
        print("No scripts started.")
        return

    print("Both scripts are running. Press Ctrl+C to stop both.")

    try:
        for _, proc in processes:
            proc.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C received. Shutting down both scripts...")
        stop_scripts(processes)


if __name__ == "__main__":
    # Ensures Ctrl+C can be handled by this launcher process.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
