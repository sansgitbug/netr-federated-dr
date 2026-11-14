"""Run the whole federation on one machine: server + dashboard + N hospital clients.

    python run_demo.py --clients 3 --rounds 5

Each hospital is a separate OS process that reads only its own data zip and talks to the
server over a network port (localhost), exactly as separate machines would.
Press Ctrl+C to stop everything.
"""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def pump(proc, tag):
    for line in iter(proc.stdout.readline, ""):
        print(f"[{tag}] {line.rstrip()}", flush=True)


def spawn(tag, args):
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "TF_ENABLE_ONEDNN_OPTS": "0",
           "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [PY, "-u", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )
    threading.Thread(target=pump, args=(proc, tag), daemon=True).start()
    return proc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--fresh", action="store_true", help="reset the global model first")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    missing = [i for i in range(1, args.clients + 1)
               if not (ROOT / "data_zips" / f"hospital_{i}_data.zip").exists()]
    if missing:
        sys.exit(f"Missing data zips for hospital(s) {missing}. "
                 "Run: python scripts/prepare_data.py --help")

    procs = []
    try:
        server_cmd = ["server/flower_server.py", "--clients", str(args.clients),
                      "--rounds", str(args.rounds), "--port", str(args.port),
                      "--host", "127.0.0.1"]
        if args.fresh:
            server_cmd.append("--fresh")
        server = spawn("server", server_cmd)
        procs.append(server)
        # The first start downloads ImageNet weights, so wait for the port, not a fixed delay.
        deadline = time.time() + 300
        while time.time() < deadline:
            if server.poll() is not None:
                sys.exit("Server exited during startup; see its output above.")
            try:
                socket.create_connection(("127.0.0.1", args.port), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            sys.exit("Server did not open its port within 5 minutes.")

        if not args.no_dashboard:
            procs.append(spawn("dashboard", ["-m", "streamlit", "run", "server/dashboard.py",
                                             "--server.port", "8501", "--server.headless", "true"]))
            print("Dashboard: http://localhost:8501", flush=True)

        for i in range(1, args.clients + 1):
            procs.append(spawn(f"hospital {i}", ["client.py", "--hospital-id", str(i),
                                                 "--server", f"127.0.0.1:{args.port}"]))
            time.sleep(1)

        server.wait()
        print("Training finished. Dashboard stays up; press Ctrl+C to exit.", flush=True)
        while any(p.poll() is None for p in procs if p is not server):
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
