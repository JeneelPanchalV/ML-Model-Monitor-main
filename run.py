"""
Convenience launcher for all system components.

Usage:
  python run.py train       — Train the model
  python run.py serve       — Start FastAPI server
  python run.py monitor     — Run one monitoring cycle
  python run.py schedule    — Start monitoring scheduler
  python run.py dashboard   — Launch Streamlit dashboard
  python run.py retrain     — Retrain if needed (add --force to always retrain)
  python run.py all         — Train + serve + schedule + dashboard (parallel)
"""
import sys
import os
import subprocess


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "train":
        subprocess.run([sys.executable, "model/train.py"], check=True)

    elif cmd == "serve":
        import threading, webbrowser, time
        def open_browser():
            time.sleep(1.5)
            webbrowser.open("http://localhost:8000")
        threading.Thread(target=open_browser, daemon=True).start()
        subprocess.run([sys.executable, "-m", "uvicorn", "serving.app:app",
                        "--reload", "--host", "0.0.0.0", "--port", "8000"], check=True)

    elif cmd == "monitor":
        subprocess.run([sys.executable, "monitoring/monitor.py"], check=True)

    elif cmd == "schedule":
        subprocess.run([sys.executable, "monitoring/scheduler.py"], check=True)

    elif cmd == "dashboard":
        subprocess.run(["streamlit", "run", "dashboard/app.py"], check=True)

    elif cmd == "ui":
        import webbrowser, pathlib
        path = pathlib.Path("dashboard/index.html").resolve().as_uri()
        print(f"Opening {path}")
        webbrowser.open(path)

    elif cmd == "retrain":
        extra = ["--force"] if "--force" in sys.argv else []
        subprocess.run([sys.executable, "retrain/retrain.py"] + extra, check=True)

    elif cmd == "all":
        import threading
        procs = [
            ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"],
            ["python", "monitoring/scheduler.py"],
            ["streamlit", "run", "dashboard/app.py"],
        ]
        processes = [subprocess.Popen(p) for p in procs]
        print("All services started. Press Ctrl+C to stop.")
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


def _run_scheduler_subprocess():
    """
    Target for the background thread in prod mode.
    Waits for the API to be ready, then launches the scheduler as a
    child process.  If it crashes it restarts automatically with a
    short back-off so a transient error doesn't kill monitoring forever.
    """
    import time, logging

    log = logging.getLogger("prod.scheduler")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [scheduler-thread] %(message)s")

    # Give uvicorn time to bind its port before the scheduler tries to
    # call any internal API endpoints on startup.
    log.info("Waiting 15 s for API to be ready…")
    time.sleep(15)

    backoff = 5          # seconds between restart attempts (doubles on each crash)
    max_backoff = 120    # cap at 2 minutes

    while True:
        log.info("Starting monitoring/scheduler.py …")
        proc = subprocess.Popen(
            [sys.executable, "monitoring/scheduler.py"],
            # Inherit stdout/stderr so Railway log aggregator sees scheduler output
            stdout=None,
            stderr=None,
        )
        exit_code = proc.wait()
        log.warning(
            "scheduler.py exited with code %s. Restarting in %s s …",
            exit_code, backoff,
        )
        time.sleep(backoff)
        backoff = min(backoff * 2, max_backoff)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prod":
        import threading
        import uvicorn

        # ── Start the monitoring scheduler in a background daemon thread ──
        # daemon=True means it is automatically killed when the main process
        # (uvicorn) exits, so Railway's restart policy stays in control.
        scheduler_thread = threading.Thread(
            target=_run_scheduler_subprocess,
            name="scheduler-thread",
            daemon=True,
        )
        scheduler_thread.start()

        # ── Start the FastAPI server (blocking — keeps the process alive) ──
        uvicorn.run(
            "serving.app:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000)),
        )
    else:
        main()