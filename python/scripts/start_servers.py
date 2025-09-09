
#!/usr/bin/env python
from __future__ import annotations
import argparse, subprocess, os, time

try:
    import requests
except Exception:
    requests = None

def launch(binary: str, host: str, port: int, seed: int, frame_skip: int, render: bool, render_every: int, agent: str, run_id: str):
    cmd = [binary, "--server", "--host", host, "--port", str(port),
           "--frame-skip", str(frame_skip), "--render-every", str(render_every),
           "--seed", str(seed), "--agent", agent, "--run_id", run_id]
    if render:
        cmd.append("--render")
    print("[START]", " ".join(cmd))
    return subprocess.Popen(cmd)

def wait_healthy(host: str, port: int, timeout_s: float = 15.0) -> bool:
    if requests is None:
        time.sleep(0.5)
        return True
    base = f"http://{host}:{port}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(base + "/health", timeout=1.5)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, help="Pfad zur C++-Server-Binary")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--base-port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-every", type=int, default=1)
    ap.add_argument("--agent", default="sb3")
    ap.add_argument("--run-id", default="partA_servers")
    args = ap.parse_args()

    procs = []
    try:
        for i in range(args.n):
            port = args.base_port + i
            p = launch(args.binary, args.host, port, args.seed + i, args.frame_skip, args.render, args.render_every, args.agent, args.run_id)
            procs.append(p)
            ok = wait_healthy(args.host, port)
            print(f"  -> port {port} healthy={ok}")
        print("[RUNNING] Press Ctrl+C to stop all servers…")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[STOP] stopping servers…")
    finally:
        for p in procs:
            try: p.terminate()
            except Exception: pass
        for p in procs:
            try: p.wait(timeout=5)
            except Exception:
                try: p.kill()
                except Exception: pass
        print("[DONE]")

if __name__ == "__main__":
    main()
