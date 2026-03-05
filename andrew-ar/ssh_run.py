"""Minimal SSH runner for RunPod."""
import subprocess, sys

HOST = "442txpidwh8lkt-64411c4e@ssh.runpod.io"
KEY = r"C:\Users\ansht\.ssh\id_ed25519"

cmds = sys.argv[1] if len(sys.argv) > 1 else "echo hello"
full = cmds + "\nexit\n"

proc = subprocess.Popen(
    ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-i", KEY, HOST],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
proc.stdin.write(full.encode())
proc.stdin.flush()
proc.stdin.close()

try:
    out, _ = proc.communicate(timeout=300)
    print(out.decode("utf-8", errors="replace"))
except subprocess.TimeoutExpired:
    proc.kill()
    out, _ = proc.communicate()
    print(out.decode("utf-8", errors="replace"))
