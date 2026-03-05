"""Helper to run commands on RunPod via SSH with PTY."""
import subprocess
import sys
import time

HOST = sys.argv[1] if len(sys.argv) > 1 else "ismx53hucp46hj-6441190e@ssh.runpod.io"
KEY = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\ansht\.ssh\id_ed25519"
SCRIPT = sys.argv[3] if len(sys.argv) > 3 else None

if SCRIPT:
    with open(SCRIPT) as f:
        commands = f.read()
else:
    commands = sys.stdin.read()

# Add exit at end and a unique marker
MARKER = "___DONE_MARKER_12345___"
full_cmd = commands.strip() + f"\necho {MARKER}\nexit\n"

proc = subprocess.Popen(
    ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-i", KEY, HOST],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Send commands
proc.stdin.write(full_cmd.encode())
proc.stdin.flush()
proc.stdin.close()

# Read output with timeout
output = b""
try:
    output, err = proc.communicate(timeout=300)
except subprocess.TimeoutExpired:
    proc.kill()
    output, err = proc.communicate()

# Print output, stripping ANSI codes and RunPod banner
text = output.decode("utf-8", errors="replace")
# Find content after the banner
lines = text.split("\n")
printing = False
for line in lines:
    clean = line.strip().replace("\r", "")
    if "root@" in clean and "#" in clean and not printing:
        printing = True
        continue
    if MARKER in clean:
        break
    if printing:
        print(clean)
