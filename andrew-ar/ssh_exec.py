"""SSH command executor for RunPod. Cleans output."""
import subprocess, sys, re

HOST = "ismx53hucp46hj-6441190e@ssh.runpod.io"
KEY = r"C:\Users\ansht\.ssh\id_ed25519"

cmds = sys.argv[1] if len(sys.argv) > 1 else "echo hello"
MARKER = "XDONE9876X"
full = cmds + f"\necho {MARKER}\nexit\n"

proc = subprocess.Popen(
    ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-i", KEY, HOST],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
proc.stdin.write(full.encode())
proc.stdin.flush()
proc.stdin.close()

timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600
try:
    out, _ = proc.communicate(timeout=timeout)
except subprocess.TimeoutExpired:
    proc.kill()
    out, _ = proc.communicate()

text = out.decode("utf-8", errors="replace")
# Strip ANSI escape codes
text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
text = re.sub(r'\x1b\]0;[^\x07]*\x07', '', text)  # title sequences
text = text.replace('\r', '')

# Extract between first command echo and MARKER
lines = text.split('\n')
capture = False
for line in lines:
    if MARKER in line:
        break
    if capture:
        # Skip prompt lines
        if re.match(r'root@\w+:.*#', line.strip()):
            continue
        print(line)
    # Start capturing after the banner
    if 'docs.runpod.io' in line:
        capture = True
