"""Download CSV from RunPod by catting it through SSH."""
import subprocess, re, os

HOST = "442txpidwh8lkt-64411c4e@ssh.runpod.io"
KEY = r"C:\Users\ansht\.ssh\id_ed25519"

MARKER_START = "CSV_START_MARKER_XYZ"
MARKER_END = "CSV_END_MARKER_XYZ"

cmd = f"echo {MARKER_START} && cat /workspace/faiss_eval/results/faiss_results.csv && echo {MARKER_END}\nexit\n"

proc = subprocess.Popen(
    ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-i", KEY, HOST],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
proc.stdin.write(cmd.encode())
proc.stdin.flush()
proc.stdin.close()

out, _ = proc.communicate(timeout=120)
text = out.decode("utf-8", errors="replace")

# Strip ANSI
text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
text = re.sub(r'\x1b\]0;[^\x07]*\x07', '', text)
text = text.replace('\r', '')

# Extract between markers
lines = text.split('\n')
capture = False
csv_lines = []
for line in lines:
    line = line.strip()
    if MARKER_END in line:
        break
    if capture and line:
        # Skip prompt lines
        if line.startswith('root@') and '#' in line:
            continue
        csv_lines.append(line)
    if MARKER_START in line:
        capture = True

if csv_lines:
    local_path = r"A:\algoverse-andrew-ar\results\faiss_results.csv"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"Downloaded {len(csv_lines)} lines to {local_path}")
else:
    print("FAILED - no CSV content found")
    print("Last 2000 chars of output:")
    print(text[-2000:])
