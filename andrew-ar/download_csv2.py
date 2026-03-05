"""Download CSV from RunPod - extract CSV lines by content pattern."""
import subprocess, re, os

HOST = "442txpidwh8lkt-64411c4e@ssh.runpod.io"
KEY = r"C:\Users\ansht\.ssh\id_ed25519"

cmd = "cat /workspace/faiss_eval/results/faiss_results.csv\nexit\n"

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

# Extract CSV lines: header starts with "regime,", data starts with "raw_ip" or "norm_ip"
csv_lines = []
for line in text.split('\n'):
    line = line.strip()
    if line.startswith('regime,') or line.startswith('raw_ip,') or line.startswith('norm_ip,'):
        csv_lines.append(line)

if csv_lines:
    local_path = r"A:\algoverse-andrew-ar\results\faiss_results.csv"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print(f"Downloaded {len(csv_lines)} lines ({len(csv_lines)-1} data rows)")
else:
    print("FAILED")
