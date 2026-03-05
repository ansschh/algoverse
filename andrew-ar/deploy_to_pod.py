"""Deploy project files to RunPod pod via SSH PTY session."""
import subprocess, sys, os, base64, re

HOST = "442txpidwh8lkt-64411c4e@ssh.runpod.io"
KEY = r"C:\Users\ansht\.ssh\id_ed25519"
REMOTE_BASE = "/workspace/faiss_eval"

# Files to transfer
LOCAL_BASE = r"A:\algoverse-andrew-ar"
files = [
    ("run_evaluation.py", "run_evaluation.py"),
    ("requirements.txt", "requirements.txt"),
    ("src/generate_synthetic_data.py", "src/generate_synthetic_data.py"),
    ("src/faiss_eval.py", "src/faiss_eval.py"),
    ("src/plot_results.py", "src/plot_results.py"),
]

# Build all commands
commands = []
commands.append(f"mkdir -p {REMOTE_BASE}/src {REMOTE_BASE}/data {REMOTE_BASE}/results {REMOTE_BASE}/plots")

for local_rel, remote_rel in files:
    local_path = os.path.join(LOCAL_BASE, local_rel)
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    remote_path = f"{REMOTE_BASE}/{remote_rel}"
    # Write via base64 decode - chunk if needed
    CHUNK = 3500
    chunks = [b64[i:i+CHUNK] for i in range(0, len(b64), CHUNK)]
    commands.append(f"rm -f /tmp/_filetmp")
    for chunk in chunks:
        commands.append(f"echo -n '{chunk}' >> /tmp/_filetmp")
    commands.append(f"base64 -d /tmp/_filetmp > {remote_path}")
    commands.append(f"echo 'Wrote {remote_rel}'")

commands.append(f"echo '---VERIFY---'")
commands.append(f"ls -la {REMOTE_BASE}/")
commands.append(f"ls -la {REMOTE_BASE}/src/")
commands.append(f"wc -l {REMOTE_BASE}/run_evaluation.py {REMOTE_BASE}/src/*.py")
commands.append("echo '---ALL_DONE---'")
commands.append("exit")

full_script = "\n".join(commands) + "\n"

print(f"Deploying {len(files)} files to {HOST}:{REMOTE_BASE}")
print(f"Total commands: {len(commands)}")

proc = subprocess.Popen(
    ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-i", KEY, HOST],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)

proc.stdin.write(full_script.encode())
proc.stdin.flush()
proc.stdin.close()

try:
    out, _ = proc.communicate(timeout=120)
    text = out.decode("utf-8", errors="replace")
    # Clean ANSI
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    text = re.sub(r'\x1b\]0;[^\x07]*\x07', '', text)

    if "ALL_DONE" in text:
        print("\nDeploy successful!")
        # Show verify section
        in_verify = False
        for line in text.split('\n'):
            line = line.strip().replace('\r', '')
            if '---ALL_DONE---' in line:
                break
            if in_verify and line and not line.startswith('root@'):
                print(f"  {line}")
            if '---VERIFY---' in line:
                in_verify = True
    else:
        print("\nDeploy may have failed. Last 1500 chars:")
        print(text[-1500:])

except subprocess.TimeoutExpired:
    proc.kill()
    print("TIMEOUT during deploy")
