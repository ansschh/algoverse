"""Transfer files to RunPod via SSH PTY by writing base64 chunks."""
import subprocess, sys, time, base64, os

HOST = "ismx53hucp46hj-6441190e@ssh.runpod.io"
KEY = r"C:\Users\ansht\.ssh\id_ed25519"

LOCAL_TAR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/faiss_eval.tar.gz"
REMOTE_DIR = sys.argv[2] if len(sys.argv) > 2 else "/workspace/faiss_eval"
# Note: when calling from MSYS2 bash, pass the path directly - Python won't mangle it

# Read and base64 encode
with open(LOCAL_TAR, "rb") as f:
    data = base64.b64encode(f.read()).decode()

CHUNK_SIZE = 4000  # safe for terminal line length
chunks = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]

print(f"File: {LOCAL_TAR} ({os.path.getsize(LOCAL_TAR)} bytes)")
print(f"Base64: {len(data)} chars in {len(chunks)} chunks")

# Build commands
commands = f"mkdir -p {REMOTE_DIR}\n"
commands += f"rm -f /tmp/transfer.b64\n"
for i, chunk in enumerate(chunks):
    commands += f"echo -n '{chunk}' >> /tmp/transfer.b64\n"
commands += f"base64 -d /tmp/transfer.b64 | tar xzf - -C {REMOTE_DIR}\n"
commands += f"echo TRANSFER_COMPLETE\n"
commands += f"ls -la {REMOTE_DIR}/\n"
commands += f"ls -la {REMOTE_DIR}/src/\n"
commands += "exit\n"

proc = subprocess.Popen(
    ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-i", KEY, HOST],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)

proc.stdin.write(commands.encode())
proc.stdin.flush()
proc.stdin.close()

try:
    out, _ = proc.communicate(timeout=120)
    text = out.decode("utf-8", errors="replace")
    if "TRANSFER_COMPLETE" in text:
        print("Transfer successful!")
        # Print just the ls output
        lines = text.split("\n")
        capture = False
        for line in lines:
            if "TRANSFER_COMPLETE" in line:
                capture = True
                continue
            if "exit" in line and capture:
                break
            if capture:
                clean = line.strip().replace("\r", "")
                if clean:
                    print(clean)
    else:
        print("Transfer may have failed. Full output:")
        print(text[-2000:])
except subprocess.TimeoutExpired:
    proc.kill()
    print("TIMEOUT")
