"""Download results from RunPod pod."""
import subprocess, sys, os, base64, re

HOST = "442txpidwh8lkt-64411c4e@ssh.runpod.io"
KEY = r"C:\Users\ansht\.ssh\id_ed25519"
LOCAL_DIR = r"A:\algoverse-andrew-ar"

files_to_download = [
    "/workspace/faiss_eval/results/faiss_results.csv",
    "/workspace/faiss_eval/plots/pareto_raw_ip.png",
    "/workspace/faiss_eval/plots/pareto_norm_ip.png",
    "/workspace/faiss_eval/plots/compression_poison_raw_ip.png",
    "/workspace/faiss_eval/plots/compression_poison_norm_ip.png",
    "/workspace/faiss_eval/plots/compression_index_raw_ip.png",
    "/workspace/faiss_eval/plots/compression_index_norm_ip.png",
    "/workspace/faiss_eval/run.log",
]

for remote_path in files_to_download:
    filename = os.path.basename(remote_path)
    # Determine local subdir
    if "results" in remote_path:
        local_path = os.path.join(LOCAL_DIR, "results", filename)
    elif "plots" in remote_path:
        local_path = os.path.join(LOCAL_DIR, "plots", filename)
    else:
        local_path = os.path.join(LOCAL_DIR, filename)

    print(f"Downloading {remote_path} -> {local_path}")

    cmd = f"base64 {remote_path}"
    full = cmd + "\nexit\n"

    proc = subprocess.Popen(
        ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-i", KEY, HOST],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    proc.stdin.write(full.encode())
    proc.stdin.flush()
    proc.stdin.close()

    try:
        out, _ = proc.communicate(timeout=60)
        text = out.decode("utf-8", errors="replace")

        # Extract base64 content between the command echo and the next prompt
        lines = text.split('\n')
        b64_lines = []
        capture = False
        for line in lines:
            clean = line.strip().replace('\r', '')
            # Stop at prompt or exit
            if capture and (clean.startswith('root@') or 'exit' in clean):
                break
            if capture and clean:
                # Filter out ANSI codes
                clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', clean)
                clean = re.sub(r'\x1b\]0;[^\x07]*\x07', '', clean)
                clean = clean.strip()
                if clean and not clean.startswith('[?') and not clean.startswith('root@'):
                    b64_lines.append(clean)
            # Start after the command is echoed
            if f"base64 {remote_path}" in line:
                capture = True

        b64_data = ''.join(b64_lines)
        if b64_data:
            data = base64.b64decode(b64_data)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(data)
            print(f"  OK ({len(data)} bytes)")
        else:
            print(f"  FAILED (no data captured)")

    except Exception as e:
        print(f"  ERROR: {e}")
