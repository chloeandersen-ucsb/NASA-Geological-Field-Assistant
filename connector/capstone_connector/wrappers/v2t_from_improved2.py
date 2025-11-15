#!/usr/bin/env python3
"""
Run capstone/voice-to-text/improved2.py (or voiceTranscription.py) and convert its stdout
into NDJSON frames expected by the connector:
  {"type":"token","t":"hello "}
It strips timestamps like [20:44:13], ignores boilerplate lines, and extracts phrases
inside single quotes [...] blocks.
"""
import os, sys, json, argparse, subprocess, re
from pathlib import Path

TS_LINE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]$")
QUOTED  = re.compile(r"'([^']+)'")  # content inside single quotes
NOISE_PREFIX = ("Using device:", "RECORDING NOW", "STREAMING COMPLETE", "FINAL TRANSCRIPT")

def guess_capstone_root():
    here = Path(__file__).resolve()
    return str(here.parents[2])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default=None, help="Path to improved2.py or voiceTranscription.py")
    args = parser.parse_args()

    cap_root = os.environ.get("CAPSTONE_ROOT", guess_capstone_root())
    script = args.script or os.path.join(cap_root, "voiceNotes", "improved2.py")
    if not os.path.exists(script):
        alt = os.path.join(cap_root, "voiceNotes", "voiceTranscription.py")
        script = alt

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(["python3", "-u", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

    acc = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = (raw or "").strip()
            if not line:
                continue
            if any(line.startswith(prefix) for prefix in NOISE_PREFIX):
                continue
            if TS_LINE.match(line):
                continue
            # Extract phrases inside quotes (there may be multiple per line)
            phrases = QUOTED.findall(line)
            if not phrases:
                continue
            for ph in phrases:
                token = ph.strip()
                if token:
                    print(json.dumps({"type":"token","t": token + " "}), flush=True)
                    acc.append(token)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

    if acc:
        print(json.dumps({"type":"final","text":" ".join(acc)}), flush=True)

if __name__ == "__main__":
    main()
