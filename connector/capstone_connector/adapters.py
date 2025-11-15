import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Optional

class ClassifierAdapter:
    """Adapter for a one-shot classifier subprocess that prints a single JSON line."""
    def __init__(self, mode: str = "subprocess", subprocess_cmd: Optional[list] = None):
        self.mode = mode
        self.subprocess_cmd = subprocess_cmd

    async def classify_once(self, opts: Optional[Dict] = None) -> Dict:
        t0 = time.time()
        if self.mode != "subprocess" or not self.subprocess_cmd:
            raise RuntimeError("ClassifierAdapter requires subprocess mode with subprocess_cmd")
        # Run the wrapper which returns normalized JSON on stdout
        proc = await asyncio.create_subprocess_exec(
            *self.subprocess_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await proc.communicate()
            # Decode bytes to string (Python 3.13 compatibility)
            s = (out.decode('utf-8') if out else "").strip()
            try:
                obj = json.loads(s)
            except Exception as e:
                raise RuntimeError(f"classifier wrapper did not return JSON: {s[:200]} ...")
            # Ensure schema
            label = str(obj.get("label", "Unknown"))
            conf = int(obj.get("confidence", 0))
            return {
                "label": label,
                "confidence": max(0, min(100, conf)),
                "latency_ms": int((time.time() - t0) * 1000),
                "ts": int(time.time())
            }
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


class TranscriberAdapter:
    """Adapter for a token-streaming subprocess that prints one NDJSON object per line."""
    def __init__(self, mode: str = "subprocess", subprocess_cmd: Optional[list] = None):
        self.mode = mode
        self.subprocess_cmd = subprocess_cmd

    async def stream_frames(self, cfg: Optional[Dict] = None) -> AsyncGenerator[Dict, None]:
        if self.mode != "subprocess" or not self.subprocess_cmd:
            raise RuntimeError("TranscriberAdapter requires subprocess mode with subprocess_cmd")

        proc = await asyncio.create_subprocess_exec(
            *self.subprocess_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        try:
            assert proc.stdout is not None
            async for line_bytes in proc.stdout:
                # Decode bytes to string (Python 3.13 compatibility)
                line = line_bytes.decode('utf-8').strip()
                if not line:
                    continue
                # Expect NDJSON: {"type":"token","t":"..."}
                try:
                    obj = json.loads(line)
                    yield obj
                except Exception:
                    # ignore non-JSON lines from the child
                    continue
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
