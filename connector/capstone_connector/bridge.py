import asyncio
import json
import time
from typing import Optional, Dict, Callable, Awaitable

from .constants import (
    STATE_IDLE, STATE_CLASSIFYING, STATE_TRANSCRIBING,
    OP_CLASSIFY_NOW, OP_PING,
    OP_ASR_START, OP_ASR_STOP, OP_ASR_CANCEL, OP_ASR_STATUS, OP_ASR_RESUME_SNAP
)

class Bridge:
    def __init__(
        self,
        classifier,
        transcriber,
        notify_status: Callable[[bytes], Awaitable[None]],
        notify_classify_result: Callable[[bytes], Awaitable[None]],
        notify_stream: Callable[[bytes], Awaitable[None]],
        asr_idle_timeout_s: int = 60,
        asr_hard_cap_min: int = 15,
        notify_rate_hz: int = 15
    ):
        self.classifier = classifier
        self.transcriber = transcriber
        self._notify_status = notify_status
        self._notify_result = notify_classify_result
        self._notify_stream = notify_stream

        self.state = STATE_IDLE
        self._asr_task: Optional[asyncio.Task] = None
        self._asr_started_ms: int = 0
        self._asr_idle_timeout_s = asr_idle_timeout_s
        self._asr_hard_cap_ms = asr_hard_cap_min * 60_000
        self._notify_period = 1.0 / max(1, notify_rate_hz)
        self._last_notify_ts = 0.0

    async def _send_status(self, obj: Dict):
        payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        await self._notify_status(payload)

    async def _set_state(self, new_state: str):
        self.state = new_state
        await self._send_status({"state": new_state, "ts": int(time.time()*1000)})

    async def handle_classify_write(self, data: bytes):
        if not data:
            return
        opcode = data[0]
        rest = data[1:]
        if opcode == OP_PING:
            await self._send_status({"state": self.state, "pong": True, "ts": int(time.time()*1000)})
            return
        if opcode != OP_CLASSIFY_NOW:
            await self._notify_result(json.dumps({"error":{"code":"BAD_CMD"}}, separators=(",",":")).encode("utf-8"))
            return
        # Optional JSON opts after byte 0 (currently ignored but reserved)
        await self._run_classify()

    async def _run_classify(self):
        await self._set_state(STATE_CLASSIFYING)
        try:
            res = await self.classifier.classify_once(None)
            payload = json.dumps(res, separators=(",", ":")).encode("utf-8")
            await self._notify_result(payload)
        except Exception as e:
            err = {"error":{"code":"CLASSIFY_FAIL","message":str(e)}}
            await self._notify_result(json.dumps(err, separators=(",", ":")).encode("utf-8"))
        finally:
            await self._set_state(STATE_IDLE)

    async def handle_transcribe_write(self, data: bytes):
        if not data:
            return
        opcode = data[0]
        if opcode == OP_ASR_START:
            await self._start_asr()
        elif opcode in (OP_ASR_STOP, OP_ASR_CANCEL):
            await self._stop_asr()
        elif opcode == OP_ASR_STATUS:
            await self._send_status({"state": self.state, "ts": int(time.time()*1000)})
        elif opcode == OP_ASR_RESUME_SNAP:
            await self._notify_stream(b"__SNAPSHOT__")
            await asyncio.sleep(0.02)
            await self._notify_stream(b"__SNAPEND__")
        else:
            await self._notify_stream(b"__ERROR:BAD_CMD:Unknown__")

    async def _start_asr(self):
        if self._asr_task and not self._asr_task.done():
            return
        await self._set_state(STATE_TRANSCRIBING)
        self._asr_started_ms = int(time.time() * 1000)
        await self._notify_stream(b"__START__")

        async def runner():
            idle_timer = time.time()
            try:
                async for frame in self.transcriber.stream_frames(None):
                    now = time.time()
                    typ = frame.get("type")
                    if typ == "token":
                        idle_timer = now
                        if (now - self._last_notify_ts) < self._notify_period:
                            continue
                        self._last_notify_ts = now
                        text = frame.get("t", "")
                        if not isinstance(text, str):
                            text = str(text)
                        await self._notify_stream(text.encode("utf-8"))
                    elif typ == "final":
                        text = frame.get("text", "")
                        if text:
                            await self._notify_stream(text.encode("utf-8"))
                        break
                    elif typ == "error":
                        code = frame.get("code","ERR")
                        msg = frame.get("message","stt error")
                        await self._notify_stream(f"__ERROR:{code}:{msg}__".encode("utf-8"))
                        break

                    # guardrails
                    if now - idle_timer > self._asr_idle_timeout_s:
                        break
                    if (int(time.time() * 1000) - self._asr_started_ms) > self._asr_hard_cap_ms:
                        break
            except Exception as e:
                await self._notify_stream(f"__ERROR:STT:{e}__".encode("utf-8"))
            finally:
                await self._notify_stream(b"__END__")

        self._asr_task = asyncio.create_task(runner())

    async def _stop_asr(self):
        if self._asr_task and not self._asr_task.done():
            self._asr_task.cancel()
            try:
                await self._asr_task
            except asyncio.CancelledError:
                pass
        await self._notify_stream(b"__END__")
        await self._set_state(STATE_IDLE)
