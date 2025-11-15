import asyncio
import os
import yaml
from types import MethodType

from dbus_next.aio import MessageBus
from dbus_next import Variant

from .ble_gatt import ObjectManager, Service, Characteristic, Advertisement, find_adapter_path, register_app, register_advertisement, secure_flags
from .constants import *
from .adapters import ClassifierAdapter, TranscriberAdapter
from .bridge import Bridge

BASE = "/com/capstone/connector"
APP_PATH = f"{BASE}/app"
CLASSIFY_SVC_PATH = f"{BASE}/svc_classify"
ASR_SVC_PATH = f"{BASE}/svc_asr"

def ch_path(svc_path: str, name: str) -> str:
    return f"{svc_path}/{name}"

async def main():
    cfg_path = os.environ.get("CAPSTONE_CONNECTOR_CONFIG", os.path.join(os.path.dirname(__file__), "config.yaml"))
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    require_pairing = bool(cfg.get("device", {}).get("require_pairing", True))
    advertised_name = cfg.get("device", {}).get("advertised_name", ADVERTISED_NAME)
    adapter_name = cfg.get("device", {}).get("adapter", "hci0")

    classifier = ClassifierAdapter(
        mode="subprocess",
        subprocess_cmd=cfg.get("classifier", {}).get("subprocess_cmd")
    )
    transcriber = TranscriberAdapter(
        mode="subprocess",
        subprocess_cmd=cfg.get("transcriber", {}).get("subprocess_cmd"),
    )

    bus = await MessageBus().connect()

    objmgr = ObjectManager(APP_PATH)
    bus.export(objmgr.path, objmgr)

    # Classification service
    classify_svc = Service(CLASSIFY_SVC_PATH, CLASSIFICATION_SERVICE_UUID, primary=True)
    bus.export(classify_svc.path, classify_svc)
    objmgr.add(classify_svc.path, {"org.bluez.GattService1": {
        "UUID": Variant("s", CLASSIFICATION_SERVICE_UUID),
        "Primary": Variant("b", True),
        "Includes": Variant("ao", [])
    }})

    classify_cmd = Characteristic(bus, ch_path(CLASSIFY_SVC_PATH, "cmd"), CLASSIFY_COMMAND_CHAR_UUID, classify_svc.path,
                                  secure_flags(["write-without-response", "write"], require_pairing))
    classify_res = Characteristic(bus, ch_path(CLASSIFY_SVC_PATH, "result"), CLASSIFY_RESULT_CHAR_UUID, classify_svc.path,
                                  secure_flags(["notify", "read"], require_pairing))
    classify_status = Characteristic(bus, ch_path(CLASSIFY_SVC_PATH, "status"), CLASS_STATUS_CHAR_UUID, classify_svc.path,
                                     secure_flags(["notify", "read"], require_pairing))

    for ch in (classify_cmd, classify_res, classify_status):
        bus.export(ch.path, ch)
        objmgr.add(ch.path, {"org.bluez.GattCharacteristic1": {
            "UUID": Variant("s", ch.uuid),
            "Service": Variant("o", ch.service_path),
            "Flags": Variant("as", ch.flags)
        }})

    # Transcription service
    asr_svc = Service(ASR_SVC_PATH, TRANSCRIPTION_SERVICE_UUID, primary=True)
    bus.export(asr_svc.path, asr_svc)
    objmgr.add(asr_svc.path, {"org.bluez.GattService1": {
        "UUID": Variant("s", TRANSCRIPTION_SERVICE_UUID),
        "Primary": Variant("b", True),
        "Includes": Variant("ao", [])
    }})

    asr_cmd = Characteristic(bus, ch_path(ASR_SVC_PATH, "cmd"), TRANSCRIBE_COMMAND_CHAR_UUID, asr_svc.path,
                             secure_flags(["write-without-response", "write"], require_pairing))
    asr_stream = Characteristic(bus, ch_path(ASR_SVC_PATH, "stream"), TRANSCRIBE_STREAM_CHAR_UUID, asr_svc.path,
                                secure_flags(["notify", "read"], require_pairing))

    for ch in (asr_cmd, asr_stream):
        bus.export(ch.path, ch)
        objmgr.add(ch.path, {"org.bluez.GattCharacteristic1": {
            "UUID": Variant("s", ch.uuid),
            "Service": Variant("o", ch.service_path),
            "Flags": Variant("as", ch.flags)
        }})

    # Bridge
    bridge = Bridge(
        classifier=classifier,
        transcriber=transcriber,
        notify_status=lambda b: classify_status.notify(b),
        notify_classify_result=lambda b: classify_res.notify(b),
        notify_stream=lambda b: asr_stream.notify(b),
        asr_idle_timeout_s=int(cfg.get("runtime", {}).get("asr_idle_timeout_s", 60)),
        asr_hard_cap_min=int(cfg.get("runtime", {}).get("asr_hard_cap_min", 15)),
        notify_rate_hz=int(cfg.get("runtime", {}).get("notify_rate_hz", DEFAULT_NOTIFY_RATE_HZ)),
    )

    def on_write_classify(self, value, options):
        asyncio.create_task(bridge.handle_classify_write(bytes(value)))
    def on_write_asr(self, value, options):
        asyncio.create_task(bridge.handle_transcribe_write(bytes(value)))

    classify_cmd.WriteValue = MethodType(lambda s, v, o: on_write_classify(s, v, o), classify_cmd)
    asr_cmd.WriteValue = MethodType(lambda s, v, o: on_write_asr(s, v, o), asr_cmd)

    await bridge._set_state(STATE_IDLE)

    # Register app + advertise both services
    adapter_path = await find_adapter_path(bus, adapter_name)
    adv = Advertisement(f"{BASE}/adv0", "peripheral", os.environ.get("CAPSTONE_AD_NAME", advertised_name),
                        [CLASSIFICATION_SERVICE_UUID, TRANSCRIPTION_SERVICE_UUID])
    bus.export(adv.path, adv)
    await register_app(bus, adapter_path, objmgr)
    await register_advertisement(bus, adapter_path, adv)

    print(f"[connector] Advertising '{advertised_name}' on {adapter_path}")
    print(f"[connector] Services ready: Classification + Transcription")

    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
