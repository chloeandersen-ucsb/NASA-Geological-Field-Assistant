import asyncio
from typing import List, Dict

from dbus_next.service import ServiceInterface, method, dbus_property
from dbus_next.aio import MessageBus
from dbus_next import Variant, Message

BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
LE_ADV_MGR_IFACE   = "org.bluez.LEAdvertisingManager1"
PROP_IFACE         = "org.freedesktop.DBus.Properties"

class ObjectManager(ServiceInterface):
    def __init__(self, path: str):
        super().__init__("org.freedesktop.DBus.ObjectManager")
        self.path = path
        self._managed: Dict[str, Dict[str, Dict[str, Variant]]] = {}

    @method()
    def GetManagedObjects(self) -> "a{oa{sa{sv}}}":
        return self._managed

    def add(self, path: str, ifaces: Dict[str, Dict[str, Variant]]):
        self._managed[path] = ifaces


class Service(ServiceInterface):
    def __init__(self, path: str, uuid: str, primary: bool = True):
        super().__init__("org.bluez.GattService1")
        self.path = path
        self.uuid = uuid
        self.primary = primary

    @dbus_property()
    def UUID(self) -> "s":
        return self.uuid

    @dbus_property()
    def Primary(self) -> "b":
        return self.primary

    @dbus_property()
    def Includes(self) -> "ao":
        return []


class Characteristic(ServiceInterface):
    def __init__(self, bus: MessageBus, path: str, uuid: str, service_path: str, flags: List[str]):
        super().__init__("org.bluez.GattCharacteristic1")
        self.bus = bus
        self.path = path
        self.uuid = uuid
        self.service_path = service_path
        self.flags = flags
        self._value = bytearray()
        self._notifying = False

    @dbus_property()
    def UUID(self) -> "s":
        return self.uuid

    @dbus_property()
    def Service(self) -> "o":
        return self.service_path

    @dbus_property()
    def Flags(self) -> "as":
        return self.flags

    @method()
    def ReadValue(self, options: "a{sv}") -> "ay":
        return self._value

    @method()
    def WriteValue(self, value: "ay", options: "a{sv}"):
        self._value = bytes(value)

    @method()
    def StartNotify(self):
        self._notifying = True

    @method()
    def StopNotify(self):
        self._notifying = False

    async def notify(self, value: bytes):
        self._value = value
        if not self._notifying:
            return
        msg = Message(
            destination=None,
            path=self.path,
            interface=PROP_IFACE,
            member="PropertiesChanged",
            signature="sa{sv}as",
            body=["org.bluez.GattCharacteristic1", {"Value": Variant("ay", self._value)}, []]
        )
        await self.bus.send(msg)


class Advertisement(ServiceInterface):
    def __init__(self, path: str, ad_type: str, local_name: str, service_uuids: List[str]):
        super().__init__("org.bluez.LEAdvertisement1")
        self.path = path
        self.ad_type = ad_type
        self.local_name = local_name
        self.service_uuids = service_uuids

    @dbus_property()
    def Type(self) -> "s":
        return self.ad_type

    @dbus_property()
    def LocalName(self) -> "s":
        return self.local_name

    @dbus_property()
    def ServiceUUIDs(self) -> "as":
        return self.service_uuids

    @method()
    def Release(self):
        pass


async def find_adapter_path(bus: MessageBus, preferred: str = "hci0") -> str:
    return f"/org/bluez/{preferred}"


async def register_app(bus: MessageBus, adapter_path: str, objmgr: ObjectManager):
    introspect = await bus.introspect(BLUEZ_SERVICE_NAME, adapter_path)
    obj = bus.get_proxy_object(BLUEZ_SERVICE_NAME, adapter_path, introspect)
    gatt_mgr = obj.get_interface(GATT_MANAGER_IFACE)
    await gatt_mgr.call_register_application(objmgr.path, {})


async def register_advertisement(bus: MessageBus, adapter_path: str, adv: Advertisement):
    introspect = await bus.introspect(BLUEZ_SERVICE_NAME, adapter_path)
    obj = bus.get_proxy_object(BLUEZ_SERVICE_NAME, adapter_path, introspect)
    adv_mgr = obj.get_interface(LE_ADV_MGR_IFACE)
    await adv_mgr.call_register_advertisement(adv.path, {})


def secure_flags(base, require_pairing: bool):
    if not require_pairing:
        return base
    flags = list(base)
    if "read" in flags and "encrypt-read" not in flags:
        flags.append("encrypt-read")
    if ("write" in flags or "write-without-response" in flags) and "encrypt-write" not in flags:
        flags.append("encrypt-write")
    if "notify" in flags and "encrypt-notify" not in flags:
        flags.append("encrypt-notify")
    return flags
