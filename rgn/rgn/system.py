"""Süreç ayarları — her betiğin başında `setup_process()` çağrılır.

    no_power_throttling   Windows: sürecin güç kısıtlamasını (EcoQoS) kapatır; arka plandaki CPU süreci aksi halde ~10 kat
                          yavaşlıyordu (AI_NEW_MODEL kural 2a; ölçüldü: ~10 adımdan sonra 55 -> ~550 ms).
    utf8_console          Windows konsolunda Türkçe karakterler bozulmasın.
"""
import sys


def no_power_throttling():
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 0x1, 0)          # PROCESS_POWER_THROTTLING_EXECUTION_SPEED kontrol edilir, durum 0 = kısıtlama yok
    return bool(k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


def utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def setup_process():
    utf8_console()
    no_power_throttling()
