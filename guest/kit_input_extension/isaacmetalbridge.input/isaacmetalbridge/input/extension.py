"""Forward native macOS viewer events to Carbonite's real Kit input devices."""

from __future__ import annotations

import os
import struct
from collections import deque

import carb
import carb.input
import omni.appwindow
import omni.ext
import omni.kit.app
import omni.kit.commands
import omni.usd


_RECORD = struct.Struct("<IHHQffffIIII")
_MAGIC = 0x31494D49
_VERSION = 1
_MAX_RECORDS_PER_UPDATE = 512

_MOUSE_EVENT_TYPES = {
    1: carb.input.MouseEventType.MOVE,
    2: carb.input.MouseEventType.LEFT_BUTTON_DOWN,
    3: carb.input.MouseEventType.LEFT_BUTTON_UP,
    4: carb.input.MouseEventType.RIGHT_BUTTON_DOWN,
    5: carb.input.MouseEventType.RIGHT_BUTTON_UP,
    6: carb.input.MouseEventType.MIDDLE_BUTTON_DOWN,
    7: carb.input.MouseEventType.MIDDLE_BUTTON_UP,
    8: carb.input.MouseEventType.SCROLL,
}

# macOS virtual key codes emitted by NSEvent, mapped to Carbonite's platform-neutral keys.
_KEY_NAMES = {
    0: "A", 1: "S", 2: "D", 3: "F", 4: "H", 5: "G", 6: "Z", 7: "X", 8: "C", 9: "V",
    11: "B", 12: "Q", 13: "W", 14: "E", 15: "R", 16: "Y", 17: "T",
    18: "KEY_1", 19: "KEY_2", 20: "KEY_3", 21: "KEY_4", 22: "KEY_6", 23: "KEY_5",
    24: "EQUAL", 25: "KEY_9", 26: "KEY_7", 27: "MINUS", 28: "KEY_8", 29: "KEY_0",
    30: "RIGHT_BRACKET", 31: "O", 32: "U", 33: "LEFT_BRACKET", 34: "I", 35: "P",
    36: "ENTER", 37: "L", 38: "J", 39: "APOSTROPHE", 40: "K", 41: "SEMICOLON",
    42: "BACKSLASH", 43: "COMMA", 44: "SLASH", 45: "N", 46: "M", 47: "PERIOD",
    48: "TAB", 49: "SPACE", 50: "GRAVE_ACCENT", 51: "BACKSPACE", 53: "ESCAPE",
    55: "LEFT_SUPER", 56: "LEFT_SHIFT", 57: "CAPS_LOCK", 58: "LEFT_ALT", 59: "LEFT_CONTROL",
    60: "RIGHT_SHIFT", 61: "RIGHT_ALT", 62: "RIGHT_CONTROL",
    65: "NUMPAD_DEL", 67: "NUMPAD_MULTIPLY", 69: "NUMPAD_ADD", 71: "NUM_LOCK",
    75: "NUMPAD_DIVIDE", 76: "NUMPAD_ENTER", 78: "NUMPAD_SUBTRACT", 81: "NUMPAD_EQUAL",
    82: "NUMPAD_0", 83: "NUMPAD_1", 84: "NUMPAD_2", 85: "NUMPAD_3", 86: "NUMPAD_4",
    87: "NUMPAD_5", 88: "NUMPAD_6", 89: "NUMPAD_7", 91: "NUMPAD_8", 92: "NUMPAD_9",
    96: "F5", 97: "F6", 98: "F7", 99: "F3", 100: "F8", 101: "F9", 103: "F11",
    109: "F10", 111: "F12", 114: "INSERT", 115: "HOME", 116: "PAGE_UP", 117: "DEL",
    118: "F4", 119: "END", 120: "F2", 121: "PAGE_DOWN", 122: "F1",
    123: "LEFT", 124: "RIGHT", 125: "DOWN", 126: "UP",
}


class InputForwardingExtension(omni.ext.IExt):
    """Read append-only viewer records and feed the real Kit input provider."""

    def on_startup(self, ext_id: str) -> None:
        del ext_id
        self._path = os.environ.get("IMB_INPUT_FILE", "")
        self._trace = os.environ.get("IMB_INPUT_TRACE", "") not in ("", "0")
        self._test_reference_url = os.environ.get("IMB_TEST_CREATE_REFERENCE_URL", "")
        self._test_reference_updates = max(
            int(os.environ.get("IMB_TEST_CREATE_REFERENCE_AFTER_UPDATES", "120")),
            1,
        )
        self._offset = 0
        self._last_sequence = 0
        self._pending = deque()
        self._provider = carb.input.acquire_input_provider()
        self._mouse = None
        self._keyboard = None
        self._subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="isaacmetalbridge.input"
        )
        if self._path:
            carb.log_info(f"isaacmetalbridge.input: forwarding events from {self._path}")
            if self._trace:
                carb.log_warn(f"isaacmetalbridge.input: trace enabled for {self._path}")
        else:
            carb.log_warn("isaacmetalbridge.input: IMB_INPUT_FILE is unset; input forwarding is disabled")
        if self._test_reference_url:
            carb.log_warn(
                "isaacmetalbridge.input: scheduled diagnostic CreateReferenceCommand "
                f"after {self._test_reference_updates} updates: {self._test_reference_url}"
            )

    def on_shutdown(self) -> None:
        self._subscription = None
        self._provider = None
        self._mouse = None
        self._keyboard = None
        self._pending.clear()

    def _on_update(self, _event) -> None:
        if self._test_reference_url:
            usd_context = omni.usd.get_context()
            if usd_context.get_stage() is not None:
                self._test_reference_updates -= 1
            if usd_context.get_stage() is not None and self._test_reference_updates <= 0:
                asset_path = self._test_reference_url
                self._test_reference_url = ""
                carb.log_warn(
                    f"isaacmetalbridge.input: executing diagnostic CreateReferenceCommand: {asset_path}"
                )
                try:
                    omni.kit.commands.execute(
                        "CreateReferenceCommand",
                        usd_context=usd_context,
                        path_to="/FlatGrid",
                        asset_path=asset_path,
                        instanceable=False,
                    )
                except Exception as error:
                    carb.log_error(
                        "isaacmetalbridge.input: diagnostic CreateReferenceCommand failed: "
                        f"{error}"
                    )
        if not self._path:
            return
        try:
            size = os.path.getsize(self._path)
            if size < self._offset:
                self._offset = 0
                self._last_sequence = 0
                self._pending.clear()
            available = size - self._offset
            record_count = min(available // _RECORD.size, _MAX_RECORDS_PER_UPDATE)
            if record_count > 0:
                byte_count = record_count * _RECORD.size
                with open(self._path, "rb", buffering=0) as input_file:
                    input_file.seek(self._offset)
                    payload = input_file.read(byte_count)
                self._offset += len(payload) - (len(payload) % _RECORD.size)
                for start in range(0, len(payload) - _RECORD.size + 1, _RECORD.size):
                    record = _RECORD.unpack_from(payload, start)
                    if self._pending and self._pending[-1][2] == 1 and record[2] == 1:
                        self._pending[-1] = record
                    else:
                        self._pending.append(record)
            # Kit UI needs press and release to be distributed on separate updates.
            if self._pending:
                self._dispatch(self._pending.popleft())
        except FileNotFoundError:
            return
        except Exception as error:
            carb.log_error(f"isaacmetalbridge.input: failed reading input: {error}")

    def _dispatch(self, values) -> None:
        magic, version, kind, sequence, x, y, delta_x, delta_y, code, modifiers, width, height = values
        if magic != _MAGIC or version != _VERSION or sequence <= self._last_sequence:
            return
        self._last_sequence = sequence
        if self._mouse is None or self._keyboard is None:
            app_window = omni.appwindow.get_default_app_window()
            self._mouse = app_window.get_mouse()
            self._keyboard = app_window.get_keyboard()

        if kind in _MOUSE_EVENT_TYPES:
            if kind == 8:
                pixel = (delta_x, delta_y)
                normalized = (
                    delta_x / max(float(width), 1.0),
                    delta_y / max(float(height), 1.0),
                )
            else:
                pixel = (x, y)
                normalized = (
                    x / max(float(width), 1.0),
                    y / max(float(height), 1.0),
                )
            self._provider.buffer_mouse_event(
                self._mouse, _MOUSE_EVENT_TYPES[kind], normalized, modifiers, pixel
            )
        elif kind in (9, 10):
            key_name = _KEY_NAMES.get(code)
            key = carb.input.KeyboardInput.__members__.get(key_name) if key_name else None
            if key is not None:
                event_type = (
                    carb.input.KeyboardEventType.KEY_PRESS
                    if kind == 9
                    else carb.input.KeyboardEventType.KEY_RELEASE
                )
                self._provider.buffer_keyboard_key_event(self._keyboard, event_type, key, modifiers)
        elif kind == 11:
            try:
                self._provider.buffer_keyboard_char_event(self._keyboard, chr(code), modifiers)
            except (ValueError, OverflowError):
                return

        if self._trace:
            carb.log_warn(
                f"isaacmetalbridge.input: event seq={sequence} kind={kind} "
                f"pos=({x:.1f},{y:.1f}) code={code} modifiers={modifiers}"
            )
