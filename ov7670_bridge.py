"""
ov7670_bridge.py

Drop-in Python replacement for indrekluuk/ArduImageCapture (the Java tool).
Reads the same serial protocol emitted by LiveOV7670's ExampleUart.cpp
(EXAMPLE = 3 in setup.h) and reassembles full frames as numpy arrays,
then hands each completed frame to a callback of your choice (save it,
POST it to a backend, push it into a queue for FastAPI/Flask, etc).

No firmware changes needed -- the Arduino side already just streams a
byte protocol over UART; this script does what ImageCapture.java /
ImageFrame.java used to do, in Python.

Protocol (mirrors ArduImageCapture's Command.java / ImageCapture.java):

  Idle / pixel-stream mode:
    byte 0x00            -> start of a command
    any other byte        -> next raw pixel byte

  Command mode (right after a 0x00):
    byte[0]  = command length N          (N = payload bytes, excludes checksum)
    byte[1..N] = payload (payload[0] is commandCode | version-nibble 0x10)
    byte[N+1] = checksum (XOR of payload bytes)

  Command codes (low nibble, version nibble 0x10 always present):
    0x01 NEW_FRAME  payload = [code, width_lo8, height_lo8, packed]
        packed bit0-1 = width  bits 8-9
        packed bit2-3 = height bits 8-9
        packed bit4-7 = pixel format (1 = RGB565, 2 = grayscale), each
                        stored with a parity bit for basic error detection
    0x03 DEBUG_DATA payload = [code, ...utf8 text]

Pixel encoding (both formats reserve 0x00 for the command marker and
carry a parity bit, see CameraOV7670/ExampleUart.cpp formatRgbPixelByteH/L
and formatPixelByteGrayscale* for the original encode side):
  RGB565:      2 bytes/pixel, H byte then L byte, each parity-checked.
  Grayscale:   2 bytes on the wire per pixel pair (YUV422 luma channel is
               parity-checked pairwise), 1 gray value out per valid byte.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import serial

# ---- protocol constants (from Command.java / ImageCapture.java) ----------

START_COMMAND = 0x00
COMMAND_VERSION = 0x10
COMMAND_MASK = 0x0F
COMMAND_NEW_FRAME = 0x01
COMMAND_DEBUG_DATA = 0x03

H_BYTE_PARITY_CHECK = 0b00100000
H_BYTE_PARITY_INVERT = 0b00001000
L_BYTE_PARITY_CHECK = 0b00001000
L_BYTE_PARITY_INVERT = 0b00100000

PIXEL_FORMAT_RGB565 = 1
PIXEL_FORMAT_GRAYSCALE = 2

MAX_W, MAX_H = 640, 480


def _rgb565_from_hl(h: int, l: int) -> tuple[int, int, int]:
    raw = (h << 8) | l
    r = (raw >> 8) & 0xF8
    g = (raw >> 3) & 0xFC
    b = (raw << 3) & 0xF8
    return r, g, b


def _is_rgb_high_byte(b: int) -> bool:
    return bool(b & H_BYTE_PARITY_CHECK) != bool(b & H_BYTE_PARITY_INVERT)


def _is_rgb_low_byte(b: int) -> bool:
    return bool(b & L_BYTE_PARITY_CHECK) == bool(b & L_BYTE_PARITY_INVERT)


def _is_gray_first_byte(b: int) -> bool:
    return (b & 1) == 0


@dataclass
class Frame:
    width: int
    height: int
    pixels: np.ndarray  # (H, W, 3) uint8, RGB


class OV7670FrameAssembler:
    """State machine: raw serial bytes in, complete Frame objects out."""

    def __init__(self, on_frame: Callable[[Frame], None],
                 on_debug: Optional[Callable[[str], None]] = None):
        self._on_frame = on_frame
        self._on_debug = on_debug or (lambda msg: None)

        self._in_command = False
        self._cmd_len = -1
        self._cmd_buf = bytearray()

        self._pixel_format = PIXEL_FORMAT_RGB565
        self._pixel_byte_buf = bytearray()

        self._width = 160
        self._height = 120
        self._buf: Optional[np.ndarray] = None
        self._row = 0
        self._col = 0

        self._start_new_frame(self._width, self._height, self._pixel_format)

    # -- public API ---------------------------------------------------

    def feed(self, data: bytes) -> None:
        for b in data:
            self._feed_byte(b)

    # -- internals ------------------------------------------------------

    def _start_new_frame(self, w: int, h: int, pixel_format: int) -> None:
        w = max(1, min(w, MAX_W))
        h = max(1, min(h, MAX_H))
        self._width, self._height = w, h
        self._pixel_format = pixel_format
        self._buf = np.zeros((h, w, 3), dtype=np.uint8)
        self._row = 0
        self._col = 0
        self._pixel_byte_buf.clear()

    def _feed_byte(self, b: int) -> None:
        if not self._in_command:
            if b == START_COMMAND:
                self._in_command = True
                self._cmd_len = -1
                self._cmd_buf.clear()
                self._pixel_byte_buf.clear()  # a stray partial pixel is discarded
            else:
                self._feed_pixel_byte(b)
            return

        # -- inside a command --
        if self._cmd_len < 0:
            self._cmd_len = b
            return

        self._cmd_buf.append(b)
        if len(self._cmd_buf) >= self._cmd_len + 1:  # +1 for checksum
            self._process_command()
            self._in_command = False

    def _process_command(self) -> None:
        payload = bytes(self._cmd_buf[: self._cmd_len])
        checksum = self._cmd_buf[self._cmd_len]

        calc = 0
        for x in payload:
            calc ^= x
        if calc != checksum:
            self._on_debug("Checksum failed -- check baud rate / firmware version")
            return

        if not payload:
            return

        command_byte = payload[0]
        code = command_byte & COMMAND_MASK

        if code == COMMAND_NEW_FRAME:
            self._handle_new_frame(payload)
        elif code == COMMAND_DEBUG_DATA:
            self._on_debug(payload[1:].decode("utf-8", errors="replace"))
        else:
            self._on_debug(f"Unknown command code {code}")

    def _handle_new_frame(self, payload: bytes) -> None:
        if len(payload) != 4:
            self._on_debug(f"Invalid NEW_FRAME length {len(payload)}")
            return

        # if the previous frame was mid-flight, emit whatever we had before resetting
        if self._buf is not None and self._row > 0:
            self._emit_frame()

        w = (payload[1] & 0xFF) | ((payload[3] & 0x03) << 8)
        h = (payload[2] & 0xFF) | ((payload[3] & 0x0C) << 6)
        pixel_format = (payload[3] & 0xF0) >> 4
        self._start_new_frame(w, h, pixel_format)

    def _feed_pixel_byte(self, b: int) -> None:
        self._pixel_byte_buf.append(b)

        if self._pixel_format == PIXEL_FORMAT_RGB565:
            if len(self._pixel_byte_buf) < 2:
                return
            h_byte, l_byte = self._pixel_byte_buf[0], self._pixel_byte_buf[1]
            if _is_rgb_high_byte(h_byte) and _is_rgb_low_byte(l_byte):
                self._put_pixel(*_rgb565_from_hl(h_byte, l_byte))
                self._pixel_byte_buf.clear()
            elif not _is_rgb_high_byte(h_byte):
                # first byte was actually a low byte -> resync
                r, g, b2 = _rgb565_from_hl(0, h_byte)
                self._put_pixel(0, 0, b2)  # only blue channel trustworthy
                self._pixel_byte_buf = bytearray([l_byte])
            else:
                r, g, b2 = _rgb565_from_hl(h_byte, 0)
                self._put_pixel(r, g, 0)  # only red/green trustworthy
                self._pixel_byte_buf = bytearray([l_byte])

        else:  # grayscale, parity-checked, byte-pairwise
            if len(self._pixel_byte_buf) < 2:
                return
            b1, b2 = self._pixel_byte_buf[0], self._pixel_byte_buf[1]
            self._pixel_byte_buf.clear()
            if _is_gray_first_byte(b1):
                self._put_pixel(b1, b1, b1)
                if not _is_gray_first_byte(b2):
                    self._put_pixel(b2, b2, b2)
                else:
                    self._pixel_byte_buf.append(b2)  # b2 starts next pair
            else:
                # b1 was a stray "second" byte -- drop it, resync on b2
                self._put_pixel(b1, b1, b1)
                self._pixel_byte_buf.append(b2)

    def _put_pixel(self, r: int, g: int, b: int) -> None:
        if self._buf is None:
            return
        if self._row < self._height:
            self._buf[self._row, self._col] = (r, g, b)
            self._col += 1
        if self._col >= self._width:
            self._col = 0
            self._row += 1
            if self._row >= self._height:
                self._emit_frame()
                self._row = 0
                self._buf = np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def _emit_frame(self) -> None:
        if self._buf is not None:
            self._on_frame(Frame(self._width, self._height, self._buf.copy()))


class SerialFrameSource:
    """Reads a serial port in a background thread and feeds an assembler."""

    def __init__(self, port: str, baud: int,
                 on_frame: Callable[[Frame], None],
                 on_debug: Optional[Callable[[str], None]] = None):
        self._ser = serial.Serial(port, baudrate=baud, timeout=0.1)
        self._assembler = OV7670FrameAssembler(on_frame, on_debug)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self._ser.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            data = self._ser.read(4096)
            if data:
                self._assembler.feed(data)
            else:
                time.sleep(0.001)


# ---------------------------------------------------------------------------
# Example: route each captured frame to a Python backend over HTTP.
# Swap send_to_backend() for whatever your backend actually expects
# (a websocket push, a shared queue if the backend runs in-process, etc).
# ---------------------------------------------------------------------------

def send_to_backend(frame: Frame, backend_url: str) -> None:
    import io
    import requests
    from PIL import Image

    img = Image.fromarray(frame.pixels, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    try:
        requests.post(
            backend_url,
            files={"frame": ("frame.jpg", buf, "image/jpeg")},
            data={"width": frame.width, "height": frame.height},
            timeout=1,
        )
    except requests.RequestException as e:
        print(f"[bridge] failed to send frame to backend: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="e.g. /dev/ttyUSB0 or COM3")
    parser.add_argument("--baud", type=int, default=500000,
                         help="must match UART_MODE's baud in ExampleUart.cpp")
    parser.add_argument("--backend-url", default="http://localhost:8000/frame",
                         help="Python backend endpoint to POST each frame to")
    args = parser.parse_args()

    def handle_frame(frame: Frame) -> None:
        print(f"[bridge] frame {frame.width}x{frame.height}")
        send_to_backend(frame, args.backend_url)

    def handle_debug(msg: str) -> None:
        print(f"[arduino] {msg}")

    source = SerialFrameSource(args.port, args.baud, handle_frame, handle_debug)
    source.start()
    print(f"[bridge] listening on {args.port} @ {args.baud} baud, forwarding to {args.backend_url}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        source.stop()
