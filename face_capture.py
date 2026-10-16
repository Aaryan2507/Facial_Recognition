"""
face_capture.py

Reads frames straight from the LiveOV7670 camera (via ov7670_bridge.py's
serial decoder) and flags "a face is probably here" on each one, in-process
-- no HTTP hop, no separate backend needed.

Detection here is a deliberately simple skin-color heuristic (see
detect_skin_blob below) instead of a compiled face-detection library like
OpenCV's Haar cascades or MediaPipe. Those need a native wheel built for
your exact Python version, and brand-new Python releases (like 3.14) often
don't have one yet -- MediaPipe currently doesn't even ship 3.13 wheels.
This detector only needs NumPy + Pillow, both of which have far broader
Python-version support, so there's no wheel-compatibility gamble at all.
The optional live preview window below uses tkinter for the same reason --
it ships with Python itself, no separate native wheel to worry about.

Trade-off: detection is crude. It flags "is there a large patch of
skin-toned pixels in frame", not "is there specifically a face". Skin-toned
walls, wood, cardboard, hands, etc. will trigger it too. Given accuracy
doesn't matter yet, that's an acceptable place to start -- and precisely
because it's this simple, swapping in a real model later only means
replacing detect_skin_blob() with something else; nothing else changes.

Behavior:
  - While nothing's been found: overwrite latest.jpg every frame,
    same as before, so you can watch the live feed on disk.
  - With --preview: also show a live window on the PC of what the camera
    is currently seeing, updated every frame.
  - The instant the heuristic fires: stop touching latest.jpg, save the
    triggering frame under a unique filename, and stop (closing the
    preview window if one was open).

Requires: pip install pillow numpy
(ov7670_bridge.py must be in the same folder -- this imports it directly)
(--preview requires tkinter, which ships with standard Python installs
 on Windows/macOS; on some Linux distros it's a separate package, e.g.
 `sudo apt install python3-tk`)
"""

from __future__ import annotations

import argparse
import queue
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

from ov7670_bridge import Frame, SerialFrameSource


def detect_skin_blob(rgb: np.ndarray, min_fraction: float = 0.12) -> bool:
    """
    Very simple, widely-used RGB skin-color rule (no ML, no compiled lib).
    Returns True if enough of the frame looks skin-colored.

    rgb: (H, W, 3) uint8 array, R/G/B in that channel order.
    min_fraction: fraction of the frame that must be "skin" to count as a hit.
                  Lower = more sensitive (more false positives). Tune freely.
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    skin_mask = (
        (r > 95) & (g > 40) & (b > 20)
        & ((r - b) > 15)
        & (np.abs(r - g) > 15)
        & (r > g) & (r > b)
    )

    skin_fraction = float(skin_mask.mean())
    return skin_fraction >= min_fraction


class PreviewWindow:
    """
    Minimal live-preview window. Runs on the main thread; frames are pushed
    in from the serial-reader background thread via a 1-slot queue so the
    UI always shows the newest frame and never backs up.
    """

    def __init__(self, scale: int = 2):
        self.scale = scale
        self._queue: "queue.Queue[Image.Image]" = queue.Queue(maxsize=1)
        self._photo = None  # keep a reference alive, tkinter needs this
        self.closed = False

        self.root = tk.Tk()
        self.root.title("LiveOV7670 preview")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self.closed = True

    def push_frame(self, pil_image: Image.Image) -> None:
        """Call from any thread. Drops the previous unshown frame if any."""
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(pil_image)
        except queue.Full:
            pass

    def _poll(self) -> None:
        try:
            img = self._queue.get_nowait()
            if self.scale != 1:
                img = img.resize(
                    (img.width * self.scale, img.height * self.scale), Image.NEAREST
                )
            self._photo = ImageTk.PhotoImage(img)
            self.label.configure(image=self._photo)
        except queue.Empty:
            pass

    def run_until(self, should_stop) -> None:
        """Blocks (pumping the Tk event loop) until should_stop() is True
        or the window is closed by the user."""
        while not self.closed and not should_stop():
            self._poll()
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.02)
        if not self.closed:
            self.root.destroy()


class FaceCapture:
    def __init__(self, output_dir: Path, min_fraction: float = 0.12):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.output_dir / "latest.jpg"
        self.min_fraction = min_fraction
        self.face_found = False

    def handle_frame(self, frame: Frame) -> None:
        if self.face_found:
            return  # already done, ignore further frames

        hit = detect_skin_blob(frame.pixels, self.min_fraction)

        if not hit:
            Image.fromarray(frame.pixels, mode="RGB").save(self.latest_path)
            print(f"[face] no hit yet, {frame.width}x{frame.height} -> {self.latest_path}")
            return

        # Hit -- stop overwriting latest.jpg, save a unique capture instead
        self.face_found = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        capture_path = self.output_dir / f"face_{timestamp}.jpg"
        Image.fromarray(frame.pixels, mode="RGB").save(capture_path)
        print(f"[face] FACE-LIKE REGION DETECTED -- saved {capture_path}")


def handle_debug(msg: str) -> None:
    print(f"[arduino] {msg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="e.g. COM10 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=500000)
    parser.add_argument("--output-dir", default="captures", help="where images are saved")
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.12,
        help="fraction of frame that must look skin-colored to count as a hit "
             "(lower = more sensitive / more false positives). Default 0.12",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="show a live window on the PC of what the camera currently sees",
    )
    parser.add_argument(
        "--preview-scale", type=int, default=2,
        help="integer upscale factor for the preview window (default 2, since "
             "160x120/320x240 is tiny at 1:1 on a modern monitor)",
    )
    args = parser.parse_args()

    capture = FaceCapture(Path(args.output_dir), args.sensitivity)
    preview = PreviewWindow(args.preview_scale) if args.preview else None

    def on_frame(frame: Frame) -> None:
        capture.handle_frame(frame)
        if preview is not None:
            preview.push_frame(Image.fromarray(frame.pixels, mode="RGB"))

    source = SerialFrameSource(args.port, args.baud, on_frame, handle_debug)
    source.start()
    print(f"[bridge] listening on {args.port} @ {args.baud} baud, watching for a face...")

    try:
        if preview is not None:
            preview.run_until(lambda: capture.face_found)
        else:
            while not capture.face_found:
                time.sleep(0.1)
        # give the last frame a moment to finish writing, then stop cleanly
        time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        print("[bridge] stopped.")
