"""Right-arrow/ESC terminal keyboard control for the deployment client.

Ported from act-triple-plus-aloha2/aloha_scripts/one_side_teleop.py's KeyboardControl, trimmed
to only the two events this client needs (right arrow to start, ESC/q to cancel). No ROS
dependency - just termios/tty/select on the controlling terminal.
"""

import os
import select
import sys
import termios
import time
import tty


class UserAborted(Exception):
    """Raised when the operator pressed ESC/q to cancel the run."""


class KeyboardControl:
    """Arrow-key / ESC reader on the controlling terminal, polled without blocking.

    cbreak mode, not raw: Ctrl+C keeps working. Right arrow starts, ESC (or q) cancels.
    When stdin is not a tty there is nothing to read from, so `enabled` is False and the
    caller should skip waiting for a keypress and just proceed.
    """

    # A lone ESC and the start of an arrow-key sequence look identical until the next bytes
    # arrive, so an unaccompanied ESC only counts once nothing followed it for this long.
    ESC_HOLDOFF_S = 0.05

    def __init__(self):
        self.enabled = False
        self._fd = None
        self._saved = None
        self._buffer = b""
        self._buffer_time = 0.0
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._saved = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self.enabled = True
        except (termios.error, ValueError, OSError) as exc:
            print(f"Keyboard control unavailable ({exc}).")

    def poll(self):
        """Return the key events seen since the last call: 'right', 'esc' - zero or more,
        in the order the corresponding bytes arrived."""
        if not self.enabled:
            return []
        while select.select([self._fd], [], [], 0)[0]:
            chunk = os.read(self._fd, 64)
            if not chunk:
                break
            self._buffer += chunk
            self._buffer_time = time.time()
        return self._parse()

    def _parse(self):
        """Drain the byte buffer into key events, leaving any incomplete sequence behind."""
        events = []
        while self._buffer:
            if not self._buffer.startswith(b"\x1b"):
                key = self._buffer[:1]
                self._buffer = self._buffer[1:]
                if key in (b"q", b"Q"):
                    events.append("esc")
                continue
            # CSI ("\x1b[C") and SS3 ("\x1bOC") both show up depending on terminal mode
            if len(self._buffer) >= 3 and self._buffer[1:2] in (b"[", b"O"):
                code = self._buffer[2:3]
                self._buffer = self._buffer[3:]
                if code == b"C":
                    events.append("right")
                continue
            if len(self._buffer) == 1 and time.time() - self._buffer_time > self.ESC_HOLDOFF_S:
                self._buffer = b""
                events.append("esc")
                continue
            break  # incomplete sequence: leave it buffered for the next poll
        return events

    def drain(self):
        """Throw away anything typed so far (e.g. before asking for a fresh keypress)."""
        self.poll()
        self._buffer = b""

    def close(self):
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None
        self.enabled = False
