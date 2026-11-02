"""
Real-time BLE pipeline: collect dual MAX30102 PPG over BLE, periodically run the
fusion detector, and drive the ESP32 buzzer from the detection result.

  * Samples arrive over BLE (same packet format / CSV format as dual_ppg_ble.py).
  * Every --interval seconds run_all_fusion.py runs on the latest flushed CSV.
  * Detection says DROWSY            -> buzzer ON (and LED ON with --led).
  * Detection changes to not-drowsy  -> buzzer OFF.
  * Buzzer is also turned OFF on shutdown (Ctrl+C) or if the detector gives no
    recognisable result is NOT treated as a change (state is held).

Stop with Ctrl+C: a final detection pass runs, then the buzzer is switched off.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

from bleak import BleakClient, BleakScanner

ROOT = Path(__file__).resolve().parent
DETECTOR = ROOT / "run_all_fusion.py"
LOGGER = logging.getLogger(__name__)
PROFILE_ENABLED = False

# ---------------- BLE configuration (matches dual_ppg_ble.py) ----------------
DEVICE_NAME = "DualPPG-ESP32"
PPG_CHARACTERISTIC_UUID = "7b2e0002-6f4a-4b9a-9c2d-3e5f6a7b8c01"
ALERT_CHARACTERISTIC_UUID = "7b2e0003-6f4a-4b9a-9c2d-3e5f6a7b8c01"
STATUS_CHARACTERISTIC_UUID = "7b2e0004-6f4a-4b9a-9c2d-3e5f6a7b8c01"

HEADER = ["Clock Time", "Elapsed_s", "Device Timestamp", "IR1", "Red1", "IR2", "Red2"]

# Default patterns used to read the detector's console output.
# The awake pattern is checked first on each line, so "not drowsy" is awake.
DEFAULT_DROWSY_PATTERN = r"drows|fatigue"
DEFAULT_AWAKE_PATTERN = r"not[ _-]?drows|non[ _-]?drows|awake|normal"


def configure_logging(log_file: Path, level: str) -> None:
    """Configure console and persistent logging for the realtime pipeline."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler])
    LOGGER.info("Logging initialized: file=%s file_level=DEBUG requested_level=%s",
                log_file, level.upper())


def _profile_calls(frame, event, _arg) -> None:
    """Log calls and returns for code defined in this module."""
    if not PROFILE_ENABLED or frame.f_code.co_filename != __file__:
        return
    function = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
    if event == "call":
        LOGGER.debug("CALL %s", function)
    elif event == "return":
        LOGGER.debug("RETURN %s", function)


def start_call_logging() -> None:
    """Enable function call logging for this process."""
    global PROFILE_ENABLED
    PROFILE_ENABLED = True
    sys.setprofile(_profile_calls)
    LOGGER.info("Function call logging enabled")


def stop_call_logging() -> None:
    """Disable function call logging for this process."""
    global PROFILE_ENABLED
    PROFILE_ENABLED = False
    sys.setprofile(None)
    LOGGER.info("Function call logging disabled")


# ---------------------------- buzzer control ----------------------------
class AlertController:
    """Sends buzzer/LED state to the ESP32 and only writes on state changes."""

    def __init__(self, client: BleakClient, use_led: bool):
        self.client = client
        self.use_led = use_led
        self.active = False
        self._lock = asyncio.Lock()

    async def _write(self, on: bool) -> None:
        LOGGER.info("Writing alert state: buzzer=%s led=%s", on, on and self.use_led)
        await self.client.write_gatt_char(
            ALERT_CHARACTERISTIC_UUID,
            bytes((int(on), int(on and self.use_led))),
            response=True,
        )

    async def set(self, on: bool, force: bool = False) -> None:
        LOGGER.info("Requested alert state: on=%s force=%s current=%s",
                    on, force, self.active)
        async with self._lock:
            if on == self.active and not force:
                LOGGER.debug("Alert state unchanged; skipping BLE write")
                return
            if not self.client.is_connected:
                LOGGER.error("BLE disconnected; cannot update buzzer")
                return
            try:
                await self._write(on)
            except Exception as exc:  # BLE write can fail transiently
                LOGGER.exception("Failed to set buzzer %s", "ON" if on else "OFF")
                return
            self.active = on
            LOGGER.info("Alert state changed: buzzer=%s", on)


# ---------------------------- CSV recording ----------------------------
class Recorder:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        LOGGER.info("Opening recording file: %s", csv_path)
        self.file = csv_path.open("w", newline="", buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(HEADER)
        self.file.flush()
        self.sample_count = 0
        self.first_ts: int | None = None
        self.elapsed = 0.0

    def on_ppg(self, _sender, data: bytearray) -> None:
        if len(data) != 20:
            LOGGER.warning("Invalid BLE PPG packet length: %s", len(data))
            return
        ts, ir1, red1, ir2, red2 = struct.unpack("<5I", data)
        if self.first_ts is None:
            self.first_ts = ts
        self.elapsed = (ts - self.first_ts) / 1000.0
        self.writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"{self.elapsed:.3f}", ts, ir1, red1, ir2, red2,
        ])
        self.file.flush()
        self.sample_count += 1
        LOGGER.debug("PPG sample recorded: count=%s timestamp=%s elapsed=%.3fs",
                     self.sample_count, ts, self.elapsed)

    def close(self) -> None:
        LOGGER.info("Closing recording file after %s samples", self.sample_count)
        self.file.close()


# ---------------------------- detection ----------------------------
async def run_detection(csv_path: Path, output_root: Path) -> tuple[int, str]:
    """Run run_all_fusion.py without blocking the BLE event loop."""
    LOGGER.info("Starting detector: input=%s output=%s", csv_path, output_root)
    env = os.environ.copy()
    env["FUSION_INPUT_FILE"] = csv_path.name
    env["FUSION_PPG_DIR"] = str(csv_path.parent)
    env["FUSION_OUT_ROOT"] = str(output_root)
    env["FUSION_REUSE"] = "0"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(DETECTOR),
        cwd=str(ROOT), env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output_lines = []
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="replace")
        output_lines.append(decoded_line)
        LOGGER.info("Detector: %s", decoded_line.rstrip())
        if decoded_line.startswith("DISPLAY: "):
            print(decoded_line[len("DISPLAY: "):].rstrip())
    await proc.wait()
    text = "".join(output_lines)
    LOGGER.info("Detector completed: return_code=%s output_chars=%s",
                proc.returncode, len(text))
    LOGGER.debug("Detector output:\n%s", text.rstrip())
    return proc.returncode or 0, text


def classify(text: str, drowsy_re: re.Pattern, awake_re: re.Pattern) -> bool | None:
    """Return the buzzer state from the final fused detector state."""
    fused_states = re.findall(
        r"\bFused:\s*(DROWSY|ALERT|SUSPECT|INSUFFICIENT[_ -]?DATA|"
        r"NOT[ _-]?DROWSY)\b",
        text,
        re.IGNORECASE,
    )
    if fused_states:
        fused_state = fused_states[-1].upper().replace("-", "_").replace(" ", "_")
        drowsy = fused_state == "DROWSY"
        LOGGER.info("Classification from fused state %s: %s", fused_state, drowsy)
        return drowsy

    state = None
    for line in text.splitlines():
        if awake_re.search(line):
            state = False
            LOGGER.debug("Classification matched awake pattern: %s", line)
        elif drowsy_re.search(line):
            state = True
            LOGGER.debug("Classification matched drowsy pattern: %s", line)
    LOGGER.info("Classification result: %s", state)
    return state


async def detect_and_alert(recorder: Recorder, alert: AlertController,
                           output_dir: Path, drowsy_re, awake_re) -> None:
    LOGGER.info("Starting detection pass at recording time %.1fs", recorder.elapsed)
    LOGGER.info("Running detection at recording time %.1fs", recorder.elapsed)
    print(f"Detection started at recording time {recorder.elapsed:.1f}s")
    code, text = await run_detection(recorder.csv_path, output_dir)
    if code != 0:
        LOGGER.error("Detector exited with code %s; alert state unchanged", code)
        return
    drowsy = classify(text, drowsy_re, awake_re)
    if drowsy is None:
        LOGGER.warning("No drowsy/awake result recognized; alert state unchanged")
        return
    LOGGER.info("Detection result: %s", "DROWSY" if drowsy else "NOT DROWSY")
    # Drowsy -> buzzer on. Any change away from drowsy -> buzzer off.
    await alert.set(drowsy)


async def run_recording_detection(csv_path: Path, output_dir: Path,
                                  drowsy_re, awake_re) -> int:
    """Run one final detection pass for an existing recording without BLE."""
    LOGGER.info("Starting offline detection for recording: %s", csv_path)
    LOGGER.info("Running final detection for recording: %s", csv_path)
    print(f"Detection started for recording: {csv_path}")
    code, text = await run_detection(csv_path, output_dir)
    if code != 0:
        LOGGER.error("Offline detector exited with code %s", code)
        return code

    drowsy = classify(text, drowsy_re, awake_re)
    if drowsy is None:
        LOGGER.error("Offline detector produced no recognisable drowsy/awake result")
        return 2

    result = "DROWSY" if drowsy else "NOT DROWSY"
    LOGGER.info("Offline detection result: %s", result)
    return 0


# ---------------------------- main ----------------------------
async def run(args) -> int:
    LOGGER.info("Pipeline starting with arguments: %s", vars(args))
    drowsy_re = re.compile(args.drowsy_pattern, re.IGNORECASE)
    awake_re = re.compile(args.awake_pattern, re.IGNORECASE)

    if args.recording_csv is not None:
        csv_path = args.recording_csv.expanduser().resolve()
        if not csv_path.is_file():
            LOGGER.error("Recording CSV does not exist: %s", csv_path)
            return 1
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return await run_recording_detection(csv_path, args.output_dir,
                                             drowsy_re, awake_re)

    ppg_dir = ROOT / "PPG_reading"
    ppg_dir.mkdir(exist_ok=True)
    csv_path = ppg_dir / args.recording
    if csv_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = ppg_dir / f"{csv_path.stem}_{stamp}{csv_path.suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Searching for BLE device %r", args.device)
    print(f"Searching for BLE device {args.device!r}...")
    device = await BleakScanner.find_device_by_name(args.device, timeout=30)
    if device is None:
        LOGGER.error("BLE device %r was not found", args.device)
        print(f"BLE device {args.device!r} not found")
        return 1

    recorder = Recorder(csv_path)
    LOGGER.info("Recording to: %s", csv_path)
    LOGGER.info("Detection output: %s", args.output_dir)
    LOGGER.info("Press Ctrl+C to stop.")

    try:
        async with BleakClient(device) as client:
            LOGGER.info("Connected to BLE device: %s", device.name)
            print(f"Bluetooth connected: {device.name}")
            alert = AlertController(client, args.led)

            def on_status(_sender, data):
                if len(data) == 2:
                    LOGGER.info("BLE status: buzzer=%s led=%s", bool(data[0]), bool(data[1]))
                else:
                    LOGGER.warning("Invalid BLE status packet length: %s", len(data))

            LOGGER.info("Starting PPG and status notifications")
            await client.start_notify(PPG_CHARACTERISTIC_UUID, recorder.on_ppg)
            await client.start_notify(STATUS_CHARACTERISTIC_UUID, on_status)
            await alert.set(False, force=True)  # known-good starting state

            last_detection = 0.0
            try:
                while client.is_connected:
                    now = time.monotonic()
                    if (recorder.elapsed >= args.minimum_seconds
                            and now - last_detection >= args.interval):
                        LOGGER.info("Detection interval reached: elapsed=%.1fs", recorder.elapsed)
                        await detect_and_alert(recorder, alert, args.output_dir,
                                               drowsy_re, awake_re)
                        last_detection = time.monotonic()
                    await asyncio.sleep(1.0)
                LOGGER.warning("BLE connection lost")
                print("Bluetooth connection lost")
            except (asyncio.CancelledError, KeyboardInterrupt):
                LOGGER.info("BLE collection interrupted; beginning shutdown")
                print("Bluetooth collection stopped")

            # Shutdown: final pass, then make sure the buzzer is silent.
            if recorder.sample_count > 0 and client.is_connected:
                try:
                    LOGGER.info("Starting final detection pass")
                    await detect_and_alert(recorder, alert, args.output_dir,
                                           drowsy_re, awake_re)
                except BaseException as exc:
                    LOGGER.exception("Final detection interrupted")
            await alert.set(False, force=True)
    finally:
        recorder.close()
        LOGGER.info("Pipeline stopped: samples=%s csv=%s", recorder.sample_count, csv_path)
    return 0


def main() -> int:
    """Parse command-line options and run the realtime pipeline."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=DEVICE_NAME, help="BLE device name")
    parser.add_argument("--interval", type=float, default=30.0,
                        help="seconds between detection passes")
    parser.add_argument("--minimum-seconds", type=float, default=95.0,
                        help="recording time required before the first detection pass")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fusion_output_realtime")
    parser.add_argument("--recording", default="dual_max30102_realtime_ble.csv")
    parser.add_argument("--recording_csv", "--recording-csv", dest="recording_csv",
                        type=Path,
                        help="run one final detection pass for an existing CSV without BLE")
    parser.add_argument("--led", action="store_true",
                        help="also switch the LED together with the buzzer")
    parser.add_argument("--drowsy-pattern", default=DEFAULT_DROWSY_PATTERN,
                        help="regex marking a drowsy line in detector output")
    parser.add_argument("--awake-pattern", default=DEFAULT_AWAKE_PATTERN,
                        help="regex marking a not-drowsy line (checked first)")
    parser.add_argument("--log-file", type=Path, default=ROOT / "realtime_drowsiness.log",
                        help="file receiving detailed event and function-call logs")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        default="INFO", help="minimum console log level")
    args = parser.parse_args()
    call_logging_started = False
    try:
        configure_logging(args.log_file, args.log_level)
        start_call_logging()
        call_logging_started = True
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        LOGGER.info("Process interrupted by user")
        return 0
    except Exception:
        LOGGER.exception("Unhandled realtime pipeline failure")
        raise
    finally:
        if call_logging_started:
            stop_call_logging()


if __name__ == "__main__":
    raise SystemExit(main())