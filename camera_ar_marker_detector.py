#!/usr/bin/env python3
"""
Multithreaded ArUco/AprilTag detection for Luxonis DepthAI (OAK) cameras.

Key behaviors:
- Robust producer:
  * waits for MXID to enumerate
  * opens with retries/backoff
  * requires first frame within timeout -> else reopen
  * detects stream stalls -> reopen
  * startup deadline -> exit non-zero if never reaches streaming_ok
- Clean shutdown on SIGINT/SIGTERM so the device is released properly.
- Heartbeats to stdout so a supervisor can see progress ("HB:<position>:<tag>").
- Saves images/videos under <log-directory>/{captured_images,captured_videos}.
"""

import os
import sys
import cv2
import time
import json
import queue
import signal
import argparse
import logging
import threading
import traceback
import datetime
import numpy as np
import depthai as dai
from cv2 import aruco
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def setup_camera_logging(camera_position: str, log_level: str, log_directory: str) -> logging.Logger:
    """
    Create a camera-specific logger which logs to <log_directory>/<camera_position>.log
    """
    logger = logging.getLogger(f"Camera-{camera_position}")
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    if logger.handlers:
        return logger  # already configured

    # Resolve log directory relative to the project root (.. from this file)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = log_directory if os.path.isabs(log_directory) else os.path.join(base_dir, log_directory)
    os.makedirs(log_dir, exist_ok=True)

    # Session separator
    sep = "\n" + "=" * 80 + "\n"
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_header = f"{sep}>>> NEW SESSION STARTED AT {ts} <<<{sep}"

    # File handler
    camera_log_path = os.path.join(log_dir, f"{camera_position}.log")
    with open(camera_log_path, "a") as f:
        f.write(session_header + "\n")

    fmt = "%(asctime)s - [%(name)s] - PID:%(process)d - %(levelname)s - %(message)s"
    fh = RotatingFileHandler(camera_log_path, maxBytes=10 * 1024 * 1024, backupCount=3)
    fh.setFormatter(logging.Formatter(fmt))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Also mirror INFO+ to console for visibility (optional)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(fmt))
    ch.setLevel(getattr(logging, log_level, logging.INFO))
    logger.addHandler(ch)

    return logger

# ---------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------
class MarkerValidator:
    def __init__(self, categories=None, category_mapping=None, led_mapping=None, item_names=None):
        self.logger = logging.getLogger("MarkerValidator")
        self.categories = categories or {}
        self.category_mapping = category_mapping or {}
        self.led_mapping = led_mapping or {}
        self.item_names = item_names or {}

    def calculate_zones(self, frame_width, frame_height):
        middle = frame_width // 2
        return {"middle": middle, "width": frame_width, "height": frame_height}

    def get_marker_zone(self, cx, cy, zones):
        return "LEFT" if cx < zones["middle"] else "RIGHT"

    def check_marker_in_zone(self, marker_id, zone):
        category_name = self.category_mapping.get(zone)
        if not category_name:
            return False
        ids = self.categories.get(category_name, set())
        return marker_id in ids

    def process_frame(self, frame, corners, ids):
        h, w = frame.shape[:2]
        zones = self.calculate_zones(w, h)

        # draw middle line
        cv2.line(frame, (zones["middle"], 0), (zones["middle"], zones["height"]), (255, 255, 255), 1)

        zone_status = {"LEFT": False, "RIGHT": False}
        filtered_zone_markers = {"LEFT": [], "RIGHT": []}

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
                zone = self.get_marker_zone(cx, cy, zones)

                info = {
                    "id": int(marker_id),
                    "center_x": cx,
                    "center_y": cy,
                    "corners": pts,
                    "correct_placement": self.check_marker_in_zone(int(marker_id), zone),
                }
                filtered_zone_markers[zone].append(info)

            # decide per zone using bottom-most marker
            for zone in ("LEFT", "RIGHT"):
                if filtered_zone_markers[zone]:
                    sorted_markers = sorted(filtered_zone_markers[zone], key=lambda m: m["center_y"], reverse=True)
                    bottom = sorted_markers[0]
                    zone_status[zone] = bottom["correct_placement"]

                    # draw visual feedback
                    for idx, m in enumerate(sorted_markers):
                        is_bottom = idx == 0
                        color = (0, 255, 0) if m["correct_placement"] else (0, 0, 255)
                        color = color if is_bottom else tuple(int(c * 0.5) for c in color)
                        thick = 2 if is_bottom else 1
                        cv2.circle(frame, (m["center_x"], m["center_y"]), 30, color, thick)
                        name = self.item_names.get(m["id"])
                        if name:
                            cv2.putText(frame, name, (m["center_x"] - 40, m["center_y"] - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return {
            "zone_status": zone_status,
            "filtered_zone_markers": filtered_zone_markers,
            "timestamp": time.time(),
        }

# ---------------------------------------------------------------------
# Serial relay client (named pipe)
# ---------------------------------------------------------------------
class SerialRelayClient:
    """Sends LED commands via the named pipe used by the serial relay server."""
    def __init__(self, port="/dev/ttyUSB0", baud_rate=9600, timeout=1.0,
                 camera_position=None, led_mapping=None, logger=None):
        self.port = port
        self.baud_rate = baud_rate
        self.camera_position = camera_position
        self.pipe_path = "/tmp/serial_relay_pipe"
        self.pipe_fd = None
        self.is_connected = False
        self.exit_event = threading.Event()
        self.send_queue = queue.Queue()
        self.logger = logger or logging.getLogger(f"SerialComm-{camera_position}")
        self.last_led_states = {}
        self.led_mapping = led_mapping or {
            "LEFT": {"green_led": "I", "red_led": "J"},
            "RIGHT": {"green_led": "K", "red_led": "L"},
        }

    def connect(self):
        try:
            if not os.path.exists(self.pipe_path):
                self.logger.error(f"Serial relay pipe not found at {self.pipe_path}")
                return False
            self.pipe_fd = os.open(self.pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            self.is_connected = True
            self.logger.info(f"Connected to serial relay at {self.pipe_path}")
            return True
        except Exception as e:
            self.logger.error(f"Pipe connect failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        if self.pipe_fd is not None:
            try:
                os.close(self.pipe_fd)
            except Exception as e:
                self.logger.error(f"Error closing pipe: {e}")
        self.is_connected = False

    def send_message(self, message: str):
        if not message:
            return
        try:
            self.send_queue.put(message, block=False)
        except queue.Full:
            try:
                _ = self.send_queue.get_nowait()
                self.send_queue.task_done()
                self.send_queue.put(message, block=False)
            except Exception:
                self.logger.warning("Send queue overflow")

    def send_zone_status(self, zone: str, is_valid: bool, timestamp: float):
        if not self.camera_position:
            return
        zone_leds = self.led_mapping.get(zone) or {}
        green_led = zone_leds.get("green_led")
        red_led = zone_leds.get("red_led")
        if not (green_led and red_led):
            return
        msg = f"{timestamp},{green_led}{1 if is_valid else 0}{red_led}{0 if is_valid else 1}"
        self.send_message(msg)

    def communication_thread(self):
        self.logger.info("Serial communication thread started")
        while not self.exit_event.is_set():
            if not self.is_connected:
                if not self.connect():
                    time.sleep(1.0)
                    continue
            try:
                msg = self.send_queue.get(timeout=0.1)
                if not msg.endswith("\n"):
                    msg += "\n"
                os.write(self.pipe_fd, msg.encode("utf-8"))
                self.send_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error writing to pipe: {e}")
                self.is_connected = False
        self.disconnect()
        self.logger.info("Serial communication thread stopped")

    def start(self):
        self.exit_event.clear()
        t = threading.Thread(target=self.communication_thread, daemon=True)
        t.start()
        return t

    def stop(self):
        self.exit_event.set()

# ---------------------------------------------------------------------
# Main detector/manager
# ---------------------------------------------------------------------
class CameraMarkerDetectionManager:
    def __init__(self):
        self.args = self.parse_arguments()
        self.logger = setup_camera_logging(
            camera_position=self.args.camera_position or "unknown",
            log_level=self.args.log_level,
            log_directory=self.args.log_directory,
        )

        self._hb("init")
        self.logger.info(f"Initializing detector for {self.args.camera_position} (MXID={self.args.camera_id})")

        categories, category_mapping, led_mapping, item_names = self.load_zone_configuration()

        # Queues & events
        self.data_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.exit_event = threading.Event()
        self.control_queue = queue.Queue()

        # Serial client
        self.serial_comm = SerialRelayClient(
            port=self.args.serial_port,
            baud_rate=9600,
            timeout=1.0,
            camera_position=self.args.camera_position,
            led_mapping=led_mapping,
            logger=self.logger,
        )
        self.serial_thread = None

        # UI state
        self.display_frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        self.exposure_time = float(self.args.exposure)
        self.sensitivity = int(self.args.iso)
        self.recording = False
        self.video_writer = None
        self.recording_path = None
        self.settings_text = f"Exposure: {self.exposure_time}ms | ISO: {self.sensitivity}"
        self.recording_text = "Press 'r' to start recording"

        # Paths for media under log-directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = self.args.log_directory if os.path.isabs(self.args.log_directory) \
            else os.path.join(base_dir, self.args.log_directory)
        self.video_dir = os.path.join(self.log_dir, "captured_videos")
        self.image_dir = os.path.join(self.log_dir, "captured_images")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

        # ArUco/AprilTag detector
        self.logger.debug("Initializing ArUco/AprilTag detector")
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Validator
        self.validator = MarkerValidator(
            categories=categories,
            category_mapping=category_mapping,
            led_mapping=led_mapping,
            item_names=item_names,
        )

        # DepthAI pipeline
        self.create_pipeline()

    # ----------------- utility -----------------
    def _hb(self, tag: str):
        # Heartbeat for supervisor/parent logs
        print(f"HB:{self.args.camera_position}:{tag}", flush=True)

    # ----------------- pipeline -----------------
    def create_pipeline(self):
        self.pipeline = dai.Pipeline()
        mono = self.pipeline.create(dai.node.MonoCamera)
        mono.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono.setFps(self.args.fps)

        xin = self.pipeline.create(dai.node.XLinkIn)
        xin.setStreamName("control")
        xin.out.link(mono.inputControl)

        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("mono")
        mono.out.link(xout.input)

    # ----------------- producer -----------------
    def producer(self):
        """
        Robust producer with visibility & self-heal:
          - Wait for MXID to enumerate
          - Open with retries/backoff
          - Require first frame within timeout → else reopen
          - Detect stream stalls → reopen
          - Startup deadline → exit non-zero if never reaches streaming_ok
        """
        import gc

        mxid = self.args.camera_id
        cam_pos = self.args.camera_position

        ENUM_TIMEOUT_S = 30.0
        ENUM_POLL_S = 0.5
        OPEN_MAX_TRIES = 30
        OPEN_BACKOFF_BASE_S = 0.6
        FIRST_FRAME_TIMEOUT = 6.0
        NO_FRAME_REOPEN_S = 3.0
        IDLE_SLEEP_S = 0.003
        STARTUP_DEADLINE_S = 45.0

        startup_t0 = time.monotonic()
        startup_ok = False

        self._hb("producer_start")

        def mxid_enumerated():
            try:
                return any(d.getMxId() == mxid for d in dai.Device.getAllAvailableDevices())
            except Exception as e:
                self.logger.debug(f"[{cam_pos}] getAllAvailableDevices() error: {e}")
                return False

        def wait_for_enum(timeout_s):
            t0 = time.monotonic()
            while not self.exit_event.is_set() and (time.monotonic() - t0) < timeout_s:
                if mxid_enumerated():
                    return True
                time.sleep(ENUM_POLL_S)
            return mxid_enumerated()

        while not self.exit_event.is_set():
            # give up entirely if startup deadline exceeded without streaming_ok
            if not startup_ok and (time.monotonic() - startup_t0) > STARTUP_DEADLINE_S:
                self._hb("startup_deadline_exceeded_exit")
                self.logger.error(f"[{cam_pos}] Startup deadline exceeded; exiting with code 14")
                sys.exit(14)

            if not mxid_enumerated():
                self._hb("enum_wait")
                self.logger.info(f"[{cam_pos}] Waiting for {mxid} to enumerate…")
                if not wait_for_enum(ENUM_TIMEOUT_S):
                    self._hb("enum_timeout")
                    self.logger.info(f"[{cam_pos}] {mxid} still not present; retrying…")
                    continue

            self._hb("enum_ok")

            device = None
            last_err = None
            target = dai.DeviceInfo(mxid)

            for attempt in range(1, OPEN_MAX_TRIES + 1):
                if self.exit_event.is_set():
                    return
                try:
                    device = dai.Device(self.pipeline, target)
                    self.logger.info(f"[{cam_pos}] Opened {mxid} (try {attempt}/{OPEN_MAX_TRIES})")
                    self._hb("open_ok")
                    break
                except Exception as e:
                    last_err = e
                    backoff = OPEN_BACKOFF_BASE_S + 0.2 * attempt
                    self._hb(f"open_fail_{attempt}")
                    self.logger.info(f"[{cam_pos}] Open failed try {attempt}: {e} — backoff {backoff:.1f}s")
                    time.sleep(backoff)

            if device is None:
                self.logger.error(f"[{cam_pos}] Failed to open {mxid}: {last_err}")
                self._hb("open_giveup")
                continue

            try:
                q_mono = device.getOutputQueue(name="mono", maxSize=4, blocking=False)
                control_q = device.getInputQueue("control")

                # initial exposure (best effort)
                try:
                    self.set_camera_exposure(self.exposure_time, self.sensitivity)
                except Exception as e:
                    self.logger.warning(f"[{cam_pos}] Exposure set failed (non-fatal): {e}")

                # demand first frame quickly
                first_ok = False
                t0 = time.monotonic()
                while not self.exit_event.is_set() and (time.monotonic() - t0) < FIRST_FRAME_TIMEOUT:
                    try:
                        while not self.control_queue.empty():
                            control_q.send(self.control_queue.get_nowait())
                    except queue.Empty:
                        pass

                    pkt = q_mono.tryGet()
                    if pkt is not None:
                        frame_gray = pkt.getCvFrame()
                        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                        try:
                            self.data_queue.put(
                                {"frame_gray": frame_gray, "frame_bgr": frame_bgr, "timestamp": time.time()},
                                block=False
                            )
                        except queue.Full:
                            pass
                        first_ok = True
                        break
                    time.sleep(0.01)

                if not first_ok:
                    self._hb("first_frame_timeout")
                    self.logger.warning(f"[{cam_pos}] Stream dry after open; closing & retrying…")
                    try:
                        device.close()
                    except Exception:
                        pass
                    gc.collect()
                    time.sleep(1.0)
                    continue

                self._hb("streaming_ok")
                startup_ok = True  # we reached streaming_ok within deadline
                self.logger.info(f"[{cam_pos}] Streaming OK")

                last_frame_ts = time.monotonic()

                # normal streaming loop with stall detection
                while not self.exit_event.is_set():
                    try:
                        while not self.control_queue.empty():
                            control_q.send(self.control_queue.get_nowait())
                    except queue.Empty:
                        pass

                    pkt = q_mono.tryGet()
                    if pkt is None:
                        if (time.monotonic() - last_frame_ts) > NO_FRAME_REOPEN_S:
                            self._hb("stream_stall")
                            self.logger.warning(f"[{cam_pos}] No frames for {NO_FRAME_REOPEN_S}s; reopening device")
                            break
                        time.sleep(IDLE_SLEEP_S)
                        continue

                    last_frame_ts = time.monotonic()
                    frame_gray = pkt.getCvFrame()
                    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                    try:
                        self.data_queue.put(
                            {"frame_gray": frame_gray, "frame_bgr": frame_bgr, "timestamp": time.time()},
                            block=False
                        )
                    except queue.Full:
                        # drop frame silently
                        pass

            except Exception as e:
                self._hb("producer_exception")
                self.logger.error(f"[{cam_pos}] Producer error: {e}")
                self.logger.error(traceback.format_exc())
            finally:
                try:
                    if device is not None:
                        self.logger.info(f"[{cam_pos}] Closing device {mxid}")
                        device.close()
                except Exception:
                    pass
                import gc as _gc
                _gc.collect()

            if not self.exit_event.is_set():
                time.sleep(0.8)

        self._hb("producer_exit")

    # ----------------- processor -----------------
    def processor(self):
        print("Processor thread started")
        try:
            while not self.exit_event.is_set():
                try:
                    data = self.data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame_bgr = data["frame_bgr"]
                frame_gray = data["frame_gray"]

                corners, ids, rejected = self.aruco_detector.detectMarkers(frame_gray)

                vis = frame_bgr.copy()
                results = self.validator.process_frame(vis, corners, ids)

                if self.serial_comm.is_connected:
                    zone_status = results["zone_status"]
                    filtered = results["filtered_zone_markers"]
                    ts = results["timestamp"]
                    for zone in ("LEFT", "RIGHT"):
                        if filtered[zone]:  # only if something present in that zone
                            self.serial_comm.send_zone_status(zone, zone_status[zone], ts)

                self.result_queue.put(vis)
                self.data_queue.task_done()
        except Exception as e:
            print(f"Processor error: {e}")
            traceback.print_exc()
        finally:
            print("Processor thread stopped")

    # ----------------- display -----------------
    def display(self):
        print("Display thread started")
        print("Controls: 'q' quit | 's' save frame | 'r' rec on/off | 'a' auto exp | 'w/e' exp -/+ | 'u/i' ISO -/+")
        TARGET_W, TARGET_H = 640, 480
        fps_counter, fps_timer, fps = 0, time.time(), 0.0

        try:
            while not self.exit_event.is_set():
                try:
                    frame = self.result_queue.get(timeout=0.1)
                except queue.Empty:
                    if self.display_frame is not None:
                        frame = self.display_frame.copy()
                        cv2.putText(frame, "Waiting for new frame...",
                                    (frame.shape[1] // 4, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow(f"AR Marker Detection ({self.args.camera_position})", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            self.exit_event.set()
                    continue

                h, w = frame.shape[:2]
                aspect = w / h
                target_aspect = TARGET_W / TARGET_H
                if aspect > target_aspect:
                    dw, dh = TARGET_W, int(TARGET_W / aspect)
                else:
                    dh, dw = TARGET_H, int(TARGET_H * aspect)
                disp = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)

                # record original frame if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)

                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps = fps_counter / (time.time() - fps_timer)
                    fps_counter, fps_timer = 0, time.time()

                # HUD
                cv2.rectangle(disp, (0, 0), (dw, 35), (0, 0, 0), -1)
                category_colors = {
                    "GREEN": (0, 255, 0),
                    "RED": (0, 0, 255),
                    "YELLOW": (0, 255, 255),
                    "WHITE": (255, 255, 255),
                    "UNKNOWN": (200, 200, 200),
                }
                left_cat = self.validator.category_mapping.get("LEFT", "UNKNOWN")
                right_cat = self.validator.category_mapping.get("RIGHT", "UNKNOWN")
                left_color = category_colors.get(left_cat, (200, 200, 200))
                right_color = category_colors.get(right_cat, (200, 200, 200))

                zones_display = self.validator.calculate_zones(dw, dh)
                centre = zones_display["middle"]
                left_cx = zones_display["middle"] // 2
                right_cx = zones_display["middle"] + (zones_display["width"] - zones_display["middle"]) // 2

                cv2.putText(disp, f"{left_cat}", (left_cx - 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
                cv2.putText(disp, f"{right_cat}", (right_cx - 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)
                cv2.putText(disp, f"Processed FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(disp, self.settings_text, (centre - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow(f"AR Marker Detection ({self.args.camera_position})", disp)
                self.display_frame = disp
                self.result_queue.task_done()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.exit_event.set()
                elif key == ord("s"):
                    fn = self.save_current_frame(frame)
                    print(f"Saved frame to {fn}")
                elif key == ord("r"):
                    if not self.recording:
                        if self.start_recording(frame):
                            self.recording_text = "Recording... Press 'r' to stop"
                    else:
                        if self.stop_recording():
                            self.recording_text = "Press 'r' to start recording"
                elif key == ord("a"):
                    self.set_auto_exposure()
                    self.settings_text = "Exposure: Auto | ISO: Auto"
                elif key == ord("e"):
                    self.exposure_time = min(33.0, self.exposure_time + 0.5)
                    self.set_camera_exposure(self.exposure_time, self.sensitivity)
                    self.settings_text = f"Exposure: {self.exposure_time}ms | ISO: {self.sensitivity}"
                elif key == ord("w"):
                    self.exposure_time = max(0.5, self.exposure_time - 0.5)
                    self.set_camera_exposure(self.exposure_time, self.sensitivity)
                    self.settings_text = f"Exposure: {self.exposure_time}ms | ISO: {self.sensitivity}"
                elif key == ord("i"):
                    self.sensitivity = min(1600, self.sensitivity + 100)
                    self.set_camera_exposure(self.exposure_time, self.sensitivity)
                    self.settings_text = f"Exposure: {self.exposure_time}ms | ISO: {self.sensitivity}"
                elif key == ord("u"):
                    self.sensitivity = max(100, self.sensitivity - 100)
                    self.set_camera_exposure(self.exposure_time, self.sensitivity)
                    self.settings_text = f"Exposure: {self.exposure_time}ms | ISO: {self.sensitivity}"

        except Exception as e:
            print(f"Display error: {e}")
            traceback.print_exc()
        finally:
            if self.recording:
                self.stop_recording()
            cv2.destroyAllWindows()
            print("Display thread stopped")

    # ----------------- camera controls -----------------
    def set_camera_exposure(self, exposure_time_ms, sensitivity):
        us = int(exposure_time_ms * 1000)
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(us, int(sensitivity))
        self.control_queue.put(ctrl)

    def set_auto_exposure(self):
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self.control_queue.put(ctrl)

    # ----------------- media -----------------
    def save_current_frame(self, frame):
        ts = time.strftime("%Y%m%d%H%M%S")
        fn = os.path.join(self.image_dir, f"{ts}_captured_image.jpg")
        cv2.imwrite(fn, frame)
        return fn

    def start_recording(self, frame):
        if self.recording:
            return False
        ts = time.strftime("%Y%m%d%H%M%S")
        self.recording_path = os.path.join(self.video_dir, f"{ts}_captured_video.mp4")
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(self.recording_path, fourcc, self.args.fps, (w, h))
        self.recording = True
        print(f"Started recording to {self.recording_path}")
        return True

    def stop_recording(self):
        if not self.recording:
            return False
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            print(f"Saved video to {self.recording_path}")
            self.video_writer = None
        return True

    # ----------------- run -----------------
    def run(self, duration=0):
        # start serial comms first
        self.serial_thread = self.serial_comm.start()
        print("Serial communication started")

        producer_thread = threading.Thread(target=self.producer, daemon=True)
        processor_thread = threading.Thread(target=self.processor, daemon=True)
        display_thread = threading.Thread(target=self.display, daemon=True)

        producer_thread.start()
        processor_thread.start()
        display_thread.start()

        start = time.time()
        try:
            while not self.exit_event.is_set() and (duration == 0 or (time.time() - start) < duration):
                time.sleep(0.1)
            if not self.exit_event.is_set() and duration > 0:
                self.exit_event.set()
        except KeyboardInterrupt:
            self.exit_event.set()
        finally:
            self.exit_event.set()
            self.serial_comm.stop()
            # join threads briefly
            if self.serial_thread:
                self.serial_thread.join(timeout=2.0)
            for t in (producer_thread, processor_thread, display_thread):
                t.join(timeout=2.0)
            # GC
            import gc
            gc.collect()
            print("Program completed")

    # ----------------- args & config -----------------
    def parse_arguments(self):
        p = argparse.ArgumentParser(description="ArUco/AprilTag detection with zone validation")
        p.add_argument("--camera-id", type=str, required=True, help="Camera MX ID")
        p.add_argument("--camera-position", type=str, required=True, help="Camera position label")
        p.add_argument("--category-mapping", type=str, required=True,
                       help="LEFT:<CAT>,RIGHT:<CAT>")
        p.add_argument("--categories", type=str, required=True, help="JSON of categories -> [ids]")
        p.add_argument("--item-names", type=str, required=True, help="JSON of id->name")
        p.add_argument("--led-mapping", type=str, default="{}", help="JSON LED mapping per zone")
        p.add_argument("--serial-port", type=str, default="/dev/ttyUSB0", help="Serial port (for logging only)")
        p.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
        p.add_argument("--exposure", type=float, default=1.5, help="Exposure (ms)")
        p.add_argument("--iso", type=int, default=100, help="ISO")
        p.add_argument("--fps", type=int, default=60, help="Camera FPS")
        p.add_argument("--log-directory", type=str, default="logs", help="Log/media directory")
        return p.parse_args()

    def load_zone_configuration(self):
        # Defaults
        categories = {}
        item_names = {}
        led_mapping = {}
        category_mapping = {}

        # Parse JSON strings safely
        try:
            cat_dict = json.loads(self.args.categories) if self.args.categories else {}
            categories = {name: set(ids) for name, ids in cat_dict.items()}
            self.logger.info("Loaded categories")
        except Exception as e:
            self.logger.warning(f"Failed to parse categories: {e}")

        try:
            item_dict = json.loads(self.args.item_names) if self.args.item_names else {}
            item_names = {int(k): v for k, v in item_dict.items()}
            self.logger.info("Loaded item names")
        except Exception as e:
            self.logger.warning(f"Failed to parse item names: {e}")

        try:
            led_mapping = json.loads(self.args.led_mapping) if self.args.led_mapping else {}
            if not isinstance(led_mapping, dict):
                led_mapping = {}
            self.logger.info("Loaded LED mapping")
        except Exception as e:
            self.logger.warning(f"Failed to parse LED mapping: {e}")
            led_mapping = {}

        try:
            # format: LEFT:CAT,RIGHT:CAT
            cm = {}
            for pair in (self.args.category_mapping or "").split(","):
                if ":" in pair:
                    zone, cat = pair.split(":", 1)
                    zone = zone.strip().upper()
                    cat = cat.strip().upper()
                    if zone in ("LEFT", "RIGHT"):
                        cm[zone] = cat
            category_mapping = cm
            self.logger.info(f"Zone mapping: {category_mapping}")
        except Exception as e:
            self.logger.warning(f"Failed to parse category mapping: {e}")

        return categories, category_mapping, led_mapping, item_names

# ---------------------------------------------------------------------
# Entrypoint with clean signal handling
# ---------------------------------------------------------------------
if __name__ == "__main__":
    mgr = CameraMarkerDetectionManager()

    def _graceful_stop(sig, frm):
        # Heartbeat & flag so the producer closes the device cleanly
        mgr._hb(f"signal_{sig}")
        mgr.exit_event.set()

    signal.signal(signal.SIGTERM, _graceful_stop)
    signal.signal(signal.SIGINT, _graceful_stop)

    # Run indefinitely; parent supervisor decides lifetime
    mgr.run(duration=0)
