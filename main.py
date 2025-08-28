#!/usr/bin/env python3
"""
Main controller for recycling truck slot entry validation system
Launches camera detector processes based on configuration
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
import subprocess
import select
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Main")

# Define a global list for cleanup during shutdown
cleanup_processes = []  # Replaces 'processes'
shutdown_in_progress = False  # Add this flag to prevent recursive signal handling


def signal_handler(sig, frame):
    """Clean up and exit when signal received"""
    global shutdown_in_progress

    # Prevent recursive calls to the signal handler
    if shutdown_in_progress:
        return
    shutdown_in_progress = True

    logger.info("Shutdown signal received")

    # Track all terminated PGIDs for better debugging
    terminated_pgids = set()

    # Clean up all processes in the global list
    for proc in cleanup_processes:
        try:
            if proc.poll() is None:  # Process is still running
                # Get the actual process group ID
                pgid = os.getpgid(proc.pid)
                terminated_pgids.add(pgid)

                # Kill entire process group
                logger.info(
                    f"Terminating process group for PID {proc.pid} (PGID: {pgid})"
                )
                os.killpg(pgid, signal.SIGTERM)

                # Wait briefly
                time.sleep(0.2)

                # Check if process terminated
                if proc.poll() is None:  # Still not terminated
                    logger.info(
                        f"Force killing process group for PID {proc.pid} (PGID: {pgid})"
                    )
                    os.killpg(pgid, signal.SIGKILL)
        except Exception as e:
            logger.error(f"Error terminating process group: {e}")

    # Report all terminated PGIDs for debugging
    if terminated_pgids:
        logger.info(f"Terminated process groups: {sorted(list(terminated_pgids))}")

    sys.exit(0)


def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)



def launch_camera_processes(config):
    
    """
    Launch camera detector processes. Do NOT gate on DepthAI enumeration here.
    Children will wait for MXID readiness and verify streaming themselves.
    """
    global cleanup_processes
    try:
        system_config = config.get("system", {})
        camera_script = system_config.get("camera_script", "src/camera_ar_marker_detector.py")
        serial_port = system_config.get("serial_port", "/dev/ttyACM0")
        log_level = system_config.get("log_level", "INFO")
        log_directory = system_config.get("log_directory", "logs")

        # Optional tiny spawn gap (spreads USB load a touch)
        spawn_gap_s = float(system_config.get("spawn_gap_s", 0.4))

        # Resolve camera script path
        if not os.path.exists(camera_script):
            alt = os.path.join(os.path.dirname(__file__), os.path.basename(camera_script))
            if os.path.exists(alt):
                camera_script = alt
            else:
                logger.error(f"Camera script {camera_script} not found, cannot continue")
                return []

        # Non-fatal enumeration snapshot (for logs only)
        try:
            import depthai as dai
            snapshot = sorted({d.getMxId() for d in dai.Device.getAllAvailableDevices()})
            logger.info(f"DepthAI enumeration snapshot (non-blocking): {snapshot}")
        except Exception:
            logger.info("DepthAI enumeration snapshot not available (non-fatal)")

        categories_json = json.dumps(config.get("categories", {}))
        item_names_json = json.dumps(config.get("item_names", {}))
        all_led_mapping = config.get("led_mapping", {})

        logger.info(f"Starting camera processes using {camera_script}")
        camera_procs = []

        cameras_cfg = config.get("cameras", [])
        for i, cam_cfg in enumerate(cameras_cfg):
            mxid = cam_cfg.get("id", "")
            if not mxid:
                logger.warning(f"Camera index {i} missing 'id' (MXID); skipping")
                continue

            position = cam_cfg.get("position", f"camera_{i}_position")

            # Per-position LED mapping (optional)
            led_mapping_json = json.dumps(all_led_mapping.get(position, {}))

            # Validate LEFT/RIGHT mapping
            cm = cam_cfg.get("category_mapping", {})
            if "LEFT" not in cm or "RIGHT" not in cm:
                logger.warning(f"Camera {mxid} ({position}) needs LEFT and RIGHT categories; skipping")
                continue
            category_mapping_arg = f"LEFT:{cm['LEFT']},RIGHT:{cm['RIGHT']}"

            # Build command
            cmd = [
                "python3", camera_script,
                "--camera-id", mxid,
                "--camera-position", position,
                "--category-mapping", category_mapping_arg,
                "--categories", categories_json,
                "--item-names", item_names_json,
                "--led-mapping", led_mapping_json,
                "--serial-port", serial_port,
                "--log-level", log_level,
                "--log-directory", log_directory,
                "--relay-pipe", relay_pipe,
            ]
            settings = cam_cfg.get("settings", {})
            if "exposure" in settings: cmd += ["--exposure", str(settings["exposure"])]
            if "iso" in settings:      cmd += ["--iso", str(settings["iso"])]
            if "fps" in settings:      cmd += ["--fps", str(settings["fps"])]

            logger.info(f"[{position}] (MXID {mxid}) launching: {' '.join(cmd)}")

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    preexec_fn=os.setsid,  # new PGID for clean shutdown
                )
                proc.cmd = cmd
                camera_procs.append(proc)
                cleanup_processes.append(proc)
                logger.info(f"[{position}] started (PID {proc.pid})")
            except Exception as e:
                logger.error(f"Failed to start process for {position} ({mxid}): {e}")

            # small spawn gap
            if i < len(cameras_cfg) - 1 and spawn_gap_s > 0:
                time.sleep(spawn_gap_s)

        return camera_procs

    except Exception as e:
        logger.error(f"Failed to launch camera processes: {e}", exc_info=True)
        return []



def launch_serial_server(config, log_level="INFO"):
    """Launch the serial relay server process"""
    try:
        # Get configuration
        system_config = config.get("system", {})
        serial_port = system_config.get("serial_port", "/dev/ttyUSB0")
        baud_rate = system_config.get("baud_rate", "9600")
        log_directory = system_config.get("log_directory", "logs")
        pipe_path = system_config.get("serial_relay_pipe", "/tmp/serial_relay_pipe")

        # Make sure log directory exists
        os.makedirs(log_directory, exist_ok=True)

        # Get script path
        serial_relay_server_script = system_config.get(
            "serial_relay_server_script", "src/serial_relay_server_V3.py"
        )

        # Check if the script exists
        if not os.path.exists(serial_relay_server_script):
            script_path = os.path.join(
                os.path.dirname(__file__), os.path.basename(serial_relay_server_script)
            )
            if os.path.exists(script_path):
                serial_relay_server_script = script_path
            else:
                logger.error(
                    f"Serial relay script {serial_relay_server_script} not found"
                )
                return None

        # Build command
        cmd = [
            "python3",
            serial_relay_server_script,
            "--serial-port",
            serial_port,
            "--baud-rate",
            baud_rate,
            "--log-level",
            log_level,
            "--log-directory",
            log_directory,
        ]
        try:
            # harmless if server ignores unknown args? If not sure, guard behind a config flag.
            cmd += ["--pipe-path", pipe_path]
        except Exception:
            pass

        # Launch the process
        logger.info(f"Starting serial relay server process on port {serial_port}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid,  # new PGID
        )

        # Store the command for potential restart
        process.cmd = cmd

        # Add to global cleanup list
        global cleanup_processes
        cleanup_processes.append(process)

        # Check if process started successfully
        if process.poll() is not None:
            # Process terminated immediately
            stderr_output = process.stderr.read() if process.stderr else ""
            logger.error(f"Serial relay server failed to start: {stderr_output}")
            return None

        logger.info(f"Serial relay server process started (PID: {process.pid})")
        return process

    except Exception as e:
        logger.error(f"Failed to start serial relay server process: {e}", exc_info=True)
        return None


def monitor_processes(processes):
    """Monitor running processes and their output"""
    # Set up process status tracking
    process_status = {}

    # Initialize status for each process
    for i, proc in enumerate(processes):
        # Get process name
        process_name = "Unknown"
        if hasattr(proc, "cmd") and proc.cmd:
            # Extract script name from command
            for arg in proc.cmd:
                if arg.endswith(".py"):
                    process_name = os.path.basename(arg)
                    break

        process_status[i] = {
            "name": process_name,
            "restarts": 0,
            "last_restart": 0,
            "cmd": getattr(proc, "cmd", None),
        }

    try:
        logger.info("All processes started, monitoring...")

        while True:
            # Check each process status
            for i, proc in enumerate(processes):
                proc_name = process_status[i]["name"]

                # Check if process is still running
                if proc.poll() is not None:  # Process has terminated
                    exit_code = proc.returncode
                    logger.warning(
                        f"Process {proc_name} (index {i}) exited with code {exit_code}"
                    )

                    # Get any error output
                    stderr_output = proc.stderr.read() if proc.stderr else ""
                    if stderr_output.strip():
                        logger.error(
                            f"Error from {proc_name}:\n{stderr_output.strip()}"
                        )

                    # Restart logic - only if not too many restarts
                    if process_status[i]["restarts"] < 5:  # Max 5 restarts
                        # Check if last restart was more than 60 seconds ago
                        if time.time() - process_status[i]["last_restart"] > 60:
                            process_status[i][
                                "restarts"
                            ] = 0  # Reset count after 60s of stability

                        # Increment restart count and timestamp
                        process_status[i]["restarts"] += 1
                        process_status[i]["last_restart"] = time.time()

                        # Only restart if we have the command
                        if process_status[i]["cmd"]:
                            logger.info(
                                f"Restarting {proc_name} (restart #{process_status[i]['restarts']})"
                            )
                            try:
                                # Before replacing the process:
                                if hasattr(proc, "stdout") and proc.stdout:
                                    proc.stdout.close()
                                if hasattr(proc, "stderr") and proc.stderr:
                                    proc.stderr.close()

                                new_proc = subprocess.Popen(
                                    process_status[i]["cmd"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    bufsize=1,
                                    universal_newlines=True,
                                    preexec_fn=os.setsid,  # ensure new PGID like first launch
                                )
                                # Keep track of the command for future restarts
                                new_proc.cmd = process_status[i]["cmd"]
                                # Replace the process in our list
                                processes[i] = new_proc

                                # Also update the global cleanup_processes list
                                global cleanup_processes
                                for idx, p in enumerate(cleanup_processes):
                                    if (
                                        p == proc
                                    ):  # Find the old process in the global list
                                        cleanup_processes[idx] = (
                                            new_proc  # Replace with new process
                                        )
                                        logger.debug(
                                            f"Updated process in global cleanup list at index {idx}"
                                        )
                                        break

                            except Exception as e:
                                logger.error(
                                    f"Failed to restart {proc_name}: {e}", exc_info=True
                                )
                        else:
                            logger.error(
                                f"Cannot restart {proc_name} - no command stored"
                            )
                    else:
                        logger.error(
                            f"Too many restarts for {proc_name}, not restarting"
                        )

                # Process is running - check for output
                else:
                    # Read any available output without blocking
                    try:
                        while True:
                            output = proc.stdout.readline()
                            if not output:
                                break
                            if output.strip():
                                logger.info(f"{proc_name}: {output.strip()}")
                    except Exception:
                        pass  # No more output available

                    # Check for stderr without blocking
                    try:
                        while (
                            proc.stderr and select.select([proc.stderr], [], [], 0)[0]
                        ):
                            error = proc.stderr.readline()
                            if not error:
                                break
                            if error.strip():
                                logger.error(f"{proc_name} error: {error.strip()}")
                    except Exception:
                        pass  # No more error output available

            # Wait before checking again
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        signal_handler(None, None)
    except Exception as e:
        logger.error(f"Error in process monitor: {e}", exc_info=True)


def setup_logging(config, args):
    """Set up logging with terminal output and error-only file logging"""

    # Get settings from config
    system_config = config.get("system", {})
    log_directory = system_config.get("log_directory", "logs")
    log_level = getattr(logging, args.log_level)

    # Create logs directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler - shows all logs
    console_format = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    console_formatter = logging.Formatter(console_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)  # Show all logs in console
    root_logger.addHandler(console_handler)

    # Create file handler - only for errors
    error_file = os.path.join(log_directory, "main.log")
    file_format = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    file_formatter = logging.Formatter(file_format)
    file_handler = TimedRotatingFileHandler(
        error_file,
        when="d",  # 'd' for days (other options: 'h' for hours, 'm' for minutes)
        interval=7,  # Keep 7 days of logs
        backupCount=7,  # Keep 7 backup files at most
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.ERROR)  # Only ERROR and above go to file
    root_logger.addHandler(file_handler)

    # For more verbose debugging with function names and line numbers
    debug_format = "%(asctime)s - [%(name)s] - PID:%(process)d - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"

    # For standard logs with just process ID added
    standard_format = (
        "%(asctime)s - [%(name)s] - PID:%(process)d - %(levelname)s - %(message)s"
    )

    if log_level == logging.DEBUG:
        console_formatter = logging.Formatter(debug_format)
    else:
        console_formatter = logging.Formatter(standard_format)

    return logging.getLogger("Main")


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Recycling truck slot entry validation controller"
    )
    parser.add_argument(
        "--config", default="config/config.json", help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logging
    logger = setup_logging(config, args)
    logger.info(f"Loading configuration from {args.config}")

    # Override log level from command line if specified
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start serial relay process
    serial_server_process = launch_serial_server(config, args.log_level)
    if not serial_server_process:
        logger.error("Failed to start serial relay server, exiting")
        return 1

    # Give relay time to initialize
    time.sleep(1.0)

    # Launch camera processes
    logger.info("Starting camera processes...")
    camera_processes = launch_camera_processes(config)

    # Build list of processes to monitor
    monitored_processes = [
        serial_server_process
    ] + camera_processes  # Replaces 'all_processes'

    # Make sure all processes are in the cleanup list
    for proc in camera_processes:
        if proc not in cleanup_processes:
            cleanup_processes.append(proc)

    if not camera_processes:
        logger.error("No camera processes could be started")
        return 1

    # Monitor all processes
    monitor_processes(monitored_processes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
