"""
Perfetto Viewer for Chisel Traces

Clean, minimal integration to open trace JSON files in Perfetto UI automatically.
"""

import json
import os
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Optional
import urllib.request


def open_trace_in_perfetto(trace_file: str) -> bool:
    """
    Open a trace file in Perfetto UI with automatic loading

    Args:
        trace_file: Path to the trace JSON file

    Returns:
        True if successful, False otherwise
    """
    viewer = PerfettoViewer(trace_file)
    return viewer.open()


class PerfettoViewer:
    """Minimal Perfetto UI integration"""

    def __init__(self, trace_file: str):
        self.trace_file = Path(trace_file)
        self.trace_processor_path: Optional[Path] = None
        self.trace_processor_proc: Optional[subprocess.Popen] = None

    def _validate_trace(self) -> bool:
        """Basic trace file validation"""
        if not self.trace_file.exists():
            print(f"❌ Error: Trace file '{self.trace_file}' not found")
            return False

        try:
            with open(self.trace_file, "r") as f:
                json.load(f)
            print(f"✅ Valid trace file: {self.trace_file.name}")
            return True
        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON file")
            return False

    def _get_trace_processor(self) -> bool:
        """Get trace_processor binary"""
        # Check standard location first
        home_path = (
            Path.home()
            / ".local"
            / "share"
            / "perfetto"
            / "prebuilts"
            / "trace_processor_shell"
        )
        if home_path.exists():
            self.trace_processor_path = home_path
            return True

        # Download if needed
        local_path = Path("trace_processor")
        self.trace_processor_path = local_path

        if local_path.exists():
            return True

        print("📥 Downloading trace_processor...")
        try:
            url = "https://raw.githubusercontent.com/google/perfetto/v35.0/tools/trace_processor"
            urllib.request.urlretrieve(url, str(local_path))
            os.chmod(local_path, 0o755)

            # Run to download binary
            subprocess.run(
                [str(local_path), "--help"], capture_output=True, text=True, check=True
            )

            # Use the downloaded binary
            if home_path.exists():
                self.trace_processor_path = home_path

            return True
        except Exception as e:
            print(f"❌ Failed to get trace_processor: {e}")
            return False

    def _start_server(self, port: int = 9001) -> bool:
        """Start trace_processor HTTP server"""
        try:
            cmd = [
                str(self.trace_processor_path),
                "--httpd",
                f"--http-port={port}",
                str(self.trace_file),
            ]

            self.trace_processor_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            time.sleep(2)  # Brief startup time

            if self.trace_processor_proc.poll() is not None:
                return False

            print(f"🚀 Server started on port {port}")
            return True
        except Exception:
            return False

    def _open_browser(self, port: int = 9001) -> bool:
        """Open Perfetto UI in browser"""
        url = f"https://ui.perfetto.dev/?rpc_port={port}"
        try:
            webbrowser.open(url)
            print("🌐 Opened Perfetto UI")
            print("⏳ Press Ctrl+C when done viewing")
            return True
        except Exception:
            print(f"❌ Failed to open browser. Manual URL: {url}")
            return True

    def _cleanup(self):
        """Stop server"""
        if self.trace_processor_proc and self.trace_processor_proc.poll() is None:
            self.trace_processor_proc.terminate()
            try:
                self.trace_processor_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.trace_processor_proc.kill()

    def open(self) -> bool:
        """Open trace in Perfetto UI"""
        print(f"🎯 Opening {self.trace_file.name} in Perfetto...")

        if not self._validate_trace():
            return False

        if not self._get_trace_processor():
            return False

        if not self._start_server():
            print("❌ Failed to start server")
            return False

        if not self._open_browser():
            return False

        try:
            while (
                self.trace_processor_proc and self.trace_processor_proc.poll() is None
            ):
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹ Stopping...")
        finally:
            self._cleanup()

        return True
