import time
import threading


class SimpleSpinner:
    def __init__(self, message="Loading"):
        self.message = message
        self.spinning = False
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.thread = None

    def _spin(self):
        idx = 0
        while self.spinning:
            print(
                f"\r{self.spinner_chars[idx % len(self.spinner_chars)]} {self.message}...",
                end="",
                flush=True,
            )
            idx += 1
            time.sleep(0.1)

    def start(self):
        self.spinning = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def stop(self, final_message=None):
        self.spinning = False
        if self.thread:
            self.thread.join()
        print("\r" + " " * 80 + "\r", end="")
        if final_message:
            print(f"✅ {final_message}")
        else:
            print(f"✅ {self.message} complete!")
