import time
import threading


class Monitor:
    def __init__(self, func):
        self.callback_func = func
        self.stop = False
        self.monitor_folder = "./dump"

    def start_monitor(self):
        thread = threading.Thread(target=self._run)
        thread.start()

    def on_change(self):
        pass

    def _run(self):
        while not self.stop:
            if self.on_change():
                self.callback_func()
                time.sleep(30)

    def stop_monitor(self):
        self.stop = True
