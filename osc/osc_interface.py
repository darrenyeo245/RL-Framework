# osc/osc_interface.py
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client
import threading
import numpy as np


class OSCInterface:
    def __init__(self):
        self.state = np.zeros(3, dtype=float)
        self.reward = 0.0
        self._reward_pending = False
        self._lock = threading.Condition()

        self.client = udp_client.SimpleUDPClient("127.0.0.1", 8000)

        dispatcher = Dispatcher()
        dispatcher.map("/adm/obj/101/xyz", self.state_handler)
        dispatcher.map("/reward", self.reward_handler)

        self.server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 9001), dispatcher)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def state_handler(self, address, *args):
        if args:
            self.state = np.array(args, dtype=float)

    def reward_handler(self, address, *args):
        if not args:
            return
        with self._lock:
            self.reward += float(args[0])
            self._reward_pending = True
            self._lock.notify_all()

    def get_state(self):
        return self.state

    def get_reward(self, wait_for_new=False, timeout=None):
        with self._lock:
            if wait_for_new and not self._reward_pending:
                self._lock.wait(timeout=timeout)

            r = self.reward
            self.reward = 0.0
            self._reward_pending = False
            return r

    def send_action(self, action):
        self.client.send_message("/adm/obj/1/xyz", action.tolist())