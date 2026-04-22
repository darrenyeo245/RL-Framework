import threading
import numpy as np
from pythonosc import osc_server, udp_client
from pythonosc.dispatcher import Dispatcher

IN_IP = "0.0.0.0"
IN_PORT = 9002
BROADCAST_IP = "127.0.0.1"
BROADCAST_PORT = 8000

STATE_OUT_ADDR = "/adm/obj/101/xyz"
RESET_IN_ADDR = "/episode/reset"
STEP_TRIGGER_ADDRS = [
    "/reward",
    "/episode/end",
    "/episode/reset_manual",
    "/training/stop",
]

def clamp_state(values):
    s = np.array(values[:3], dtype=np.float32)
    s[0] = np.clip(s[0], -1.0, 1.0)
    s[1] = np.clip(s[1], -1.0, 1.0)
    s[2] = np.clip(s[2], -1.0, 1.0)
    return s


state = clamp_state([0.0, 0.0, 0.0])
lock = threading.Lock()
client = udp_client.SimpleUDPClient(BROADCAST_IP, BROADCAST_PORT, allow_broadcast=False)

RNG = np.random.default_rng()

STEP_SCALE = 0.2
DRIFT_SCALE = 0.1
Z_BIAS = 0.01


def publish_state():
    with lock:
        payload = state.tolist()
    client.send_message(STATE_OUT_ADDR, payload)
    print(f"[sim->osc] {STATE_OUT_ADDR} {payload} -> {BROADCAST_IP}:{BROADCAST_PORT}", flush=True)


def reset_handler(address, *args):
    del address
    global state
    if len(args) < 3:
        return
    with lock:
        state = clamp_state(args)
    publish_state()


def step_handler(address, *args):
    del address, args
    with lock:
        noise = RNG.normal(0.0, STEP_SCALE, size=3)
        drift = -state * DRIFT_SCALE
        drift[2] = abs(drift[2]) + Z_BIAS
        state[:] = clamp_state(state + noise + drift)
    publish_state()


dispatcher = Dispatcher()
dispatcher.map(RESET_IN_ADDR, reset_handler)
for addr in STEP_TRIGGER_ADDRS:
    dispatcher.map(addr, step_handler)

server = osc_server.ThreadingOSCUDPServer((IN_IP, IN_PORT), dispatcher)
print(f"State simulator listening on {IN_IP}:{IN_PORT}")
publish_state()
server.serve_forever()
