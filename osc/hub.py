from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client

clients = [
    ("127.0.0.1", 9001), #OSC-Interface/RL-System
    ("127.0.0.1", 9002), #State-Simulator
    ("127.0.0.1", 9004), #Visualizer
]


def broadcast_handler(address, *args):
    for ip, port in clients:
        client = udp_client.SimpleUDPClient(ip, port)
        client.send_message(address, list(args))


dispatcher = Dispatcher()
dispatcher.set_default_handler(broadcast_handler)

server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 8000), dispatcher)
print("Server started at 0.0.0.0:8000")
server.serve_forever()