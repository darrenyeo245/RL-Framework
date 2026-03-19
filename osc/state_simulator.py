import time, random
from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 8000)

while True:
    xyz = [random.uniform(-1,1) for _ in range(3)]
    client.send_message("/adm/obj/101/xyz", xyz)
    time.sleep(0.1)