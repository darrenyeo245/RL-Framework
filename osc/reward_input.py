from pythonosc import udp_client

client = udp_client.SimpleUDPClient("127.0.0.1", 8000)

print("Reward input gestartet: '+' = +1, '-' = -1, 'q' = Ende")

while True:
    key = input("+ / - / q: ").strip()

    if key == "+":
        client.send_message("/reward", 1)
        print("Sent reward: +1")
    elif key == "-":
        client.send_message("/reward", -1)
        print("Sent reward: -1")
    elif key.lower() == "q":
        break
