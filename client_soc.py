import socket
import os

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send a message to the server
server_address = ('192.168.250.11', 9999)
message = 'Hello, server!'
while True:
    with open("xy.txt",'r+') as fo:
        if os.path.getsize('xy.txt')!=0:
            response = fo.readlines()[-1]
            fo.truncate(0)
            print(response) 
            sock.sendto(response.encode(), server_address)
# Wait for a response from the server
response, server = sock.recvfrom(4096)

# Print the response from the server
print(f'Received "{response.decode()}" from {server}')
