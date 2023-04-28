import socket
import os
import sys
import time

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to a specific IP address and port
server_address = ('192.168.250.11', 9999)
sock.bind(server_address)

#pipe_path = '/home/roshni/Downloads/rxdata'
#pipe_fd = os.open(pipe_path, os.O_WRONLY)
#pipe_file = os.fdopen(pipe_fd)

#file1 = open("locData.txt", "a")
    
#print('Server is running and waiting for messages...')

# Continuously listen for incoming datagrams
while True:
    data, address = sock.recvfrom(4096)
    decodedData = data.decode()
    file1 = open("locData.txt", "w") #TODO: Make it w instead of a
    file1.write(str(decodedData))
    file1.close()
    #print(str(decodedData)+'\n')
    
    # Print received message and client address
    print(f'Received "{data.decode()}" from {address}')
    
    # Send a response back to the client
    #response = f'Received "{data.decode()}"'
    #sock.sendto(response.encode(), address)
