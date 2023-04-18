import routeros_api
import sys
import socket
import time

# Replace these variables with your own Mikrotik RouterOS credentials and router IP address
USERNAME = "admin"
ROUTER_IP = "192.168.250.242"
PASSWORD = ""
# Get the TX sector value using the RouterOS library
def get_tx_sector():
    # try:
    # Connect to the Mikrotik router using RouterOS API over SSH
    connection = routeros_api.RouterOsApiPool(ROUTER_IP, username=USERNAME, password=PASSWORD, plaintext_login=True)
    api = connection.get_api()
    for i in range(7):
    # api.get_binary_resource('/interface/w60g').call('set',{'id':'*0','tx-sector':'25'})
    	t=time.time()
    	api.get_resource('/interface/w60g').call('set',{'numbers':'0', 'tx-sector':str(25+i*2)})
    	print("Dem times be :" + str(time.time()-t))
    	tx_sector = api.get_resource('/interface/w60g').get()[0]['tx-sector']
    	print(tx_sector)
    # Close the RouterOS API connection
    connection.disconnect()
    # except (socket.error, routeros_api.RouterOsApiException) as e:
    #     print(f"Error: {e}")
    #     return None

    return tx_sector

# Main program
if __name__ == "__main__":

    tx_sector = get_tx_sector()
   
