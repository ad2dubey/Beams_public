import sys
import socket
import time
import os
import math
import numpy as np
import pandas as pd
import statistics
import routeros_api

# Assumptions: 
# 1. Only STA 1 moves 
# 2. Both STA 1 and STA 2 are within the beam width of each other
USERNAME = "admin"
ROUTER_IP = "192.168.250.241"
PASSWORD = ""


# Get the TX sector value using the RouterOS library
def set_tx_sector(sectorNo):
    # try:
    # Connect to the Mikrotik router using RouterOS API over SSH
    connection = routeros_api.RouterOsApiPool(ROUTER_IP, username=USERNAME, password=PASSWORD, plaintext_login=True)
    api = connection.get_api()
    api.get_resource('/interface/w60g').call('set',{'numbers':'0', 'tx-sector':str(sectorNo)})
    connection.disconnect()


def computeAlphaAtSTA(x1, y1, x2, y2,phi,imu_sta1_phi,imu_sta2_phi):
    # Orientation angle is the angle between STA1's axes and global axes
    gamma=np.arctan2(y2-y1,x2-x1)*180/np.pi;
    alpha1 = imu_sta1_phi-phi+gamma;
    alpha2 = 180-(imu_sta2_phi-phi+gamma);    
    return alpha1, alpha2;
    
def getTxSector(alpha):
    if alpha > 22.8:
        txSector = 24;
    elif alpha <= 22.8 and alpha > 15.2:
        txSector = 25;
    elif alpha <= 15.2 and alpha > 7.6:
        txSector = 26;
    elif alpha <= 7.6 and alpha > 0:
        txSector = 27;
    elif alpha < -22.8:
        txSector = 31;
    elif alpha >= -22.8 and alpha < -15.2:
        txSector = 30;
    elif alpha >= -15.2 and alpha < -7.6:
        txSector = 29;
    elif alpha >= -7.6 and alpha <= 0:
        txSector = 28;
    	
    return txSector

if __name__ == "__main__":

    phi=45
    imu_sta2_phi=120
    sta1_txSector = 0
    started = 0
    sta1_latest_sector = 0

    #pipe_path = '/home/roshni/Downloads/rxdata'
    #pipe_fd = os.open(pipe_path, os.O_RDONLY)
    #pipe_file = os.fdopen(pipe_fd)
    
    while True:
        
        print("1. For calculating angle\n2. Sending sector to station\n")
        val = int(input())
        if val == 1:
            if started == 0:
                phi = input("Enter angle between radar axes and magnetic north: ")
                phi = float(phi)
                imu_sta2_phi = input("Enter angle between STA 2 axes and magnetic north: ")
                imu_sta2_phi = float(imu_sta2_phi)
                started = 1
    
            action = input("Collected IMU values? y/n ")
            if action == 'y':
                xfile = input("IMU data file")
                df = pd.read_excel(xfile+'.xls')
                df = df.to_numpy()
                angle = []
                sum = 0
                for i in range(0,100):
                    angle.append(np.arctan2(df[i][2],df[i][1]))
                imu_sta1_phi = statistics.mean(angle)
            
            with open('locData.txt', 'r+') as f:
                if os.path.getsize('locData.txt') != 0:
                    last_line = f.readlines()[-1]
                    f.truncate(0)
                    coordinates = last_line.split(',')
                    x1 = float(coordinates[0])
                    y1 = float(coordinates[1])
                    x2 = float(coordinates[2])
                    y2 = float(coordinates[3])
                    print(x1,y1,x2,y2)
            

            # Get sta1_theta, sta2_theta from STA
            [alpha1, alpha2] = computeAlphaAtSTA(x1, y1, x2, y2,phi,imu_sta1_phi,imu_sta2_phi);
            sta1_txSector = getTxSector(alpha1) # TODO: Change tx sector
            set_tx_sector(sta1_txSector)
            sta1_latest_sector = getTxSector(alpha2)
        
        elif val ==2:
            #get tx-sector
            connection = routeros_api.RouterOsApiPool(ROUTER_IP, username=USERNAME, password=PASSWORD, plaintext_login=True)
            api = connection.get_api()
            sector_val = api.get_resource('/interface/w60g').get()[0]['tx-sector']
            if sta1_txSector != 0 and int(sta1_txSector) == int(sector_val):

                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                server_address = ('192.168.250.20', 9999)
                print(f'Sending sector value = {sta1_latest_sector}')
                sock.sendto(str(sta1_latest_sector).encode(), server_address)
                connection.disconnect()
        
        else:
            continue
        # fsend=open("alpha.txt", "w")
        # fsend.write(str(sta1_latest_sector))
        # fsend.close()     
