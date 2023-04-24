import sys
import socket
import time
import os
import math
import numpy as np
import pandas as pd
import statistics

# Assumptions: 
# 1. Only STA 1 moves 
# 2. Both STA 1 and STA 2 are within the beam width of each other

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
    
    started = 0

    #pipe_path = '/home/roshni/Downloads/rxdata'
    #pipe_fd = os.open(pipe_path, os.O_RDONLY)
    #pipe_file = os.fdopen(pipe_fd)
    
    while True:
        #line = pipe_file.readline()
        #print(line)
        
        if started == 0:
            phi = input("Enter angle between radar axes and magnetic north: ")
            phi = int(phi)
            imu_sta2_phi = input("Enter angle between STA 2 axes and magnetic north: ")
            imu_sta2_phi = int(imu_sta2_phi)
            started = 1
   
        action = input("Collected IMU values? y/n ")
        if action == 'y':
            df = pd.read_excel('a.xls')
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
                x1 = int(coordinates[0])
                y1 = int(coordinates[1])
                x2 = int(coordinates[2])
                y2 = int(coordinates[3])
                print(x1,y1,x2,y2)
		

	    # Get sta1_theta, sta2_theta from STA
        [alpha1, alpha2] = computeAlphaAtSTA(x1, y1, x2, y2,phi,imu_sta1_phi,imu_sta2_phi);
        sta1_txSector = getTxSector(alpha1) # TODO: Change tx sector
        sta2_txSector = getTxSector(alpha2)

        fsend=open("alpha.txt", "w")
        fsend.write(str(sta2_txSector))
        fsend.close()     
        
                
        
