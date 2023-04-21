import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from munkres import Munkres, print_matrix
import sys
import os

# global constants wont change
eps = 0.16
min_pts = 250
alpha = 0.25
L = np.array([10000,10000,10000]) #Limit type1
L2 = 10000 #Limit type2
thresh = 5 #threshold for the dsitance above which the distance is made L2
maxFails = 10 #Maximum number of frames for which a track is checked. Failure after maxFails will lead to removal of the track

#structs monitoring tracks
global tracks
global track_status
global counter
tracks = []
track_status = []
counter = 0

def frame_processing(frame):
    fr =  frame
    xyz = fr[['X','Y','Z']].to_numpy()
    xyz[:,2] = 0.0 #Removing the z coordinate data
    labels = dbscan(xyz) #Performs dbscan

    doppler_array = fr[['doppler']].to_numpy()
    centroid_array = []
    centroid_doppler = []
    centroid_label = []

    labelList = set(labels)
    for x in labelList:
        if x == -1:
            continue
        points_in_cluster = xyz[labels==x,:]
        dopplers_in_cluster = doppler_array[labels==x,:]
        centroid = np.mean(points_in_cluster,axis=0)
        mean_doppler = np.mean(dopplers_in_cluster,axis=0)
        centroid_array.append(centroid)
        centroid_doppler.append(mean_doppler)
        centroid_label.append(x)
    # if not (centroid_array or centroid_label):
    #     continue
    ####*** Need to see what happens if no centroids exist in a frame ***#####
    tracks_update(centroid_array,centroid_doppler,centroid_label)
    
def dbscan(xyz):
    scaled_points = StandardScaler().fit_transform(xyz)
    print(f"Scaled points are : Max = {max(scaled_points[0])}, Min = {min(scaled_points[0])}")
    ##Clustering frame using dbscan
    model = DBSCAN(eps= eps,min_samples= min_pts)
    model.fit(scaled_points)
    return model.labels_

'''
Centroid doppler and label not being used
'''
def tracks_update(centroid_array, centroid_doppler,centroid_label):

    global tracks
    global track_status
    global counter

    print("\nFrame number = ", counter)
    counter += 1
    if not tracks:
        for coordinate in centroid_array:
            tracks.append([coordinate])
        track_status = [0]*len(tracks)
        return
    
    #Pick the last coordinates from each track. Also, convert current frame to a list of numpy arrays
    track_points = extract_last_points(tracks) #1. Last coordinates from track
    fr_points = convert(centroid_array) #2. Convert the numpy array of array to lsit of numpy arrays

    print(f"Frame {counter} statistics:")
    print(f"tracks length = {len(tracks)}")
    print(f"fr_points length = {len(fr_points)}")
    print(f"This set points must be returned = {fr_points}?")

    #Padding the shorter array with L, where L is a far away coordinate wrt all axes.
    lendiff = len(track_points) - len(fr_points)
    appender = [L]*abs(lendiff)
    if lendiff > 0:
        fr_points.extend(appender)
    elif lendiff < 0:
        track_points.extend(appender)

    #Create matrix of distances from the two lists
    matrix = dist_matrix(track_points,fr_points)
    
    #Applying Hungarian algorithm
    m = Munkres()
    indexes = m.compute(matrix)

    for row, column in indexes:
        d = matrix[row][column]
        print(f'({row}, {column}) -> {d}')
        if d < L2:
            #Adding the point fr_points[column] to the tracks[row]
            tracks[row].append(fr_points[column])

    for row, column in indexes:
        d = matrix[row][column]
        if d >= L2:
            #1. Process the tracks and track_status arrays.
            update_fail_tracks(row)
            #2. Add the cluster from the current frame as a new track.
            add_to_tracks(fr_points[column],column)
    #Delete tracks that are marked for deletion
    #Visualizing
    #visualize()
    

def dist(p1, p2):
    #Calculating distance between two points. Z-axis is weighted to reduce importance.
    if all(p1 == L) or all(p2 == L):
        return L2
    distance = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 +  alpha*(p1[2]-p2[2])**2
    return distance

def extract_last_points(arr):
    return [item[-1] for item in arr]

def convert(arr):   
    lister = [] 
    for x in arr:
        lister.append(x)
    return lister

def dist_matrix(m,n):
    #Create distance matrix from two arrays
    mat = []
    for i in range(len(m)):
        row = []
        for j in range(len(n)):
            row.append(dist(m[i],n[j]))
        mat.append(row)
    mat = np.array(mat)
    #Thresholding the matrix so that any value above thresh is made L2
    mat[mat > thresh] = L2
    mat = mat.tolist()
    return mat

def update_fail_tracks(ind):
    if ind < len(track_status):
        track_status[ind] += 1

def delete_tracks():
    i = 0
    length = len(track_status)
    #If failure happens for maxFails times
    while i < length:
        if track_status[i] >= maxFails:
            del tracks[i]
            del track_status[i]
            i -= 1
            length -= 1
        i += 1

def add_to_tracks(point,column):
    #If the point was an augmented point, ignore
    if all(point == L):
        return
    #Else add point as a new track
    tracks.append([point])
    track_status.append(0)

def visualize( ):
    printset_master = []
    print(f"Tested {counter} frames")
    labelind = 0
    printset = np.array(tracks[0][-1])
    # print("printset is ",printset)
    printlabels = np.array([labelind])#*len(tracks[0]))
    # print("printlabels is ",printlabels)
    i = 0
    for row in tracks:
        if i == 0:
            i +=1
            continue
        printset = np.vstack([printset,np.array(row[-1])])
        labelind+=1
        printlabels = np.append(printlabels, [labelind])
    printset = np.vstack([printset,np.array([0,0,0])])
    print("printset is \n", printset)
    pcd_track = o3d.geometry.PointCloud()
    pcd_track.points = o3d.utility.Vector3dVector(printset)
    numClusters = len(printlabels)
    colors_track = plt.get_cmap("tab10")(printlabels/ (numClusters if numClusters > 0 else 1))
    pcd_track.colors= o3d.utility.Vector3dVector(colors_track[:,:3])

    printset_master.append(printset)
    #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.001, origin=[0, 0, 0])
    # o3d.visualization.RenderOption(point_size=2)
    print(f"Length of tracks is {len(tracks)}")
    o3d.visualization.draw_geometries([pcd_track], mesh_show_wireframe=True)


def preprocess(df):
    header = pd.DataFrame([float(x) for x in df.columns]).T #Current Header is meant to be the first row
    df.columns = ['timestamp', 'frame' , 'point' , 'X', 'Y', 'Z', 'doppler', 'intensity'] #renaming rows
    header.columns = df.columns #Using the same columns of dataframs as the column names for the header.
    df = pd.concat([header,df]).reset_index(drop=True) #Concatenating the first row that was previously the header with the rest of the dataframe.
    frames = dict(tuple(df.groupby('frame')))
    return frames

if __name__ == '__main__':
    # if frame start processing, else keep reading
    #df = pd.read_csv('Apr10/walk_stand_1.csv')
    
    #open tx-sector files

    #frames= preprocess(df)
    #keys = list(frames.keys())
    
    pipe_path = '/home/marga3/group4/anotherrec'
    pipe_fd = os.open(pipe_path, os.O_RDONLY)
    pipe_file = os.fdopen(pipe_fd)
        
    print('Pipe opened, lets hope nothing blows up')
    active_frame = ''
    current_frame = ''
    df_columns = ['timestamp', 'frame' , 'point' , 'X', 'Y', 'Z', 'doppler', 'intensity']
    df = pd.DataFrame(columns = df_columns) 
    while True:
        line = pipe_file.readline()
        line = line.split(',')
        line = [float(x) for x in line]
        
        if not current_frame == '':
            current_frame = line[1]
        
        if active_frame == '' and current_frame == '':
            active_frame = line[1]
            current_frame = line[1]
        
        if active_frame == current_frame:
            df.loc[len(df)] = line 
        
        else:
            print('******************Frame starts******************')
            frame_processing(df)
            print('******************Frame ends******************\n\n')
            df = pd.DataFrame(columns = df_columns)
            active_frame = current_frame
        
        if not line:
            break
