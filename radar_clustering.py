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

eps = 0.15
min_pts = 250
df = pd.read_csv('Apr10/zigzagwalk_1.csv')
alpha = 0.25
L = np.array([10000,10000,10000]) #Limit type1
L2 = 10000 #Limit type2
thresh = 5 #threshold for the dsitance above which the distance is made L2
maxFails = 10#int(sys.argv[1]) #Maximum number of frames for which a track is checked. Failure after maxFails will lead to removal of the track

def dist(p1, p2):
    #Calculating distance between two points. Z-axis is weighted to reduce importance.
    if all(p1 == L) or all(p2 == L):
        return L2
    distance = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + alpha*(p1[2]-p2[2])**2
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

def update_fail_tracks(ind,tracks,track_status):
    if ind < len(track_status):
        track_status[ind] += 1

def delete_tracks(tracks,track_status):
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

def add_to_tracks(point,tracks,track_status,column):
    #If the point was an augmented point, ignore
    if all(point == L):
        return
    #Else add point as a new track
    tracks.append([point])
    track_status.append(0)

#Creating dataframe
header = pd.DataFrame([float(x) for x in df.columns]).T #Header is currently meant to be the first row
df.columns = ['timestamp', 'frame' , 'point' , 'X', 'Y', 'Z', 'doppler', 'intensity'] #renaming rows
header.columns = df.columns #Using the same columns of dataframs as the column names for the header.
df = pd.concat([header,df]).reset_index(drop=True) #Concatenating the first rwo that was previously the header with the rest of the dataframe.

#Breaking dataframes into subframes using frame number column
frames = dict(tuple(df.groupby('frame')))

keys = list(frames.keys())
perFrameCluster = {}
pcd_list = []
for k in keys:
    #Plotting data from a frame before processing it
    fr = frames[k]
    #Converting dataframe to a numpy array
    xyz = fr[['X','Y','Z']].to_numpy()
    xyz[:,2] = 0
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # points = np.asarray(pcd.points).copy()

    ##Normalizing the points
    scaled_points = StandardScaler().fit_transform(xyz)
    print(f"Scaled points are : Max = {max(scaled_points[0])}, Min = {min(scaled_points[0])}")
    ##Clustering frame using dbscan
    model = DBSCAN(eps=eps,min_samples=min_pts)
    model.fit(scaled_points)

    ##Getting the labels
    labels = model.labels_
    n_clusters = len(set(labels))
    perFrameCluster[k] = (fr,labels)

#Converting the clusters to centroid
dict_of_cores = {}
pcd_core_list = []
i = 2
for k,v in perFrameCluster.items():
    f = v[0] # f ==> frame
    f_array = f[['X','Y','Z']].to_numpy()
    doppler_array = f[['doppler']].to_numpy()
    l = v[1] # l ==> label
    centroid_array = []
    centroid_doppler = []
    centroid_label = []
    labelList= set(l)
    for x in labelList:
        if x == -1:
            continue
        points_in_cluster = f_array[l==x,:]
        dopplers_in_cluster = doppler_array[l==x,:]
        centroid = np.mean(points_in_cluster,axis=0)
        mean_doppler = np.mean(dopplers_in_cluster,axis=0)
        centroid_array.append(centroid)
        centroid_doppler.append(mean_doppler)
        centroid_label.append(x)
    if not (centroid_array or centroid_label):
        continue
    dict_of_cores[k] = (np.array(centroid_array),np.array(centroid_label),np.array(centroid_doppler))

#Checking if the cluster is a valid cluster of noise
#Creating a tracks dictionary
tracks = []
track_status = []
alpha = 0.25
counter = 0
printset_master = []

for key,value in dict_of_cores.items():
    counter +=1
    print("\nFrame number = ", counter)
    fr = value[0]
    #If tracks is empty, make each cluster of the fram an independent track
    #Then start again from the next frame
    if not tracks:
        for coordinate in fr:
            tracks.append([coordinate])
        track_status = [0]*len(tracks)
        continue
    
    #Pick the last coordinates from each track. Also, convert current frame to a list of numpy arrays
    track_points = extract_last_points(tracks) #1. Last coordinates from track
    fr_points = convert(fr) #2. Convert the numpy array of array to lsit of numpy arrays

    print(f"Frame {counter} statistics:")
    print(f"tracks length = {len(tracks)}")
    print(f"fr_points length = {len(fr_points)}")
    
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
    # print_matrix(matrix, msg='Lowest cost through this matrix:')
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
            update_fail_tracks(row,tracks,track_status)
            #2. Add the cluster from the current frame as a new track.
            add_to_tracks(fr_points[column],tracks,track_status,column)
    #Delete tracks that are marked for deletion
    delete_tracks(tracks,track_status)

    #Displaing all the points left in the tracks
    if counter % 1 == 0:
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
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        # o3d.visualization.RenderOption(point_size=2)
        print(f"Length of tracks is {len(tracks)}")
        # o3d.visualization.draw_geometries([pcd_track,mesh_frame], mesh_show_wireframe=True)

stacker = np.array([0,0,0])
for ele in printset_master:
    stacker = np.vstack([stacker,ele])
stacker[:,2] = 0
print("printset master is \n",stacker)

pcd_master = o3d.geometry.PointCloud()
pcd_master.points = o3d.utility.Vector3dVector(stacker)
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd_master,mesh_frame])
o3d.visualization.draw_geometries([pcd_master])