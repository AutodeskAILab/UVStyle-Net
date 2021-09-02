import struct

import matplotlib.pyplot as plt
import numpy as np


def read_binary_stl(file):
    #with open(file,"rb") as fichier :
    data = open(file).read()
    nbtriangles=data[80] + data[81] + data[82] + data[83] 
    #print(nbtriangles)
    nbtriangles=struct.unpack("<I", nbtriangles)[0]
    #print(nbtriangles)
    v1 = []
    v2 = [] 
    v3 = []
    for x in range(0,nbtriangles):
        xc = data[84+x*50] + data[85+x*50] + data[86+x*50] + data[87+x*50]
        yc = data[88+x*50] + data[89+x*50] + data[90+x*50] + data[91+x*50]
        zc = data[92+x*50] + data[93+x*50] + data[94+x*50] + data[95+x*50]
        xc = str(struct.unpack('f',xc)[0])
        yc = str(struct.unpack('f',yc)[0])
        zc = str(struct.unpack('f',zc)[0])
        #print(xc,yc,zc)
        vertexs = []
        for y in range(1,4):
            xc = data[84+y*12+x*50] + data[85+y*12+x*50] + data[86+y*12+x*50] + data[87+y*12+x*50]
            yc = data[88+y*12+x*50] + data[89+y*12+x*50] + data[90+y*12+x*50] + data[91+y*12+x*50]
            zc = data[92+y*12+x*50] + data[93+y*12+x*50] + data[94+y*12+x*50] + data[95+y*12+x*50]
            xc = struct.unpack('f',xc)[0]
            yc = struct.unpack('f',yc)[0]
            zc = struct.unpack('f',zc)[0]
            vertexs.append([xc,yc,zc])
            # v1s.append(v1)
            # v2s.append(v2)
            # v3s.append(v3)
        #print(vertexs)
        v1.append(vertexs[0])
        v2.append(vertexs[1])
        v3.append(vertexs[2])
        #raise "err"
        #print("///")
    #print(v1s, v2s, v3s)
    return v1, v2, v3

def triangle_area_vector(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)

def sample_with_vertices(v1,v2,v3, num_points):
    areas = triangle_area_vector(v1, v2, v3)
    probs = areas/areas.sum()
    weighted_random_indexes = np.random.choice(range(len(areas)), size=num_points, p=probs)
    
    choosen_v1 = v1[weighted_random_indexes]
    choosen_v2 = v2[weighted_random_indexes]
    choosen_v3 = v3[weighted_random_indexes]
    
    u = np.random.rand(num_points, 1)
    v = np.random.rand(num_points, 1)
    is_a_problem = u + v > 1
    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]
    w = 1 - (u + v)
    sampled_points = choosen_v1*u + choosen_v2*v + choosen_v3*w  
    #print(sampled_points.shape)
    return sampled_points

def sketch_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def pc_normalize(pc):
    """ pc: NxC, return NxC """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

