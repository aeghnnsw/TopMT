import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import skew

class USR():
    # USR stands for Ultrafast Shape Recognition
    def __init__(self,coors):
        self.N_atom = len(coors)
        assert self.N_atom>0, 'No points'
        self.coors = np.array(coors)
        self.centroid = None
        self.CalcCentroid()
        self.cloest = None
        self.furthest = None
        self.furthest2 = None
        self.CalcCloseAndFurthestAtoms()
        self.usr = np.zeros(12)

    def CalcCentroid(self):
        '''
        calculate the centroied of points
        '''
        self.centroid = np.mean(self.coors,0)
    
    def CalcCloseAndFurthestAtoms(self):
        '''
        return the id of the cloest and furthest point to centroid and the furthest to the furthest poing
        '''
        centroid = self.centroid.reshape(1,-1)
        d1 = distance_matrix(centroid,self.coors)
        self.cloest = np.argmin(d1)
        self.furthest = np.argmax(d1)
        furthest = self.coors[self.furthest].reshape(1,-1)
        d2 = distance_matrix(furthest,self.coors)
        self.furthest2 = np.argmax(d2)

    def MomentsToPoint(self,center,coors):
        center = center.reshape(1,-1)
        N = len(coors)
        d = distance_matrix(center,coors).reshape(-1)
        m1 = np.mean(d)
        m2 = np.std(d)
        m3 = skew(d)
        #print(m1,m2,m3)
        return m1,m2,m3

    def calcUSRRDescriptors(self):
        center1 = self.centroid
        self.usr[0],self.usr[1],self.usr[2] = self.MomentsToPoint(center1,self.coors)
        center2 = self.coors[self.cloest]
        coors2 = np.delete(self.coors,self.cloest,0)
        self.usr[3],self.usr[4],self.usr[5] = self.MomentsToPoint(center2,coors2)
        center3 = self.coors[self.furthest]
        coors3 = np.delete(self.coors,self.furthest,0)
        self.usr[6],self.usr[7],self.usr[8] = self.MomentsToPoint(center3,coors3)
        center4 = self.coors[self.furthest2]
        coors4 = np.delete(self.coors,self.furthest2,0)
        self.usr[9],self.usr[10],self.usr[11] = self.MomentsToPoint(center4,coors4)
        return self.usr