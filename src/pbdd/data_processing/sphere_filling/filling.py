import numpy as np
import torch
import torch_geometric
from rdkit import Chem
from scipy.spatial import distance
from torch_geometric.data import Data

from pbdd.data_processing.sphere_filling.usr import USR
from pbdd.data_processing.utils import (distance_match, get_connectivity,
                                        get_nearest_distance, rotate_coord,
                                        simplify_atomtype, translate_coord)

# Reference: https://github.com/Tankx/Equal-sphere-packing

class Pack:
    def __init__(self, x, y, z, r):
        self.x = x  # Length of domain
        self.y = y  # Width of domain
        self.z = z  # Height of domain
        self.r = r  # Node radius

    def simple_cubic(self):
        x_nodes = int(self.x // (self.r * 2))
        y_nodes = int(self.y // (self.r * 2))
        z_nodes = int(self.z // (self.r * 2))
        
        coordinates = []

        for i in range (x_nodes + 1):
            for j in range (y_nodes + 1):
                for k in range (z_nodes + 1):
                    x_val = i * self.r * 2 + self.r
                    y_val = j * self.r * 2 + self.r
                    z_val = k * self.r * 2 + self.r
                    if (x_val < (self.x - self.r) and y_val < (self.y - self.r) and z_val < (self.z - self.r) and 
                        x_val >= 0 and y_val >= 0 and z_val >= 0):
                        coordinates.append([x_val, y_val, z_val])
        return np.array(coordinates)

    def body_centered(self):
        root = (np.sqrt(3/4))
        limit = int(max(self.x/self.r, self.y/self.r, self.z/self.r))
        
        coordinates = []

        for i in range(limit):
            for j in range(limit):
                for k in range(limit):
                    x_val = ((0.5*(-i+j+k)) / root) * self.r * 2 + self.r
                    y_val= ((0.5*(i-j+k)) / root) * self.r * 2 + self.r
                    z_val = ((0.5*(i+j-k)) / root) * self.r * 2 + self.r
                    if (x_val < (self.x - self.r) and y_val < (self.y - self.r) and z_val < (self.z - self.r) and 
                        x_val >= 0 and y_val >= 0 and z_val >= 0):
                        coordinates.append([x_val, y_val, z_val])
        return np.array(coordinates)

    def face_centered(self):
        root = np.sqrt(0.5)
        limit = int(max(self.x/self.r, self.y/self.r, self.z/self.r))

        coordinates = []

        for i in range(-limit, limit):
            for j in range(-limit, limit):
                for k in range(-limit, limit):
                    x_val = ((0.5 * (j + k)) / root) * self.r * 2 + self.r
                    y_val = ((0.5 * (i + k)) / root) * self.r * 2 + self.r
                    z_val = ((0.5 * (i + j)) / root) * self.r * 2 + self.r
                    if (x_val < (self.x - self.r) and y_val < (self.y - self.r) and z_val < (self.z - self.r) and 
                        x_val >= 0 and y_val >= 0 and z_val >= 0):
                        coordinates.append([x_val, y_val, z_val])
        return np.array(coordinates)

    def hex_centered(self):
        x_nodes = int(self.x // self.r)
        y_nodes = int(self.y // self.r)
        z_nodes = int(self.z // self.r)
        
        coordinates = []

        for i in range(x_nodes + 1):
            for j in range(y_nodes + 1):
                for k in range(z_nodes + 1):
                    x_val = (2 * i + ((j + k) % 2)) *self.r + self.r
                    y_val = (np.sqrt(3) * (j + (1 / 3) * (k % 2))) * self.r + self.r
                    z_val = (((2 * np.sqrt(6)) / 3) * k) * self.r + self.r
                    if (x_val < (self.x - self.r) and y_val < (self.y - self.r) and z_val < (self.z - self.r) and 
                        x_val >= 0 and y_val >= 0 and z_val >= 0):
                        coordinates.append([x_val, y_val, z_val])
        return np.array(coordinates)
    
def mol2sphereG(mol,d2,fill_mode='fcc',radius=1.6,n_max_nodes=50,boundary=0.5,\
                reverse=False,fill_match=True,n_max_spheres:int=80):
    # Convert a molecule to a sphere graph
    # if fill_match is False, ignore the bond information
    # bonds = mol.GetBonds()
    atoms = mol.GetAtoms()
    n_nodes = len(atoms)
    if n_nodes<10:
        return None
    if n_nodes>n_max_nodes and n_max_nodes!=-1:
        return None
    # N_edges = len(bonds)
    atom_type = torch.zeros((n_nodes,1),dtype=torch.long)
    # edge_type = torch.zeros([N_edges],dtype=torch.float)
    # mol_edge_index = torch.zeros((2,N_edges),dtype=torch.long)
    mol_pos = torch.zeros((n_nodes,3),dtype=torch.float)
    # for i in range(N_edges):
    #     bond = bonds[i]
    #     mol_edge_index[0,i] = bond.GetBeginAtomIdx()
    #     mol_edge_index[1,i] = bond.GetEndAtomIdx()
    #     edge_type[i] = bond.GetBondTypeAsDouble()
    # print(edge_type)
    for j in range(n_nodes):
        atom = atoms[j]
        atom_type[j,0] = atom.GetAtomicNum()
        #node_f[j,1] = atom.GetFormalCharge()
        coor = mol.GetConformer().GetAtomPosition(j)
        mol_pos[j,0],mol_pos[j,1],mol_pos[j,2] = coor.x,coor.y,coor.z
    # print(atom_type)
    # new_atomtype = simplify_atomtype(atom_type)
    # if new_atomtype is None:
        # return None
    # new_atomtype = torch.tensor(new_atomtype,dtype=torch.long)
    # print(new_atomtype)
    # print(mol_pos)
    pos_x_min,pos_x_max = torch.min(mol_pos[:,0]),torch.max(mol_pos[:,0])
    pos_y_min,pos_y_max = torch.min(mol_pos[:,1]),torch.max(mol_pos[:,1])
    pos_z_min,pos_z_max = torch.min(mol_pos[:,2]),torch.max(mol_pos[:,2])
    pos_center = torch.tensor([(pos_x_min+pos_x_max)/2,(pos_y_min+pos_y_max)/2,(pos_z_min+pos_z_max)/2])
    mol_pos = mol_pos-pos_center
    # print(pos_center)
    # print(mol_pos)
    if fill_match:
        bonds = mol.GetBonds()
        n_edges = len(bonds)
        edge_type = torch.zeros([n_edges],dtype=torch.float)
        mol_edge_index = torch.zeros((2,n_edges),dtype=torch.long)
        for i in range(n_edges):
            bond = bonds[i]
            mol_edge_index[0,i] = bond.GetBeginAtomIdx()
            mol_edge_index[1,i] = bond.GetEndAtomIdx()
            edge_type[i] = bond.GetBondTypeAsDouble()
        new_atomtype = simplify_atomtype(atom_type)
        if new_atomtype is None:
            return None
        new_atomtype = torch.tensor(new_atomtype,dtype=torch.long)
        filltest=True
        for i in range(15):
            if reverse:
                sphere_clouds,matched_spheres,matched_ids = random_fill_pos_reverse(mol_pos,boundary=boundary,N=1,radius=radius,fill_mode=fill_mode)
            else:
                sphere_clouds,matched_spheres,matched_ids = random_fill_pos(mol_pos,boundary=boundary,N=1,radius=radius,fill_mode=fill_mode)
            if len(sphere_clouds)==0:
                continue
            else:
                sphere_temp = sphere_clouds[0]
                matched_temp = matched_spheres[0]
                matched_id = matched_ids[0]
            # print(f'matched_id: {matched_id}')
            d_pairs = list()
            d_pairs2 = list()
            global_e1 = list()
            global_e2 = list()
            for i,j in zip(*mol_edge_index):
                i = int(i)
                j = int(j)
                x1 = matched_temp[i,:]
                x2 = matched_temp[j,:]
                global_e1.append(matched_id[i])
                global_e2.append(matched_id[j])
                # x11 = sphere_temp[matched_id[i]]
                # x12 = sphere_temp[matched_id[j]]
                deltax = np.linalg.norm(x1-x2)
                # deltax2 = np.linalg.norm(x11-x12)
                # print(deltax,deltax2)
                # d_possible.add(np.around(deltax,3))
                d_pairs.append(deltax)
            d_max = max(d_pairs)
            global_edges = [global_e1,global_e2]
            if d_max<d2/2+0.01:
                filltest=False
                break
        if filltest:
            return None
        n_spheres = len(sphere_temp)
        e1,e2,e3 = get_connectivity(sphere_temp,'fcc',radius/2)
        N_total_edges = len(e2[0])
        usr = USR(sphere_temp)
        usr_temp = usr.calcUSRRDescriptors()
        # usr2 = USR(lig_pos)
        # usr_temp2 = usr2.calcUSRRDescriptors()
        # print('USR')
        # print(usr_temp)
        # print(usr_temp2)
        # atomtype_x = torch.zeros((N_spheres,1))
        atomtype_y = torch.zeros((n_spheres,1),dtype=torch.long)
        atomtype_y[matched_id,0] = new_atomtype
        edge_attr_x = torch.ones((N_total_edges,1))
        edge_attr_y = torch.zeros((N_total_edges,1))
        
        # print(N_total_edges)
        # print(len(d_pairs))
        # print(filltest)
        # print(len(global_edges[0]))
        # print(len(edge_type))
        # print(global_edges)
        # cnt_edge = 0
        
        for i,j,edge_double in zip(*global_edges,edge_type):
            for k in range(N_total_edges):
                m = e2[0][k]
                n = e2[1][k]
                if (i==m and j==n) or (i==n and j==m):
                    # cnt_edge+=1
                    if edge_double==0:
                        edge_attr_y[k] = 1
                    else:
                        edge_attr_y[k] = edge_double
        # print(torch.sum(edge_attr_y))
        # print(cnt_edge)
        edge_attr = torch.cat((edge_attr_x,edge_attr_y),dim=1)
        # print(f'edge_attr {edge_attr}')
        # print(edge_attr.shape)
        e2 = torch.tensor(e2,dtype=torch.long)
        global_edges = torch.tensor(global_edges,dtype=torch.long)
        edge_index,edge_attr = torch_geometric.utils.to_undirected(e2,edge_attr)
        global_edges = torch_geometric.utils.to_undirected(global_edges)
        degree_x = torch_geometric.utils.degree(edge_index[0],n_spheres).unsqueeze(1)
        degree_y = torch_geometric.utils.degree(global_edges[0],n_spheres).unsqueeze(1)
        sphere_temp = torch.tensor(sphere_temp,dtype=torch.float)
        x = torch.cat((sphere_temp,degree_x),dim=1)
        y = torch.cat((atomtype_y,degree_y),dim=1)
        # print(x.shape)
        # print(y.shape)
        # print('x',x)
        # print('y',y)
        # print(torch.sum(edge_attr_y>0),torch.sum(edge_attr[:,1]>0),torch.sum(degree_y)) 
        d = Data(x,edge_index,edge_attr=edge_attr,y=y,USR=usr_temp,pos=sphere_temp)
    else:
        sphere_clouds,angles = random_fill_pos_nomatch(mol_pos,boundary=boundary)
        n_spheres = len(sphere_clouds)
        # if N_spheres>100, random select consecutive 100 spheres to use
        if n_spheres>n_max_spheres:
            sphere_clouds = select_subclouds(sphere_clouds,n_max_spheres)
            # print(sphere_clouds.shape)
            n_spheres = len(sphere_clouds)
        e1,e2,e3 = get_connectivity(sphere_clouds,'fcc',radius/2)
        N_total_edges = len(e2[0])
        e2 = torch.tensor(e2,dtype=torch.long)
        edge_attr = torch.ones((N_total_edges,2))
        edge_index,edge_attr = torch_geometric.utils.to_undirected(e2,edge_attr)
        # e2 = torch.tensor(e2,dtype=torch.long)
        degree_x = torch_geometric.utils.degree(edge_index[0],n_spheres).unsqueeze(1)
        sphere_clouds = torch.tensor(sphere_clouds,dtype=torch.float)
        x = torch.cat((sphere_clouds,degree_x),dim=1)
        d = Data(x,edge_index,edge_attr,pos_center=pos_center,rot_angles=angles,pos=sphere_clouds)
    return d


def select_subclouds(sphere_clouds,n_spheres,radius=1.6):
    # select consecutive n_spheres from sphere_clouds
    # sphere_clouds: n_spheres * 3 (coordinates)
    def is_adjacent(point, selected_points, radius=radius):
        for sp in selected_points:
            if np.linalg.norm(point - sp) < radius+0.01:
                return True
        return False

    n1 = len(sphere_clouds)
    print('number of spheres:', n1)
    if n1<=n_spheres:
        return sphere_clouds
    else:
        flag = True
        while flag:
            selected_indices = [np.random.randint(0,n1)]
            selected_points = [sphere_clouds[selected_indices[0]]]
            while len(selected_points) < n_spheres:
                # Calculate the centroid of the current shape
                centroid = np.mean(selected_points, axis=0)
            # Find all eligible (adjacent and not yet selected) points
                eligible_points = []
                for i, point in enumerate(sphere_clouds):
                    if i not in selected_indices and is_adjacent(point, selected_points, radius=radius):
                        eligible_points.append((i, point))
                if len(eligible_points) == 0: # If no eligible points, break to avoid infinite loop
                    flag = False
                    break
                # Apply heuristic: distance from centroid + randomness, to eligible points
                costs = [np.linalg.norm(point - centroid) for _, point in eligible_points]
                # normalize costs
                costs = np.array(costs)
                costs = costs/np.max(costs)
                # add randomness
                costs = costs + 1.5 * np.random.rand(len(costs))
                next_point_index, next_point = max(zip(eligible_points, costs), key=lambda x: x[1])[0]
                # Add the selected point
                selected_indices.append(next_point_index)
                selected_points.append(next_point)
            # print(selected_points)
            if len(selected_points) == n_spheres:
                break
        sub_cloud = np.array(selected_points)
        return sub_cloud




def random_fill_pos(lig_pos,boundary=0.5,N=5,fill_mode='fcc',size=48,radius=0.8*2,max_attempts=50):
    '''
    fill_mode choice: 'scp','bcc', 'fcc', 'hcp'
    '''
    packed = Pack(size,size,size,radius)
    assert fill_mode in ['scp','bcc','fcc','hcp'],'fill mode choice \'scp\',\'bcc\',\'fcc\',\'hcp\'.'
    if fill_mode=='scp':
        coors = packed.simple_cubic()
    elif fill_mode=='bcc':
        coors = packed.body_centered()
    elif fill_mode=='fcc':
        coors = packed.face_centered()
    elif fill_mode=='hcp':
        coors = packed.hex_centered()
    rotate_center = [23.5,23.5,23.5]
    direction = [-23.5,-23.5,-23.5]
    i = 0
    k = 0 
    sphere_clouds = list()
    matched_spheres = list()
    matched_ids = list()
    while i<N:
        k+=1
        coors_new = rotate_coord(rotate_center,coors)
        translate_dir = radius*np.random.rand(3)
        coors_new = translate_coord(coors_new,translate_dir)
        coors_new = translate_coord(coors_new,direction)/2
        d1 = distance.cdist(coors_new,lig_pos)
        within = np.any(d1<(radius/2+boundary),axis=1)
        in_points = coors_new[within]
        d2 = distance.cdist(lig_pos,in_points)
        print(d2.shape)
        closest = distance_match(d2)
        matched = in_points[closest]
        if closest is None:
            # print(f'no matching, attempt{k}')
            if k>=max_attempts:
                break
            else:
                continue
        sphere_clouds.append(in_points)
        matched_spheres.append(matched)
        matched_ids.append(closest)
        i+=1
    return sphere_clouds,matched_spheres,matched_ids

def random_fill_pos_nomatch(lig_pos,boundary=0.6,fill_mode='fcc',size=48,radius=0.8*2):
    '''
    fill_mode choice: 'scp','bcc', 'fcc', 'hcp'
    '''
    packed = Pack(size,size,size,radius)
    assert fill_mode in ['scp','bcc','fcc','hcp'],'fill mode choice \'scp\',\'bcc\',\'fcc\',\'hcp\'.'
    if fill_mode=='scp':
        coors = packed.simple_cubic()
    elif fill_mode=='bcc':
        coors = packed.body_centered()
    elif fill_mode=='fcc':
        coors = packed.face_centered()
    elif fill_mode=='hcp':
        coors = packed.hex_centered()
    direction = [-23.5,-23.5,-23.5]
    coors = translate_coord(coors,direction)/2
    # lig_x_min,lig_x_max = np.min(lig_pos[:,0]),np.max(lig_pos[:,0])
    # lig_y_min,lig_y_max = np.min(lig_pos[:,1]),np.max(lig_pos[:,1])
    # lig_z_min,lig_z_max = np.min(lig_pos[:,2]),np.max(lig_pos[:,2])
    # lig_x_mean = (lig_x_min+lig_x_max)/2
    # lig_y_mean = (lig_y_min+lig_y_max)/2
    # lig_z_mean = (lig_z_min+lig_z_max)/2
    # lig_new_pos = translate_coord(lig_pos,[-lig_x_mean,-lig_y_mean,-lig_z_mean]) 
    if isinstance(lig_pos,torch.Tensor):
        lig_pos = lig_pos.numpy()
    lig_coors_new,angels = rotate_coord([0,0,0],lig_pos,return_angles=True)
    d1 = distance.cdist(coors,lig_coors_new)
    within = np.any(d1<(radius/2+boundary),axis=1)
    in_points = coors[within]
    # print(in_points)
    return in_points,angels

def random_fill_pos_reverse(lig_pos,boundary=0.5,N=5,fill_mode='fcc',size=48,radius=0.8*2,max_attempts=50):
    # fill_mode choice: 'scp','bcc', 'fcc', 'hcp'
    packed = Pack(size,size,size,radius)
    assert fill_mode in ['scp','bcc','fcc','hcp'],\
        'fill mode choice \'scp\',\'bcc\',\'fcc\',\'hcp\'.'
    if fill_mode=='scp':
        coors = packed.simple_cubic()
    elif fill_mode=='bcc':
        coors = packed.body_centered()
    elif fill_mode=='fcc':
        coors = packed.face_centered()
    elif fill_mode=='hcp':
        coors = packed.hex_centered()
    direction = [-23.5,-23.5,-23.5]
    coors = translate_coord(coors,direction)/2
    # lig_x_min,lig_x_max = np.min(lig_pos[:,0]),np.max(lig_pos[:,0])
    # lig_y_min,lig_y_max = np.min(lig_pos[:,1]),np.max(lig_pos[:,1])
    # lig_z_min,lig_z_max = np.min(lig_pos[:,2]),np.max(lig_pos[:,2])
    # lig_x_mean = (lig_x_min+lig_x_max)/2
    # lig_y_mean = (lig_y_min+lig_y_max)/2
    # lig_z_mean = (lig_z_min+lig_z_max)/2
    # lig_new_pos = translate_coord(lig_pos,[-lig_x_mean,-lig_y_mean,-lig_z_mean]) 
    if isinstance(lig_pos,torch.Tensor):
        lig_pos = lig_pos.numpy()
    i = 0
    k = 0 
    sphere_clouds = list()
    matched_spheres = list()
    matched_ids = list()
    while i<N:
        k+=1
        lig_coors_new = rotate_coord([0,0,0],lig_pos)
        translate_dir = radius*np.random.rand(3)
        lig_coors_new = translate_coord(lig_coors_new,translate_dir)
        # print(coors)
        # print(lig_coors_new)
        d1 = distance.cdist(coors,lig_coors_new)
        within = np.any(d1<(radius/2+boundary),axis=1)
        in_points = coors[within]
        # print(in_points)
        d2 = distance.cdist(lig_coors_new,in_points)
        # print(d2.shape)
        closest = distance_match(d2)
        matched = in_points[closest]
        if closest is None:
            # print(f'no matching, attempt{k}')
            if k>=max_attempts:
                break
            else:
                continue
        sphere_clouds.append(in_points)
        matched_spheres.append(matched)
        matched_ids.append(closest)
        i+=1
    return sphere_clouds,matched_spheres,matched_ids

def get_grids_from_pdb(pdb_file,fill_mode='fcc',radius=1.6,boundary=0.5,\
                       n_max_nodes=-1,n_max_spheres:int=80):
    # cavity pdb can obtained from Ligbuilder
    cavity_mol = Chem.MolFromPDBFile(pdb_file,sanitize=False,removeHs=True,proximityBonding=False)
    n_points = cavity_mol.GetNumAtoms()
    d1,d2,d3 = get_nearest_distance(fill_mode=fill_mode,radius=radius)
    shape = mol2sphereG(cavity_mol,d2,boundary=boundary,fill_match=False,\
                        n_max_nodes=n_max_nodes,n_max_spheres=n_max_spheres)
    return shape

def get_grids_from_mol2(mol2_file,fill_mode='fcc',radius=1.6,boundary=0.5,\
                        n_max_nodes=-1,n_max_spheres:int=80):
    mol = Chem.MolFromMol2File(mol2_file,sanitize=False,removeHs=True)
    d1,d2,d3 = get_nearest_distance(fill_mode=fill_mode,radius=radius)
    shape = mol2sphereG(mol,d2,boundary=boundary,fill_match=False,\
                        n_max_nodes=n_max_nodes,n_max_spheres=n_max_spheres)
    return shape
