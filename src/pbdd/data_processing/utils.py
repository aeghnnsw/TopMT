import numpy as np
from random import randrange
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import networkx as nx
from typing import Optional
from torch_geometric.utils import to_networkx,to_undirected
import torch
from rdkit import Chem
from copy import deepcopy

def distance_match(distance_matrix):
    """
    Distance_matrix:    N_atoms * N_spheres
    Find the non-overlapping sphere ids that can minimize RMSD 
    between atom centers and corresponding spheres
    Only choose extend to second nearest spheres, otherwise, return False
    return new match index or None for failed matching
    """
    min_indices = np.argpartition(distance_matrix,1,axis=1)[:,:2]
    N_atoms,_ = min_indices.shape
    closest_ids = min_indices[:,0]
    next_closest_ids = min_indices[:,1]
    if len(set(closest_ids))==N_atoms:
        return closest_ids
    #new_match = copy.deepcopy(closest_ids)
    index_temp = 0
    for i in range(N_atoms):
        for j in range(i+1,N_atoms):
            if closest_ids[j] == closest_ids[i]:
                if index_temp==0:
                    index_temp = j
                else:
                    return None
        if index_temp!=0:
            index_i = next_closest_ids[i]
            index_j = next_closest_ids[index_temp]
            in_i = (index_i in closest_ids)
            in_j = (index_j in closest_ids)
            if in_i and in_j:
                return None
            elif in_i and (not in_j):
                closest_ids[index_temp] = index_j
            elif (not in_i) and in_j:
                closest_ids[i] = index_i
            else:
                d1 = distance_matrix[i,closest_ids[i]] + distance_matrix[index_temp,index_j]
                d2 = distance_matrix[i,index_i]+distance_matrix[index_temp,closest_ids[index_temp]]
                if d1<d2:
                    closest_ids[index_temp] = index_j
                else:
                    closest_ids[i] = index_i                
            index_temp=0
    return closest_ids

def draw_graph(graph,show=True,return_np=False,save_fig:Optional[str]=None,color:str='steelblue'):
    # input graph is PyG Data
    nx_graph = to_networkx(graph)
    nx_graph = nx_graph.to_undirected()
    fig = plt.figure()
    nx.draw_kamada_kawai(nx_graph,node_size=200,width=3,node_color=color)
    if save_fig is not None:
        fig.savefig(save_fig)    
    if show:
        plt.show()
    if return_np:
        ax = plt.gca()
        ax.axis('off')
        ax.margins(0)
        fig.canvas.draw()
        # fig.tight_layout(pad=0)
        np_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # reshape to C x H x W
        np_fig = np.transpose(np_fig, (2, 0, 1))
        plt.close(fig)
        return np_fig
    plt.close(fig)
    return None

def draw_graph_with_type(graph,show=True,return_np=False,save_fig:Optional[str]=None):
    # graph has edge_index, edge_attr, and x
    # x is the atom type, N_atom * 9
    # atom type:    C:0, N:1, O:2, F:3, P:4, S:5, Cl:6, Br:7
    # atom color:   
    #               C: 909090
    #               N: 3050F8
    #               O: FF0D0D
    #               F: 90E050
    #               P: FF8000
    #               S: FFFF30
    #               Cl: 1FF01F
    #               Br: A62929
    # edge_attr is the bond type, N_bond * 3
    # bond type:    single:0, double:1, triple:2
    atom_type = torch.argmax(graph.x,dim=1).numpy()
    edge_type = torch.argmax(graph.edge_attr,dim=1).numpy()
    atom_labels = ['C','N','O','F','P','S','Cl','Br']
    atom_colors = ['#909090','#3050F8','#FF0D0D','#90E050',\
                  '#FF8000','#FFFF30','#1FF01F','#A62929']
    atom_color_list = [atom_colors[int(i)] for i in atom_type]
    atom_label_dict = {}
    for i,index in enumerate(atom_type):
        atom_label_dict[i] = atom_labels[int(index)]
    nx_graph = to_networkx(graph)
    nx_graph = nx_graph.to_undirected()
    pos = nx.kamada_kawai_layout(nx_graph)
    fig = plt.figure()
    # plot nodes
    nx.draw_networkx_nodes(nx_graph,pos=pos,node_size=350,node_color=atom_color_list)
    # plot labels
    nx.draw_networkx_labels(nx_graph,pos=pos,labels=atom_label_dict)

    edge_width_list = []
    # edge_label_dict = {}

    for i in range(graph.edge_index.shape[1]):
        edge_type_temp = edge_type[i]
        if edge_type_temp==0:
            # edge_label_dict[i] = 'single'
            edge_width_list.append(1.0)
        elif edge_type_temp==1:
            # edge_label_dict[i] = 'double'
            edge_width_list.append(2.0)
        elif edge_type_temp==2:
            # edge_label_dict[i] = 'triple'
            edge_width_list.append(4.0)
    # plot edges
    nx.draw_networkx_edges(nx_graph,pos=pos,width=edge_width_list)
    # plot edge labels
    # nx.draw_networkx_edge_labels(nx_graph,pos=pos,edge_labels=edge_label_dict)
    if save_fig is not None:
        fig.savefig(save_fig)
    if show:
        plt.show()
    if return_np:
        ax = plt.gca()
        ax.axis('off')
        ax.margins(0)
        fig.canvas.draw()
        # fig.tight_layout(pad=0)
        np_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # reshape to C x H x W
        np_fig = np.transpose(np_fig, (2, 0, 1))
        plt.close(fig)
        return np_fig
    plt.close(fig)
    return None

def get_connectivity(coors,fill_mode,radius):
    '''
    convert a list of coordinates to first shell nearest neighbor graph, output is PyG style edge index list
    fill_mode choice: 'scp','bcc', 'fcc', 'hcp'
    'scp': 
            6   nearest
            12  next nearest
            8   next-next nearest
    'bcc':
            8   nearest
            6   next-next nearest
            8   next-next nearest
    'fcc': 
            12  nearest
            6   next nearest
            24  next-next nearest
    'hcp':
            12  nearest
            x   next nearest
            x   next-next nearest
            
    output
        edges_1st:  nearest edges
        edges_2nd:  nearest and next nearest edges
    '''
    assert fill_mode in ['scp','bcc','fcc','hcp'],'fill mode choice \'scp\',\'bcc\',\'fcc\',\'hcp\'.'
    d1 = 2*radius
    if fill_mode=='scp':
        d2 = 2*np.sqrt(2)*radius
        d3 = 2*np.sqrt(3)*radius
    elif fill_mode=='bcc':
        d2 = 4*np.sqrt(3)*radius/3
        d3 = 4*np.sqrt(6)*radius/3
    elif fill_mode=='fcc':
        d2 = 2*np.sqrt(2)*radius
        d3 = 2*np.sqrt(3)*radius
    elif fill_mode=='hcp':
        d2 = 2*np.sqrt(2)*radius
        d3 = 4*np.sqrt(6)*radius/3
        d4 = 2*np.sqrt(3)*radius
    N = len(coors)
    e1_1 = list()
    e1_2 = list()
    e2_1 = list()
    e2_2 = list()
    e3_1 = list()
    e3_2 = list()
    for i in range(N):
        for j in range(i+1,N):
            d_temp = np.linalg.norm(coors[i,:]-coors[j,:])
            if abs(d_temp-d1)<0.01:
                e1_1.append(i)
                e1_2.append(j)
            elif abs(d_temp-d2)<0.01:
                e2_1.append(i)
                e2_2.append(j)
            elif abs(d_temp-d3)<0.01:
                e3_1.append(i)
                e3_2.append(j)
    edges_1st = [e1_1,e1_2]
    edges_2nd = [e1_1+e2_1,e1_2+e2_2]
    edges_3rd = [e1_1+e2_1+e3_1,e1_2+e2_2+e3_2]
    # print(d1)
    # print(d2)
    # print(d3) 
    return edges_1st,edges_2nd,edges_3rd


def get_pocket_graph(protein,lig,cutoff=15):
    # protein rdkit mol object
    # lig rdkit mol object
    pass

def get_lig_center(lig_coord): 
    dims = [lig_coord[:, x] for x in range(lig_coord.shape[1])]
    center = [(x.max() + x.min()) / 2.0 for x in dims]
    return center 

def get_nearest_distance(fill_mode,radius):
    '''
    convert a list of coordinates to first shell nearest neighbor graph, output is PyG style edge index list
    fill_mode choice: 'scp','bcc', 'fcc', 'hcp'
    'scp': 
            6   nearest
            12  next nearest
            8   next-next nearest
    'bcc':
            8   nearest
            6   next-next nearest
            8   next-next nearest
    'fcc': 
            12  nearest
            6   next nearest
            24  next-next nearest
    'hcp':
            12  nearest
            x   next nearest
            x   next-next nearest
            
    output
        edges_1st:  nearest edges
        edges_2nd:  nearest and next nearest edges
    '''
    assert fill_mode in ['scp','bcc','fcc','hcp'],'fill mode choice \'scp\',\'bcc\',\'fcc\',\'hcp\'.'
    d1 = 2*radius
    if fill_mode=='scp':
        d2 = 2*np.sqrt(2)*radius
        d3 = 2*np.sqrt(3)*radius
    elif fill_mode=='bcc':
        d2 = 4*np.sqrt(3)*radius/3
        d3 = 4*np.sqrt(6)*radius/3
    elif fill_mode=='fcc':
        d2 = 2*np.sqrt(2)*radius
        d3 = 2*np.sqrt(3)*radius
    elif fill_mode=='hcp':
        d2 = 2*np.sqrt(2)*radius
        d3 = 4*np.sqrt(6)*radius/3
        d3 = 2*np.sqrt(3)*radius
    return d1,d2,d3 

def perturb_graph(edge_index,edge_attr_r,x_r,alpha:float,mode:str=0):
    # perturb the real graph
    # mode 0: random add edges
    # mode 1: random delete edges
    # mode 2: random change edge
    # random change edge
    # Random select n_edge * alpha edges to perturb
    assert mode in [0,1,2],'mode choice 0 or 1'
    assert alpha>=0 and alpha<=1,'alpha should be in [0,1]'
    # selected_index = torch.tensor(selected_index,dtype=torch.long).to(edge_attr_r.device)
    edge_attr_p = edge_attr_r.clone().squeeze()
    x_p = torch.zeros_like(x_r)
    if mode==0:
        n_edges = len(edge_attr_r)
        n_perturb = int(len(edge_attr_r)*alpha)
        selected_index = np.random.choice(n_edges,n_perturb,replace=False)
        edge_attr_p[selected_index] = 1
        _,edge_attr_p = to_undirected(edge_index,edge_attr_p,reduce='max')
        edge_attr_p = edge_attr_p+torch.randn_like(edge_attr_p)*0.1
        edge_attr_p = torch.clamp(edge_attr_p,0,1)
        # assert (_ == edge_index).all(),'edge index changed'
    elif mode==1:
        n_pos_edges = torch.sum(edge_attr_p>0.5)
        n_delete = int(n_pos_edges*alpha)
        pos_index = torch.argwhere(edge_attr_p>0.5).squeeze().cpu().numpy()
        selected_index = np.random.choice(pos_index,n_delete,replace=False)
        edge_attr_p[selected_index] = 0
        _,edge_attr_p = to_undirected(edge_index,edge_attr_p,reduce='min')
        edge_attr_p = edge_attr_p+torch.randn_like(edge_attr_p)*0.1
        edge_attr_p = torch.clamp(edge_attr_p,0,1)
    elif mode==2:
        n_pos_edges = torch.sum(edge_attr_p>0.5)
        # keep pos neg ratio the same
        n_perturb = int(n_pos_edges*alpha)
        # n_neg_perturb = n_perturb-n_pos_perturb
        pos_index = torch.argwhere(edge_attr_p>0.5).squeeze().cpu().numpy()
        neg_index = torch.argwhere(edge_attr_p<0.5).squeeze().cpu().numpy()
        pos_selected_index = np.random.choice(pos_index,n_perturb,replace=False)
        neg_selected_index = np.random.choice(neg_index,n_perturb,replace=False)
        edge_attr_p[pos_selected_index] = 0
        _,edge_attr_p = to_undirected(edge_index,edge_attr_p,reduce='min')
        edge_attr_p[neg_selected_index] = 1
        _,edge_attr_p = to_undirected(edge_index,edge_attr_p,reduce='max')
        edge_attr_p = edge_attr_p+torch.randn_like(edge_attr_p)*0.1
        edge_attr_p = torch.clamp(edge_attr_p,0,1)
        
    connected_nodes = edge_index[0][edge_attr_p>0.5]
    edge_attr_p = edge_attr_p.unsqueeze(1)
    connected_nodes = torch.unique(connected_nodes)
    x_p[connected_nodes] = 1
    x_p = x_p + torch.randn_like(x_p)*0.1
    x_p = torch.clamp(x_p,0,1)
    return edge_attr_p,x_p

def perturb_graph_type(edge_index,edge_attr,x,alpha:float):
    # edge_attr N_bond * N_bondtype
    # x N_atom * N_atomtype
    # change some of the bond type and atom type
    
    # Random select n_edge * alpha edges to perturb
    n_edges,n_edge_type = edge_attr.shape
    n_perturb_edge = int(len(edge_attr)*alpha)
    edge_index = np.random.choice(n_edges,n_perturb_edge,replace=False)
    edge_attr_p = edge_attr.clone()
    for index in edge_index:
        # change edge type
        edge_attr_p[index,:] = 0
        edge_attr_p[index,np.random.randint(n_edge_type)] = 1
        # change atom type
    n_atoms,n_atom_type = x.shape
    n_perturb_atom = int(len(x)*alpha)
    atom_index = np.random.choice(n_atoms,n_perturb_atom,replace=False)
    x_p = x.clone()
    for index in atom_index:
        x_p[index,:] = 0
        x_p[index,np.random.randint(n_atom_type)] = 1
    return edge_attr_p,x_p

def plot_func(data): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2])
    plt.show()

def rotate_coord(coord_center, coord, return_angles=False): 
    angles = [randrange(180) for x in range(3)]
    r_matrix = R.from_euler('xyz', angles, degrees = True).as_matrix()
    rot_coord = ((coord - coord_center) @ r_matrix) + coord_center
    if return_angles==False:
        return rot_coord
    else:
        return rot_coord,angles

def restore_coord_with_angle(coord_center,coord,angles):
    r_matrix = R.from_euler('zyx', angles, degrees = True).as_matrix()
    rot_coord = ((coord - coord_center) @ r_matrix) + coord_center
    return rot_coord

def restore_origin_coors(coors,pos_center,angles):
    angles_temp = deepcopy(angles)
    if isinstance(coors,torch.Tensor):
        coors = coors.numpy()
    if isinstance(pos_center,torch.Tensor):
        pos_center = pos_center.numpy()
    angles_temp.reverse()
    angles_temp = [-a for a in angles_temp]
    coors_rot = restore_coord_with_angle([0,0,0],coors,angles_temp)
    coors_trans = coors_rot+pos_center
    return coors_trans

def simplify_atomtype(atoms):
    '''
    atoms is a list of atomic nubmer
    Here we only use H:1, C:6, N:7, and O:8
    if there are other tpyes, 
        P:15  -> N:7
        S:16  -> O:8
        F:9   -> H:1
        Cl:17 -> H:1
    And H:1, C:2, N:3, O:4
    if there exists other elements, return None      
    '''
    atom_dict = {1:1,6:2,7:3,8:4,15:3,16:4,9:1,17:1}
    N = len(atoms)
    new_atoms = np.zeros(N)
    for i in range(N):
        key = int(atoms[i])
        if key not in atom_dict.keys():
            return None
        else:
            new_atoms[i] = atom_dict[key]
    return new_atoms

def spheres2SDF(coors,edges,f_sdf):
    element='X'
    N_atoms = len(coors)
    edge_list = list()
    for i,j in zip(edges[0],edges[1]):
        edge_temp1 = [int(i)+1,int(j)+1]
        edge_temp2 = [int(j)+1,int(i)+1]
        if edge_temp1 not in edge_list and edge_temp2 not in edge_list:
            edge_list.append(edge_temp1)
    N_bonds = len(edge_list)
    #print(edges)
    f = open(f_sdf,'w')
    f.write('XXX\nConverted From Graph\n\n')
    f.write(f'{N_atoms:<3}{N_bonds:<3}  0     0  0  0  0  0999 V2000\n')
    for coor in coors:
        line = f'{coor[0]:10.4f}{coor[1]:10.4f}{coor[2]:10.4f} {element:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n'
        f.write(line)
        # print(line)
    for edge in edge_list:
        line = f'{edge[0]:3}{edge[1]:3}  1  0  0  0  0\n'
        f.write(line)
    f.close()
    return None

def strip_mol_graph(graph,edge_attr):
    # strip the mol graph from fully connected graph
    # input graph is a PyG graph object from DGT graph dataset
    edge_attr = edge_attr.squeeze()
    edge_subset = (edge_attr>=0.5).nonzero()
    # print(edge_subset)
    if len(edge_subset)<=1:
        node_subset = list(set(graph.edge_index[0].tolist()))
        edge_subset = edge_subset.squeeze()
    else:
        edge_subset = edge_subset.squeeze()
        node_subset = list(set(graph.edge_index[0][edge_subset].tolist()))
    node_subset = torch.tensor(node_subset,dtype=torch.long)
    g_sub = graph.edge_subgraph(edge_subset)
    g_sub = g_sub.subgraph(node_subset)
    return g_sub

def translate_coord(coord, direction): 
    return np.add(coord, direction)

def get_pocket_graph(mol_p,mol_l,cutoff):
    # Get pocket graph from protein and ligand
    # mol_p and mol_l are rdkit mol objects
    # return a list of atoms in the pocket within cutoff distance to the ligand
    # get lig coords
    # return a list of atoms wich position, atom type and partial charge
    lig_coords = mol_l.GetConformer().GetPositions()
    print(lig_coords.shape)
    conf_p = mol_p.GetConformer().GetPositions()
    pockect = []
    for i in range(len(conf_p)):
        pos_i = conf_p[i]
        dist = np.linalg.norm(lig_coords-pos_i,axis=1)
        if np.min(dist) < cutoff:
            pockect.append([mol_p.GetAtomWithIdx(i).GetSymbol(),\
                            mol_p.GetAtomWithIdx(i).GetAtomicNum(),\
                            pos_i])
    return pockect


