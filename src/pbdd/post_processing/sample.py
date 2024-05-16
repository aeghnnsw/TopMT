import multiprocessing
import os
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch.nn import DataParallel
import torch_geometric
from pbdd.data_processing.utils import restore_origin_coors
from pbdd.post_processing.mol_search import MolSearch
from pbdd.post_processing.scoring import calc_qed, vina_score_with_convert
from pbdd.post_processing.utils import (combine_multi_mols,
                                        convert_dummpy_frag, convert_rdkit_mol,
                                        convert_rdkit_pdbqt_str,
                                        convert_sdf_pdbqt,
                                        convert_to_rdkit_mol_with_type,
                                        pickel_mol)
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from torch_geometric.utils import subgraph, to_networkx, to_undirected

RDLogger.DisableLog('rdApp.*')

# Suppress RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

import pickle
import warnings
from copy import deepcopy
from itertools import product

from rdkit.Chem import Draw
from tqdm import tqdm


def assign_type(mol_assign_model,edge_index,n_nodes,n_samples=100,batch_size=64,pos=None,\
                qed_threshold:Optional[float]=0.5):
    # print('start assign type')
    edge_index = to_undirected(edge_index)
    n_mol = 0
    mols = []
    smi_list = []
    sample_count = 0
    max_sample = (n_samples*50//batch_size)+1
    n_edges = edge_index.shape[1]
    # print('check 1')
    # print(n_samples)
    qed_penalty = max(qed_threshold,0.8)
    while n_mol<n_samples:
        # print('check 2')
        batch_list = [torch_geometric.data.Data(edge_index=edge_index,\
                                                x = torch.randn(n_nodes,1),\
                                                edge_attr = torch.randn(n_edges,1)) \
                                                for i in range(batch_size)]
        batch = torch_geometric.data.Batch.from_data_list(batch_list)
        z = batch.x
        edge_z = batch.edge_attr
        edge_index_batch = batch.edge_index
        # print('check 3')
        # print(mol_assign_model)
        batch_x_g,batch_edge_g = mol_assign_model(z,edge_index_batch,edge_z,batch.batch)
        batch_x_g = batch_x_g.detach()
        batch_edge_g = batch_edge_g.detach()
        # print('check 4')
        for i in range(batch_size):
            x_g = batch_x_g[i*n_nodes:(i+1)*n_nodes]
            edge_g = batch_edge_g[i*n_edges:(i+1)*n_edges]
            mol_temp = convert_to_rdkit_mol_with_type(edge_index,x_g,edge_g,pos=pos)
            if mol_temp is not None:
                if qed_threshold is not None:
                    qed_temp = calc_qed(mol_temp)
                    # if qed larger than a threshold, accept the mol
                    # if qed is lower, accept the mol with a probability
                    if qed_temp<qed_threshold:
                        p_acc = qed_temp/qed_penalty
                        if np.random.rand()>p_acc:
                            continue
                    # reject the mol if it has too many chiral centers
                    chiral_centers = Chem.FindMolChiralCenters(mol_temp,force=True, includeUnassigned=True)
                    n_chiral = len(chiral_centers)
                    p_acc = 1/(1+np.exp(n_chiral-2))
                    if np.random.rand()>p_acc:
                        continue
                smi_temp = Chem.MolToSmiles(mol_temp)
                if smi_temp not in smi_list:
                    smi_list.append(smi_temp)
                    mols.append(mol_temp)
                    n_mol += 1
                    # print('found mol: ',n_mol)
        sample_count += 1
        if sample_count>max_sample:
            break
        # print('sample one batch')
    # print('Done assign type')
    return mols

def assign_mols(mol_graphs,mol_assign_model,n_samples=100,batch_size=64,num_process=0,\
                qed_threshold:Optional[float]=0.5):
    if len(mol_graphs)==0:
        return None
    if num_process == 0:
        num_process = os.cpu_count()//4
    num_process = min(len(mol_graphs),num_process)
    task_list = []
    for graph_temp in mol_graphs:
        # if isinstance(graph_temp,list):
        #     # print('restrore origin coors')
        #     graph = graph_temp[0]
        #     pos_center = graph_temp[1]
        #     rot_angles = graph_temp[2]
        #     pos = graph.x
        #     pos = restore_origin_coors(pos,pos_center,rot_angles)
        # else:
        graph=graph_temp
        pos = graph.x
        pos_center = graph.pos_center
        rot_angles = graph.rot_angles
        if pos_center is None or rot_angles is None:
            warnings.warn('pos_center or rot_angles is None')
        else:
            pos = restore_origin_coors(pos,pos_center,rot_angles)
        edge_index = graph.edge_index
        n_nodes = graph.x.shape[0]
        task_list.append((mol_assign_model,edge_index,n_nodes,n_samples,batch_size,pos,\
                          qed_threshold))
    with multiprocessing.get_context('fork').Pool(num_process) as pool:
        results_pool = pool.starmap(assign_type,task_list)
    mols = []
    for mols_temp in results_pool:
        for mol_temp in mols_temp:
            mols.append(mol_temp)
    return mols


def sample_graphs(g_model,edge_index,pos,n_preds,min_atoms:int,\
                  threshold=0.4,device=torch.device('cpu'),**kwargs):
    # sample graphs from generator
    # (Optional) filter by discriminator
    # assert g_model device == device
    assert g_model.device == device,'model and specified device are not the same'
    n_node = pos.shape[0]
    # print('n_node',n_node)
    gs = []
    z_data_list = []
    for i in range(n_preds):
        z = torch.randn(n_node,1).type_as(pos)
        edge_z = torch.ones(edge_index.shape[1],1).type_as(pos)
        g_z = torch_geometric.data.Data(x=z,edge_index=edge_index,edge_attr=edge_z,pos=pos)
        z_data_list.append(g_z)
    z_batch = torch_geometric.data.Batch.from_data_list(z_data_list).to(device)
    x_g_batch,edge_attr_g_batch = g_model(z_batch.x,z_batch.edge_index,\
                                          z_batch.edge_attr,z_batch.batch,z_batch.pos)
    g_pred_batch = g_model.discriminator(x_g_batch,z_batch.edge_index,\
                                         edge_attr_g_batch,z_batch.batch,z_batch.pos)
    g_pred_batch = torch.sigmoid(g_pred_batch)
    n_edges = edge_index.shape[1]
    # print(g_pred_batch)
    for i in range(n_preds):
        if g_pred_batch[i]<threshold:
            continue
        edge_attr_g = edge_attr_g_batch[i*n_edges:(i+1)*n_edges].squeeze().to(torch.device('cpu'))
        # remove nodes with no edges connected
        edge_positive = edge_index[:,edge_attr_g>0.5]
        e0 = list(edge_positive[0].numpy().astype(int))
        e1 = list(edge_positive[1].numpy().astype(int))
        subset = list(set(e0+e1))
        # print('subset node number',len(subset))
        if len(subset)<min_atoms*1.1:
            continue
        subset.sort()
        edge_index_new,edge_attr_new = subgraph(subset,edge_index,edge_attr=edge_attr_g,\
                                                relabel_nodes=True)
        pos_new = pos[subset,:].to(torch.device('cpu'))
        # edge_index_new = edge_index_new.numpy().astype(int)
        edge_attr_new = edge_attr_new.detach()
        pos_new = pos_new.numpy().astype(float)
        g_new = torch_geometric.data.Data(edge_index=edge_index_new,\
                                          edge_attr=edge_attr_new,pos=pos_new)
        g_networkx = to_networkx(g_new,to_undirected=True)
        largest_cc = list(max(nx.connected_components(g_networkx), key=len))
        # print('largest_cc',len(largest_cc))
        if len(largest_cc)<min_atoms:
            continue
        # print(largest_cc)
        edge_index_new,edge_attr_new = subgraph(largest_cc,edge_index_new,edge_attr=edge_attr_new,\
                                                relabel_nodes=True)
        pos_new = pos_new[largest_cc,:]
        g_new = torch_geometric.data.Data(edge_index=edge_index_new,\
                                          edge_attr=edge_attr_new,pos=pos_new,**kwargs)
        gs.append(g_new)
    return gs

def sample_graphs_serial(g_model,edge_index,pos,n_preds,threshold=0.5,device=torch.device('cpu')):
    # sample graphs from generator
    # (Optional) filter by discriminator
    n_node = pos.shape[0]
    gs = []
    for i in range(n_preds):
        z = torch.randn(n_node,1).type_as(pos)
        edge_z = torch.ones(edge_index.shape[1],1).type_as(pos)
        batch = torch.zeros(n_node).long()
        x_g,edge_attr_g = g_model(z,edge_index,edge_z,batch,pos)
        g_pred = g_model.discriminator(x_g,edge_index,edge_attr_g,batch,pos)
        g_pred = torch.sigmoid(g_pred)
        print(g_pred)
        if g_pred<threshold:
            continue
        edge_attr_g = edge_attr_g.squeeze()
        # remove nodes with no edges connected
        edge_positive = edge_index[:,edge_attr_g>0.5]
        e0 = list(edge_positive[0].numpy().astype(int))
        e1 = list(edge_positive[1].numpy().astype(int))
        subset = list(set(e0+e1))
        subset.sort()
        edge_index_new,edge_attr_new = subgraph(subset,edge_index,edge_attr=edge_attr_g,\
                                                relabel_nodes=True)
        edge_attr_new = edge_attr_new.detach()
        pos_new = pos[subset,:]
        g_new = torch_geometric.data.Data(edge_index=edge_index_new,\
                                          edge_attr=edge_attr_new.detach(),pos=pos_new)
        gs.append(g_new)
    return gs

def sample_tops(gs,n_min:int,n_max:int,rot_min:int,rot_max:int,\
                search_repeats:int=5,steps:int=200,num_process:int=0):
    # gs are graphs sampled from model
    # search valid mol topologies from gs
    # restore coordinates after search
    mol_list = []
    results_list = []
    if len(gs)==0:
        return mol_list,results_list
    if num_process == 0:
        num_process = os.cpu_count()
    num_process = min(len(gs),num_process)
    # print('gs',len(gs))
    # print(f'num_process: {num_process}')
    task_list = [(g,n_min,n_max,rot_min,rot_max,search_repeats,steps) for g in gs]
    # print(task_list)
    with multiprocessing.get_context('fork').Pool(num_process) as pool:
        results_pool = pool.starmap(top_search,task_list)
    combine_search = MolSearch(gs[0].pos,gs[0].edge_index,gs[0].edge_attr,\
                               n_min=n_min,n_max=n_max)
    for mol_graphss,results in results_pool:
        for mol_g,result in zip(mol_graphss,results):
            if not combine_search.check_duplicate_results(result):
                combine_search.results.append(result)
                results_list.append(result)
                mol_list.append(mol_g)
    print('valid tops before filtering', len(mol_list))
    return mol_list,results_list



def top_search(g,n_min:int,n_max:int,rot_min:int,rot_max:int,\
               search_repeats:int,search_steps:int,max_results:int=20):
    # single process search top for g
    assert hasattr(g,'pos_center'), 'g does not have pos_center'
    assert hasattr(g,'rot_angles'), 'g does not have rot_angles'
    edge_index = g.edge_index
    edge_w = g.edge_attr
    e1_pred = edge_index[0][edge_w>0.5]
    e2_pred = edge_index[1][edge_w>0.5]
    pred_edge_pairs = list()

    for e1,e2 in zip(e1_pred,e2_pred):
        e1 = int(e1)
        e2 = int(e2)
        pred_edge_pairs.append((e1,e2))
    # print(edge_w.shape)
    edge_w = edge_w.unsqueeze(1)
    # print(edge_w.shape)
    # print(edge_index.shape)
    s1 = MolSearch(g.pos,edge_index,edge_w,n_min=n_min,n_max=n_max)
    results_list = []
    mol_list = []
    for i in range(search_repeats):
        results = s1.search(n_max=search_steps)
        # print('check results',results,len(results))
        if len(results)>0:
            # print('check process results1')
            for result in results:
                # print('check process results2')
                edge_pair_temp = result.pairs
                n_edges = len(edge_pair_temp)
                edge_highlight = np.zeros(n_edges)
                for i in range(n_edges):
                    if edge_pair_temp[i] in pred_edge_pairs:
                        edge_highlight[i] = 1
                new_pos,new_edge_pairs = s1.update_index(result)
                new_edge1 = [edge[0] for edge in new_edge_pairs]
                new_edge2 = [edge[1] for edge in new_edge_pairs]
                new_edge = [new_edge1,new_edge2]
                new_edge = torch.tensor(new_edge).long()
                mol_graph = torch_geometric.data.Data(x=new_pos,\
                                                      edge_index=new_edge,\
                                                      edge_attr=edge_highlight,\
                                                      pos_center=g.pos_center,\
                                                      rot_angles=g.rot_angles)
                rdkit_mol_temp,_= convert_rdkit_mol(new_edge)
                smi_temp = Chem.MolToSmiles(rdkit_mol_temp)
                rdkit_mol_temp = Chem.MolFromSmiles(smi_temp)
                # if rdkit_mol_temp is None:
                    # continue
                n_atoms = rdkit_mol_temp.GetNumAtoms()
                n_rot_temp = Descriptors.NumRotatableBonds(rdkit_mol_temp)
                if n_rot_temp>=rot_min and n_rot_temp<=rot_max:
                    print(rot_min,rot_max,n_rot_temp,n_atoms)
                if n_rot_temp>=rot_min and n_rot_temp<=rot_max:
                    mol_list.append(mol_graph)
                    results_list.append(result)
        if len(results_list)>=max_results:
            break
    return mol_list,results_list
