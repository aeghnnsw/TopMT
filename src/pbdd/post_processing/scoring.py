import os
import sys

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, Descriptors, Descriptors3D, RDConfig
from tqdm import tqdm

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from typing import Optional

import networkx as nx
import sascorer
from meeko import PDBQTMolecule, RDKitMolCreate
from pbdd.post_processing.utils import collect_2nd_neighbor_tops,convert_rdkit_pdbqt_str
from vina import Vina




def calc_sas(mol):
    return sascorer.calculateScore(mol)

def calc_qed(mol):
    return QED.qed(mol)

def vina_score_with_convert(mol_temp,receptor_pdbqt,pos_center,save_name:Optional[str]=None,\
               save_threshold:float=-8,\
               box_length:float=25.0,write_pose:bool=True,max_step:int=0):
    # convert frag to pdbqt str
    pdbat_str = convert_rdkit_pdbqt_str(mol_temp,add_h=True)
    return vina_score(pdbat_str,receptor_pdbqt,pos_center,save_name=save_name,\
                      save_threshold=save_threshold,box_length=box_length,\
                      write_pose=write_pose,max_step=max_step)

def vina_score(ligand_str,receptor_pdbqt,pos_center,save_name:Optional[str]=None,\
               save_threshold:float=-8,\
               box_length:float=25.0,write_pose:bool=True,max_step:int=0,):
    try:
        v = Vina(verbosity=0)
        # v = Vina()
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_string(ligand_str)
        v.compute_vina_maps(pos_center,[box_length,box_length,box_length])
        scores = v.optimize(max_steps=max_step)
        if save_name is not None:
            if scores[0]<save_threshold and write_pose:
                save_path = save_name+'_'+str(scores[0])+'.pdbqt'
                v.write_pose(save_path)
    except Exception as e:
        print(e)
        scores = [10,10]
    return scores

def vina_docking(ligand_pdbqt,receptor_pdbqt,pos_center,save_name:Optional[str]=None,\
                 save_threshold:float=-8,\
                 box_length:float=30.0,write_pose:bool=True,\
                 exhaustiveness:int=16,write_sdf:bool=True):
    try:
        v = Vina(verbosity=0)
        # v = Vina()
        v.set_receptor(receptor_pdbqt)
        if os.path.exists(ligand_pdbqt):
            v.set_ligand_from_file(ligand_pdbqt)
        else:
            v.set_ligand_from_string(ligand_pdbqt)
        v.compute_vina_maps(pos_center,[box_length,box_length,box_length])
        v.dock(exhaustiveness=exhaustiveness, n_poses=1)    
        score = v.energies(1)[0][0]
        print(score)

        if score<save_threshold and write_pose:
            # print('write pose')
            if write_sdf:
                    save_path = save_name+'_'+str(score)+'.sdf'
                    pose_str = v.poses(1)
                    # print(pose_str)
                    pdbqt_mol = PDBQTMolecule(pose_str,is_dlg=False,skip_typing=True)
                    rdkitmol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0]
                    w = Chem.SDWriter(save_path)
                    w.write(rdkitmol)
                    w.close()
            else:
                save_path = save_name+'_'+str(score)+'.pdbqt'
                v.write_pose(save_path)
    except Exception as e:
        print(e)
        score = 10
    return score

def topology_freq_filter(top,top_freq_dict):
    # convert top to networkx graph
    # top is torch_geometric.data.Data
    # if topology in top_freq_dict, return True
    # else return False
    sub_tops = collect_2nd_neighbor_tops(top)
    for sub_top in sub_tops:
        n_nodes = sub_top.number_of_nodes()
        for top_temp in top_freq_dict[n_nodes]:
            if nx.is_isomorphic(sub_top,top_temp[0]):
                return True
    return False

def calc_properties(mol,smiles=False,calc_conformer=False):
    '''
    Calculate properties for a molecules sdf file
    calculated properties: 
        MW, logP, #HBA, #HBD, #rotatable bonds, SAS, QED, NPR1, NPR2
    return pandas dataframe
    '''
    if mol is None:
        prop_dict = {'MW':np.nan, 'logP':np.nan, '#HBA':np.nan, '#HBD':np.nan,\
                    '#rotatable bonds':np.nan, 'SAS':np.nan, 'QED':np.nan,\
                    'NPR1':np.nan, 'NPR2':np.nan,'SMILES':None}
    if calc_conformer:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol,randomSeed=1)
        mol = Chem.RemoveHs(mol)
    mw = Descriptors.MolWt(mol)
    logp = Chem.Crippen.MolLogP(mol)
    num_hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
    num_hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
    num_rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    sas = sascorer.calculateScore(mol)
    qed = QED.qed(mol)
    npr1 = Descriptors3D.NPR1(mol)
    npr2 = Descriptors3D.NPR2(mol)
    if smiles:
        smi = Chem.MolToSmiles(mol)
    else:
        smi = None
    prop_dict = {'MW':mw, 'logP':logp, '#HBA':num_hba, '#HBD':num_hbd,\
                '#rotatable bonds':num_rot_bonds, 'SAS':sas, 'QED':qed,\
                'NPR1':npr1, 'NPR2':npr2,'SMILES':smi}
    return prop_dict

def calc_intdiv(mol_list,sim_mat=None,rank:int=1):
    '''Calculate internal diversity of a list of molecules   '''
    if sim_mat is None:
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2,2048) for mol in mol_list]
        sim_mat = np.ones((len(mol_list),len(mol_list)))
        for i in tqdm(range(1,len(mol_list))):
            sim_temp = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
            sim_mat[i,:i] = sim_temp
            sim_mat[:i,i] = sim_temp
    # copy the sims to the other half
    # don't use diagonal
    sim_values = sim_mat[np.triu_indices(len(mol_list),k=1)]
    int_div = 1 - np.power(np.mean(np.power(sim_values,rank)),1/rank)
    return int_div

def calc_int_sim_triu(fps):
    '''
    Calculate internal similarity matrix of a list of molecules
    only store the upper triangle of the matrix
    '''
    sim_triu = []
    for i in tqdm(range(len(fps))):
        sim_temp = DataStructs.BulkTanimotoSimilarity(fps[i],fps[i:])
        sim_triu.extend(sim_temp)
    return sim_triu

def calc_ext_sim_matrix(fps1,fps2):
    """
    Calculate the extended similarity matrix between two sets of fingerprints.
    """
    sim_mat = np.ones((len(fps1),len(fps2)))
    for i in tqdm(range(len(fps1))):
        sim_temp = DataStructs.BulkTanimotoSimilarity(fps1[i],fps2)
        sim_mat[i,:] = sim_temp
        sim_mat
    return sim_mat