import multiprocessing
import os
import pickle
import random
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import time
import signal

import networkx as nx
import numpy as np
import rdkit
import torch
import torch_geometric
from dimorphite_dl import DimorphiteDL
from meeko import MoleculePreparation, PDBQTWriterLegacy
from pbdd.data_processing.utils import restore_origin_coors
from rdkit import Chem
from rdkit.Chem import (AllChem, rdDepictor, rdDetermineBonds, rdFMCS,
                        rdMolAlign)
from rdkit.Chem.MolStandardize import rdMolStandardize
# from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole
from rdkit.Geometry import Point3D
from tqdm import tqdm


def collect_2nd_neighbor_tops(mol):
    # convert mol to nx graph
    G = nx.Graph()
    if isinstance(mol,rdkit.Chem.rdchem.Mol):
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    elif isinstance(mol,torch_geometric.data.Data):
        edge_index = mol.edge_index
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0,i].item(),edge_index[1,i].item())
    # plt.figure()
    # nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
    tops = []
    for node in G.nodes():
        first_neighbors = list(G.neighbors(node))
        second_neighbors = []
        for neighbor in first_neighbors:
            second_neighbors += list(G.neighbors(neighbor))
        all_neighbors = list(set(first_neighbors + second_neighbors))
        sub_G = G.subgraph(all_neighbors)
        sub_G = nx.convert_node_labels_to_integers(sub_G)
        tops.append(sub_G)
    return tops

def combine_multi_mols(mols):
    # combine multiple mols into one mol
    # return a rdkit mol
    if len(mols)==1:
        return mols[0]
    mol = mols[0]    
    for m in mols[1:]:
        mol = Chem.CombineMols(mol,m)
    return mol

def convert_dummpy_frag(frag):
    # convert a fragment with dummy atoms to a complete molecule that can be docked
    # dummpy atom will be replaced by carbon atom
    # return a rdkit mol
    mol_new = deepcopy(frag)
    # change to RWMol
    mol_new = Chem.RWMol(mol_new)
    for atom in mol_new.GetAtoms():
        if atom.GetAtomicNum()==0:
            atom.SetAtomicNum(6)
    mol_new = mol_new.GetMol()
    # clean up the mol
    mol_new = rdMolStandardize.Cleanup(mol_new)
    return mol_new

def convert_sdf_pdbqt(sdf_file, pdbqt_file):
    # convert sdf file to pdbqt file
    assert os.path.exists(sdf_file), f'{sdf_file} does not exist'
    os.system(f'obabel -isdf {sdf_file} -opdbqt -O {pdbqt_file} -p 7.4')

def convert_rdkit_pdbqt_str(rdkit_mol,add_h:bool=False):
    # convert rdkit mol to pdbqt string using meeko
    if add_h:
        rdkit_mol = Chem.AddHs(rdkit_mol)
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(rdkit_mol)
    setup = mol_setups[0]
    pdbqt_str,is_ok,error_msg = PDBQTWriterLegacy().write_string(setup)
    if not is_ok:
        print(f'failed to convert rdkit mol to pdbqt string: {error_msg}')
        return None
    return pdbqt_str

def convert_pdbqt_pdb(pdbqt_file, pdb_file):
    # convert pdbqt file to pdb file
    assert os.path.exists(pdbqt_file), f'{pdbqt_file} does not exist'
    os.system(f'obabel -ipdbqt {pdbqt_file} -opdb -O {pdb_file}')
    print(f'pdbqt file {pdbqt_file} converted to pdb file {pdb_file}')


def convert_rdkit_mol(edge_index,highlight=None,pos=None):
    # Convert edge_index of pyg to a rdkit molecule
    # All atom types are C, all bonds are single bonds
    atom_C = Chem.Atom('C')
    mol = Chem.RWMol()
    N_atoms = int(max(max(edge_index[0]),max(edge_index[1]))+1)
    N_bonds = len(edge_index[0])
    for i in range(N_atoms):
        mol.AddAtom(atom_C)
    for idx1,idx2 in zip(*edge_index):
        idx1 = int(idx1)
        idx2 = int(idx2)
        mol.AddBond(idx1,idx2,Chem.rdchem.BondType.SINGLE)
    mol = mol.GetMol()
    hightlight_bonds = list()
    if highlight is not None:
        for i in range(N_bonds):
            if highlight[i]==1:
                hightlight_bonds.append(i)
    if pos is not None:
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            x,y,z = pos[i,:]
            conf.SetAtomPosition(i,Point3D(x,y,z))
    return mol,hightlight_bonds

def convert_to_rdkit_mol_with_type(edge_index,atoms,edges,pos=None):
    # Convert edge_index of pyg to a rdkit molecule
    # All atom types are C, all bonds are single bonds
    # atom type:    C:0, N:1, O:2, F:3, P:4, S:5, Cl:6, Br:7
    # edge_attr is the bond type, N_bond * 3
    # bond type:    single:0, double:1, triple:2
    # check_flag = False

    atom_type = torch.argmax(atoms,dim=1).numpy()
    edge_type = torch.argmax(edges,dim=1).numpy()
    atom_labels = ['C','N','O','F','P','S','Cl','Br']
    bond_labels = [Chem.rdchem.BondType.SINGLE,\
                   Chem.rdchem.BondType.DOUBLE,\
                   Chem.rdchem.BondType.TRIPLE,\
                   Chem.rdchem.BondType.UNSPECIFIED]

    mol = Chem.RWMol()
    N_atoms = int(max(max(edge_index[0]),max(edge_index[1]))+1)
    # N_bonds = len(edge_index[0])
    for i in range(N_atoms):
        atom = Chem.Atom(atom_labels[atom_type[i]])
        mol.AddAtom(atom)
    visited_edges = []
    for i,[idx1,idx2] in enumerate(zip(*edge_index)):
        if [idx1,idx2] in visited_edges or [idx2,idx1] in visited_edges:
            continue
        idx1 = int(idx1)
        idx2 = int(idx2)
        visited_edges.append([idx1,idx2])
        try:
            mol.AddBond(idx1,idx2,bond_labels[edge_type[i]])
        except:
            mol.AddBond(idx1,idx2,bond_labels[3])
    # change all bonds in 3-ring to single bond
    for bond in mol.GetBonds():
        if bond.IsInRingSize(3):
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        # Triple bonds not in ring, only allow for nitriles
        if bond.GetBondType()==Chem.rdchem.BondType.TRIPLE:
            if bond.IsInRing():
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                # print('triple bond set to single')
                # check_flag = True
        # bridgehead double bond is not permitted
        if bond.GetBondType()==Chem.rdchem.BondType.DOUBLE:
            id1 = bond.GetBeginAtomIdx()
            id2 = bond.GetEndAtomIdx()
            if mol.GetAtomWithIdx(id1).IsInRing() or mol.GetAtomWithIdx(id2).IsInRing():
                if not bond.IsInRing():
                    bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                    # print('bridgehead double bond set to single')
                    # check_flag = True
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
    for atom in mol.GetAtoms():
        # make sure valence are correct and no charge on atoms
        # change allenes C=C=C to C-C=C
        if atom.GetSymbol()=='C':
            id1 = atom.GetIdx()
            n_neighbors = len(atom.GetNeighbors())
            if n_neighbors==2:
                id11 = atom.GetNeighbors()[0].GetIdx()
                id12 = atom.GetNeighbors()[1].GetIdx()
                if mol.GetAtomWithIdx(id11).GetSymbol()=='C' and mol.GetAtomWithIdx(id12).GetSymbol()=='C':
                    if mol.GetBondBetweenAtoms(id1,id11).GetBondType()==Chem.rdchem.BondType.DOUBLE and \
                        mol.GetBondBetweenAtoms(id1,id12).GetBondType()==Chem.rdchem.BondType.DOUBLE:
                        # print('found allenes')
                        # check_flag = True
                        mol.GetBondBetweenAtoms(id1,id11).SetBondType(Chem.rdchem.BondType.SINGLE)
                        # mol.GetBondBetweenAtoms(id1,id12).SetBondType(Chem.rdchem.BondType.UNSPECIFIED)
                        atom.SetHybridization(rdkit.Chem.rdchem.HybridizationType.SP2)
                        atom.SetAtomMapNum(id1)
    
    mol = mol.GetMol()

    # if check_flag:
        # print(Chem.MolToSmiles(mol))
    # Sanitize mol
    try:
        Chem.SanitizeMol(mol)
        # mol = rdMolStandardize.Cleanup(mol)
        # print(mol)
        # mol = rdMolStandardize.Uncharger().uncharge(mol)

        # print(f'{check_flag}, bond order assignment failed')
        # smi_temp = Chem.MolToSmiles(mol)
        # if check_flag:
            # print(smi_temp)
        # Chem.MolFromSmiles(smi_temp)
    except Exception as e:
        # print(e)
        # print(f'{check_flag}, Sanitization failed')
        return None
    # print(Chem.MolToSmiles(mol))   
    
    if pos is not None:
        rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            x,y,z = pos[i,:]+np.random.normal(0,0.05,3)
            conf.SetAtomPosition(i,Point3D(x,y,z))
        mol_opt= deepcopy(mol)
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol_opt)
            ff = AllChem.MMFFGetMoleculeForceField(mol_opt, mp)
            atom_list = range(len(mol.GetAtoms()))
            for i in atom_list:
                ff.MMFFAddPositionConstraint(i, 0.7, 1e3)
        except:
            # print('failed to optimize')
            return None
        # ff.AddFixedPoint(i)
        result=ff.Minimize(maxIts=5000)
        # print(result)
        if result!=0:
            # print ('not converged')
            return None
        else:
            mol_opt = rdMolStandardize.Cleanup(mol_opt)
            return mol_opt
    else:
        return mol

def get_force_field(mol_temp):
    mp = AllChem.MMFFGetMoleculeProperties(mol_temp)
    if mp is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol_temp, mp)
    return ff

def mol_partial_minimize(mol_temp, minimize_atom_idx,maxIts=1000):
    try:
        ff = get_force_field(mol_temp)
    except:
        return None
    if ff is None:
        return None
    for atom_id in minimize_atom_idx:
        ff.MMFFAddPositionConstraint(atom_id, 0.3, 1e2)
    try:
        result = ff.Minimize(maxIts=maxIts)
    except:
        result = 1
    return result


def pickel_mol(mol):
    pickleProps = (
        Chem.PropertyPickleOptions.MolProps |  
        Chem.PropertyPickleOptions.AtomProps  
    )
    Chem.SetDefaultPickleProperties(pickleProps)
    mol_pkl = pickle.dumps(mol,protocol=pickle.HIGHEST_PROTOCOL)
    return mol_pkl

def protonate_mol(rdkit_mol,keep_coords:bool=True,max_variants:int=2):
    # assign protonated state to mol and retain the 3D coordinates
    smi_temp = Chem.MolToSmiles(rdkit_mol)
    protonator = DimorphiteDL(min_ph=7.4,max_ph=7.4,max_variants=max_variants)
    protonated_smis = protonator.protonate(smi_temp)
    # generate 3D coordinates for protonated mol
    if protonated_smis is None:
        return None
    # rdkit_mol = Chem.AddHs(rdkit_mol)
    protonated_mols = [rdkit_mol]
    rdkit_smi = Chem.MolToSmiles(rdkit_mol)
    # pickleProps = (
    #     Chem.PropertyPickleOptions.MolProps |  
    #     Chem.PropertyPickleOptions.AtomProps)
    # Chem.SetDefaultPickleProperties(pickleProps)
    for protonated_smi in protonated_smis:
        if protonated_smi==rdkit_smi:
            continue
        # find maximum common 
        mcs = rdFMCS.FindMCS((rdkit_mol,Chem.MolFromSmiles(protonated_smi)),completeRingsOnly=True,\
                                ringMatchesRingOnly=True,\
                                atomCompare=rdFMCS.AtomCompare.CompareAny, \
                                bondCompare=rdFMCS.BondCompare.CompareAny)
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        protonated_mol = Chem.MolFromSmiles(protonated_smi)
        protonated_mol=Chem.AddHs(protonated_mol)

        mcs_protonated_ids = protonated_mol.GetSubstructMatch(mcs_mol)        
        mcs_ref_ids = rdkit_mol.GetSubstructMatch(mcs_mol)
        try:
            AllChem.EmbedMolecule(protonated_mol)
            AllChem.MMFFOptimizeMolecule(protonated_mol)
        except:
            continue
        # set coordinates of mcs atoms to rdkit_mol and add position constraints to these atoms
        for i in range(len(mcs_ref_ids)):
            pos = rdkit_mol.GetConformer().GetAtomPosition(mcs_ref_ids[i])
            protonated_mol.GetConformer().SetAtomPosition(mcs_protonated_ids[i],pos)

        result = mol_partial_minimize(protonated_mol,mcs_protonated_ids,maxIts=1000)    
        if result!=0:
            # print('not converged')
            continue
        # protonated_mol = Chem.RemoveHs(protonated_mol)
        if keep_coords:
            # use rdMolAlign to align protonated mol to rdkit_mol
            rmsd = rdMolAlign.AlignMol(protonated_mol,rdkit_mol,atomMap=list(zip(mcs_protonated_ids,mcs_ref_ids)))
        # copy all properties from rdkit_mol to protonated_mol
        for prop in rdkit_mol.GetPropNames():
            protonated_mol.SetProp(prop,rdkit_mol.GetProp(prop))
        if rdkit_mol.HasProp('_Name'):
            protonated_mol.SetProp('_Name',rdkit_mol.GetProp('_Name'))
            # target_pkl = pickle.dumps(target_temp,protocol=pickle.HIGHEST_PROTOCOL)
        protonated_mols.append(protonated_mol)
    return protonated_mols

def hide_mol_number(mol):
    for atom in mol.GetAtoms():
        atom.ClearProp("molAtomMapNumber")


def to_directed(edge_index):
    # remove duplicate edges in undirected graph
    e1,e2 = edge_index
    pairs = []
    for i,j in zip(e1,e2):
        if [i,j] not in pairs and [j,i] not in pairs:
            pairs.append([i,j])
    return np.array(pairs).T

def read_multi_pdbqt(pdbqt_file):
    # read pdbqt file with multiple molecules
    # return a list of pdbqt strings
    mol_strs = []
    with open(pdbqt_file,'r') as f:
        lines = f.readlines()
        mol_temp = ''
        for line in lines:
            if line.startswith('MODEL'):
                continue
            if line.startswith('ENDMDL'):
                mol_strs.append(mol_temp)
                mol_temp = ''
                continue
            mol_temp += line
    return mol_strs

def read_pdbqt_mol(mol_template,pdbqt_file:str,temp_pdb_file:str,rm_pdb:bool=True):
    '''
    read pdbqt mol as rdkit mol
    convert pdbqt to pdb first, and then assign bond orders based on template mol
    '''
    convert_pdbqt_pdb(pdbqt_file,temp_pdb_file)
    mol = Chem.MolFromPDBFile(temp_pdb_file,removeHs=False)
    if rm_pdb:
        os.system(f'rm {temp_pdb_file}')
    if mol is None:
        print(f'failed to read {pdbqt_file}')
        return None
    mol = Chem.RemoveHs(mol)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(mol_template,mol)
    except:
        print(f'failed to assign bond orders for {pdbqt_file}')
        return None
    # check if number of atoms for mol and mol_template are the same
    # mol = Chem.RemoveHs(mol)
    # if mol.GetNumAtoms()!=mol_template.GetNumAtoms():
    #     print(mol.GetNumAtoms(),mol_template.GetNumAtoms())
    #     print(f'number of atoms for {pdbqt_file} is different from template')
    #     return None
    # # check the atom order (element type order)
    # for i in range(mol.GetNumAtoms()):
    #     if mol.GetAtomWithIdx(i).GetSymbol()!=mol_template.GetAtomWithIdx(i).GetSymbol():
    #         print(f'atom order for {pdbqt_file} is different from template')
    #         return None
    # # if the order is passed, assign mol coordinates to mol_template
    # conf = mol.GetConformer()
    # for i in range(mol.GetNumAtoms()):
    #     pos = conf.GetAtomPosition(i)
    #     mol_template.GetConformer().SetAtomPosition(i,pos)
    # calculate RMSD between mol and mol_template and add as property
    try:
        rmsd = Chem.rdMolAlign.CalcRMS(mol,mol_template)
        mol.SetProp('RMSD',str(rmsd))
    except:
        print(f'failed to calculate RMSD for {pdbqt_file}')
        return None
    return mol

def show_mol_number(mol,removeConformers=True):
    if removeConformers:
        mol.RemoveAllConformers()
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))


def topology_check_similairity(top_query,top_compare,min_diff:int=2):
    # check if top_query is similar to any top in tops
    # if len(top_list)==0:
    #     return True
    n_atom_query = top_query.GetNumAtoms()
    top_atom_num = top_compare.GetNumAtoms()
    if abs(n_atom_query-top_atom_num)<=2*min_diff:
        # check if top_query is similar to top_mol
        mcs = rdFMCS.FindMCS((top_query,top_compare),completeRingsOnly=False,\
                                ringMatchesRingOnly=True,\
                                atomCompare=rdFMCS.AtomCompare.CompareAny, \
                                bondCompare=rdFMCS.BondCompare.CompareAny)
        n_atom_match = mcs.numAtoms
        if n_atom_match>=n_atom_query-min_diff:
            return True
        else:
            return False

def topology_filter(tops,min_diff:int=2,pre_converted:bool=True,show_progress:bool=False):
    # remove all similar topologies, at least 2*min_diff atom difference
    # print('tops before filter:',len(tops))    
    # tops_filtered = []
    tops_mol_filtered = []
    if pre_converted:
        rdkit_tops = tops
    else:
        rdkit_tops = []
        for top in tops:
            rdkit_top, _ = convert_rdkit_mol(top.edge_index,pos=top.pos)
            rdkit_tops.append(rdkit_top)
    if show_progress:
        rdkit_tops = tqdm(rdkit_tops)
    for rdkit_top in rdkit_tops:
        if topology_check_similairity(rdkit_top,tops_mol_filtered,min_diff=min_diff):
            tops_mol_filtered.append(rdkit_top)
    print('number of tops after filter:',len(tops_mol_filtered))
    return tops_mol_filtered

def topology_filter_mp(tops,min_diff:int=2,n_process:int=16,batch_size:int=128,\
                       pre_converted:bool=True,shuffle:bool=False):
    # hierarchical filter for topologies, 
    #   divide tops into n_process batches, first filter within each batch
    #   Aggregate all filtered tops and filter again
    # convert to rdkit mol before multiprocessing
    if pre_converted:
        rdkit_tops = tops
    else:
        rdkit_tops = []
        for top in tqdm(tops):
            pos_center = top.pos_center
            rot_angles = top.rot_angles
            pos = top.x
            if pos_center is None or rot_angles is None:
                warnings.warn('pos_center or rot_angles is None')
            else:
                pos = restore_origin_coors(pos,pos_center,rot_angles)
            rdkit_top, _ = convert_rdkit_mol(top.edge_index,pos=pos)
            rdkit_tops.append(rdkit_top)
    if shuffle:
        random.shuffle(rdkit_tops)
    N_top = len(tops)
    task_list = []
    n_tasks = int(np.ceil(N_top/batch_size))
    print(f'number of tasks: {n_tasks}, batch size: {batch_size}')
    for i in range(n_tasks):
        if i<n_tasks-1:
            task_list.append([rdkit_tops[i*batch_size:(i+1)*batch_size],min_diff,True,False])
        else:
            task_list.append([rdkit_tops[i*batch_size:],min_diff,True,False])
    # for task in task_list:
        # print(f'number of tops in task: {len(task[0])}')
    n_process = min(n_process,len(task_list))
    with multiprocessing.get_context('forkserver').Pool(n_process) as pool:
        tops_filtered = pool.starmap(topology_filter,task_list)
    tops_filtered = [top for sublist in tops_filtered for top in sublist]
    # tops_filtered = topology_filter(tops_filtered,min_diff)
    return tops_filtered


def topology_check_dissimilarity_mp(current_top, existing_top,stop_event, min_diff:int=2):
    if stop_event.is_set():
        return True  # Early exit if a similarity has been found
    
    result = not topology_check_similairity(current_top, existing_top,min_diff=min_diff)
    if not result:
        stop_event.set()  # Set the stop event to signal other threads to stop
    return result

def topology_is_dissimilar(current_top, accepted_tops,max_workers:int=2,min_diff:int=2,wall_time:int=None):
    if len(accepted_tops) == 0:
        return True
    stop_event = threading.Event()
    if wall_time is None:
        wall_time = 100
    threads = []
    for exist in accepted_tops:
        thread = threading.Thread(target=topology_check_dissimilarity_mp, args=(current_top, exist, stop_event, min_diff))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(wall_time)
        if thread.is_alive():
            return False
    if stop_event.is_set():
        return False
    return True
        
        
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(topology_check_dissimilarity_mp, current_top, exist, stop_event, wall_time, min_diff): exist for exist in accepted_tops}
    #     for future in as_completed(futures):
    #         result_temp = future.result()
    #         if result_temp is False:
    #             stop_event.set()
    #             return False
    #         results.append(result_temp)
    # return all(results)
    # def watcher(stop_event, wall_time):
    #     time.sleep(wall_time)
    #     stop_event.set()
    # if len(accepted_tops) == 0:
    #     return True
    # if wall_time is None:
    #     wall_time = 100
    # stop_event = threading.Event()
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     watcher_thread = threading.Thread(target=watcher, args=(stop_event, wall_time))
    #     watcher_thread.start()
    #     futures = {executor.submit(topology_check_dissimilarity_mp, current_top, exist, stop_event, min_diff): exist for exist in accepted_tops}
        
    #     start_time = time.time()
    #     results = []

    #     try:
    #         # Iterate over futures as they complete
    #         for future in as_completed(futures):
    #             remaining_time = wall_time - (time.time() - start_time)
    #             if remaining_time <= 0:
    #                 print("Wall time exceeded before completion")
    #                 return False
    #             try:
    #                 result = future.result(timeout=remaining_time)  # Wait only the remaining time
    #                 results.append(result)
    #                 if not result:
    #                     stop_event.set()  # Signal to stop if a similar topology was found
    #                     return False  # Since a similar topology was found, return False immediately
    #             except TimeoutError:
    #                 stop_event.set()  # Signal to stop processing due to timeout
    #                 print("Operation timed out. Cancelling remaining tasks.")
    #                 break  # Exit the loop if we hit the timeout during waiting for result

    #     finally:
    #         # Cancel any still-running futures
    #         for future in futures:
    #             future.cancel()

    #     if stop_event.is_set():
    #         return False  # Return True if we were stopped by an event or timeout
        
    #     return all(results)

def topology_filter_mp_v2(tops,min_diff:int=2,n_process:int=128,pre_converted:bool=True):
    if pre_converted:
        rdkit_tops = tops
    else:
        rdkit_tops = []
        for top in tqdm(tops):
            pos_center = top.pos_center
            rot_angles = top.rot_angles
            pos = top.x
            if pos_center is None or rot_angles is None:
                warnings.warn('pos_center or rot_angles is None')
            else:
                pos = restore_origin_coors(pos,pos_center,rot_angles)
            rdkit_top, _ = convert_rdkit_mol(top.edge_index,pos=pos)
            rdkit_tops.append(rdkit_top)
    accected_tops = []
    for top in tqdm(rdkit_tops):
        if topology_is_dissimilar(top,accected_tops,max_workers=n_process,min_diff=min_diff):
            accected_tops.append(top)
        if len(accected_tops)%100==0:
            print(f'number of tops after filtering: {len(accected_tops)}')
    return accected_tops
    
def topology_filter_merge_batch(top_batch1,top_batch2,max_workers:int=128,min_diff:int=2):
    # merge two batches of topologies
    # remove similar topologies in top_batch2
    # top_batch1 is the accepted topologies
    # top_batch2 is the new topologies
    if len(top_batch1)==0:
        return top_batch2
    # loop the longer batch
    l_batch1 = len(top_batch1)
    l_batch2 = len(top_batch2)
    if l_batch1>l_batch2:
        batch1 = top_batch1
        batch2 = top_batch2
    else:
        batch1 = top_batch2
        batch2 = top_batch1
    top_batch1_filtered = []
    for top in tqdm(batch1):
        if topology_is_dissimilar(top,batch2,max_workers=max_workers,min_diff=min_diff):
            top_batch1_filtered.append(top)
    top_batch_filtered = batch2 + top_batch1_filtered
    return top_batch_filtered