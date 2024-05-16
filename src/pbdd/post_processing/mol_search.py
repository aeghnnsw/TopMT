import copy
import time
from copy import deepcopy
from heapq import *

import networkx as nx
import numpy as np
import torch
import torch_geometric


def find_n_ring(pairs,i,j,n):
    paths = [[i]]
    for k in range(n-1):
        paths_new = list()
        for path in paths:
            neighbors = find_nearest_neighbor(pairs,path[-1])
            for neighbor in neighbors:
                if neighbor not in path:
                    paths_new.append(path+[neighbor])
        if len(paths_new)==0:
            return paths_new
        paths = paths_new
    rings = list()
    for path in paths:
        if path[-1]==j:
            rings.append(path)
    return rings
    
def find_nearest_neighbor(pairs,i):
    neighbors = list()
    if len(pairs)==0:
        return neighbors
    for pair in pairs:
        m,n = pair
        if m==i:
            neighbors.append(n)
        if n==i:
            neighbors.append(m)
    return neighbors

# def find_nth_neighbor(pairs,i,n):
#     neighbors = list()
#     visited = set()
#     dfs(start, [start])
#     return results

# def find_nearest_neighbor(pairs,i):
#     neighbors = list()
#     if len(pairs)==0:
#         return neighbors
#     for pair in pairs:
#         m,n = pair
#         if m==i:
#             neighbors.append(n)
#         if n==i:
#             neighbors.append(m)
#     return neighbors

# def find_nth_neighbor(pairs,i,n):
#     neighbors = list()
#     visited = set()
#     neighbors_prev = [i]
#     for j in range(n):
#         neighbors_temp = list()
#         for head_temp in neighbors_prev:
#             neighbors_temp += find_nearest_neighbor(pairs,head_temp)
#         neighbors_temp = set(neighbors_temp)
#         for node_temp in visited:
#             if node_temp in neighbors_temp:
#                 neighbors_temp.remove(node_temp)
#         neighbors_temp = list(neighbors_temp)
#         visited = visited.union(neighbors_temp)
#         neighbors.append(neighbors_temp)
#         neighbors_prev = neighbors_temp
#     return neighbors


def pair_in_pairs(pair,pairs):
    i,j = pair
    for pair_temp  in pairs:
        m,n = pair_temp
        if m==i and n==j:
            return True
        if m==j and n==i:
            return True
    return False

class State():
    def __init__(self,head,id1,id2,pairs):
        self.head = head
        self.id = id1
        self.parent_id = id2
        self.pairs = pairs
        self.successors = None
        self.frontier = None
        self.priority_edge = 0
        self.priority_ring = 0
        self.priority_3rings = 0
        self.priority_valence = 0
        self.priority_cross = 0
        self.priority = 0
    
    def add_successors(self,successors):
        self.successors = successors
        
    def add_fontiers(self,frontiers):
        self.frontier = frontiers

    def show(self):
        # show info of the state
        print('id',self.id)
        print('head',self.head)
        print('pairs',self.pairs)
        print(f'number of paris: {len(self.pairs)}')
        all_nodes = list()
        for pair in self.pairs:
            all_nodes+=pair
        print(f'number of nodes: {len(set(all_nodes))}')
        print('priority',self.priority)
        print('priority_edge',self.priority_edge)
        print('priority_valence',self.priority_valence)
        print('priority_ring',self.priority_ring)
        print('priority_cross',self.priority_cross)
        print('successors',self.successors)

        
        
class MolSearch():
    def __init__(self,pos,edge_index,edge_w,n_min:int,n_max:int):
        self.N_atom = len(pos)
        self.N_edge = len(edge_w)
        self.adj_w = torch_geometric.utils.to_dense_adj(edge_index=edge_index,edge_attr=edge_w).squeeze().detach().numpy()
        self.pos = pos
        self.edge_index = edge_index.numpy()
        self.edge_w = edge_w
        self.states = list()
        self.n_states = 0
        self.frontiers = list()
        self.results = list()
        self.n_min = n_min
        self.n_max = n_max
        # self.state_pairs = list()
        
    def accept_choice(self,p_acc):
        # p_acc is acceptance probability
        # don't check if p_acc is larger than 1
        p_rand = np.random.rand()
        if p_rand<p_acc:
            return True
        else:
            return False

    def compare_pairs(self,pairs1,pairs2):
        if len(pairs1)!=len(pairs2):
            return False
        for pair_temp1 in pairs1:
            i,j = pair_temp1
            pair_temp2 = (j,i)
            if pair_temp1 not in pairs2 and pair_temp2 not in pairs2:
                return False
        return True
                
    
    def check_duplicate_results(self,state:State):
        pairs1 = state.pairs
        for state_temp in self.results:
            pairs2 = state_temp.pairs
            if self.compare_pairs(pairs1,pairs2):
                return True
        return False
        
    def init_head(self):
        init = True
        while init:
            head_temp = np.random.randint(self.N_atom)
            for i in range(self.N_edge):
                node1 = self.edge_index[0][i]
                if node1 == head_temp and self.edge_w[i]>0.5:
                    init=False
                    break
        state0 = State(node1,0,0,[])
        self.states = list()
        self.states.append(state0)
        self.n_states=1
        self.frontiers = list()
        heapify(self.frontiers)
        self.results = list()
        return state0
    
    def get_successors(self,state:State):
        head = state.head
        successors = list()
        if state.id!=0:
            parent_state = self.states[state.parent_id]
            successors = copy.deepcopy(parent_state.successors)
            # print('parent successors',successors)
            # print('pair',state.pairs[-1])
        for i in range(self.N_edge):
            if head==self.edge_index[0][i]:
                id_temp = int(self.edge_index[1][i])
                if (id_temp,head) not in state.pairs and (head,id_temp) not in state.pairs:
                    pair_temp = (head,id_temp)
                    pair_temp_reversed = (id_temp,head)
                    if pair_temp not in successors and pair_temp_reversed not in successors:
                        successors.append(pair_temp)
        if state.id!=0:
            successors.remove(state.pairs[-1])
        state.add_successors(successors)  
        return successors 
    
    def get_frontiers(self,state:State):
        if state.successors is None:
            print('Get Successors first')
            return False
        frontiers = list()
        parent_pairs = state.pairs
        for successor in state.successors:
            head = successor[1]
            parent_id = state.id
            child_id = self.n_states
            pairs = parent_pairs+[successor]
            frontier_temp = State(head,child_id,parent_id,pairs)
            # if self.check_duplicate(frontier_temp):
            #     continue
            self.n_states+=1
            self.update_priority(frontier_temp)
            frontiers.append(frontier_temp)
            self.states.append(frontier_temp)
        return frontiers
    
    def update_priority(self,state:State):
        if state.id == 0:
            state.priority_edge = 0
            state.priority_valence = 0
            state.priority_ring = 0
            state.priority_3rings = 0
            state.priority_cross = 0
            state.priority = 0
            return 0
        parent_id = state.parent_id
        parent_state = self.states[parent_id]
        priority_edge_new = self.edge_priority(state)
        state.priority_edge = parent_state.priority_edge+priority_edge_new
        state.priority_valence = self.valence_priority(state)
        # state.priority_ring = parent_state.priority_ring+self.ring_priority(state)
        state.priority_ring = self.ring_priority(state)
        state.priority_3rings += parent_state.priority_3rings
        state.priority_cross = parent_state.priority_cross+self.cross_priority(state)
        state.priority = state.priority_edge+state.priority_valence+state.priority_cross+state.priority_ring+state.priority_3rings
        # print('check state')
        # print('id',state.id)
        # print('pairs',state.pairs)
        # print('successors',state.successors)
        # print('frontier',state.frontier)
        # print('all priority',state.priority)
        # print('priority_edge',state.priority_edge)
        # print('priority_valence',state.priority_valence)
        # print('priority_ring',state.priority_ring)
        # print('priority_cross',state.priority_cross)

        return state.priority
    
    def edge_priority(self,state:State):
        if state.id ==0:
            return 0
        i,j = state.pairs[-1]
        cost = - 22 * self.adj_w[i,j]
        cost = cost + 8 - 10 * np.random.rand()
        # random_term = np.random.rand()
        return cost
    
    def valence_priority(self,state:State):
        cost = 0
        i,j = state.pairs[-1]
        degree_i = 0
        degree_j = 0
        for pair in state.pairs:
            if pair[0]==i or pair[1]==i:
                degree_i+=1
            if pair[0]==j or pair[1]==j:
                degree_j+=1
        # add cost to degree 4, this will introduce chiral center
        if degree_i>3 or degree_j>3:
            if self.accept_choice(0.03):
                cost = 30 + 50 * np.random.rand()
            else:
                cost = 1000
        if degree_i>4 or degree_j>4:
            cost = 1000
        return cost
    
    # def rotate_priority(self,state:State):
    #     # penalize if has too many rotatable bonds
    #     # count number of nodes that has only two edges and not in rings
    #     cost = 0


    # def ring_priority(self,state:State):
    #     # penalize all 3,4,7 member rings
    #     # if no ring is formed, also penalize for long chain
    #     cost = 0
    #     i,j = state.pairs[-1]
    #     rings_3 = find_n_ring(state.pairs,i,j,3)
    #     if len(rings_3)>0:
    #         cost += (25+20*np.random.rand())
    #         if len(rings_3)>1:
    #             cost += 200
    #             return cost
    #     rings_4 = find_n_ring(state.pairs,i,j,4) 
    #     # 4-ring is not encouraged
    #     # Two 3-ring is not allowed       
    #     if len(rings_4)>0:
    #         cost += (25+30*np.random.rand())
    #         if len(rings_3)>0:
    #             cost+=200
    #             return cost
    #     rings_5 = find_n_ring(state.pairs,i,j,5)
    #     rings_6 = find_n_ring(state.pairs,i,j,6)
    #     rings_7 = find_n_ring(state.pairs,i,j,7)
    #     # promote 5-ring and 6-ring
    #     if len(rings_5)>0:
    #         # 3-ring and 4-ring are not allowed
    #         cost = cost - len(rings_5)*10
    #         for ring5_temp in rings_5:
    #             if len(rings_3)>0:
    #                 for ring3_temp in rings_3:
    #                     if len(set(ring5_temp).intersection(set(ring3_temp)))==3:
    #                         cost+=200
    #                         return cost
    #             if len(rings_4)>0:
    #                 for ring4_temp in rings_4:
    #                     if len(set(ring5_temp).intersection(set(ring4_temp)))==4:
    #                         cost+=200
    #                         return cost

    #     if len(rings_6)>0:
    #         cost = cost - len(rings_6)*12
    #         # two 4-ring is not allowed and 
    #         # 3-ring + 5-ring is not encouraged
    #         for ring6_temp in rings_6:
    #             if len(rings_5)>0:
    #                 for ring5_temp in rings_5:
    #                     if len(set(ring6_temp).intersection(set(ring5_temp)))==5:
    #                         cost+=200
    #                         return cost
    #             if len(rings_4)>0:
    #                 for ring4_temp in rings_4:
    #                     if len(set(ring6_temp).intersection(set(ring4_temp)))==4:
    #                         cost+=200
    #                         return cost
    #             if len(rings_3)>0:
    #                 for ring3_temp in rings_3:
    #                     if len(set(ring6_temp).intersection(set(ring3_temp)))==3:
    #                         cost+=100
    #                         return cost
    #     # 3-ring + 6-ring or 4-ring + 5-ring or 7-ring are all allowed
    #     if len(rings_7)>0:
    #         cost+=5
    #     rings_8 = find_n_ring(state.pairs,i,j,8)
    #     # Only Two 5-rings are allowed
    #     # eight ring is not encoraged
    #     if len(rings_8)>0:
    #         for ring8_temp in rings_8:
    #             has_5ring = False
    #             if len(rings_5)>0:
    #                 for ring5_temp in rings_5:
    #                     if len(set(ring8_temp).intersection(set(ring5_temp)))==5:
    #                         has_5ring=True
    #             if has_5ring==False:
    #                 cost+=100
    #                 return cost
    #     rings_9 = find_n_ring(state.pairs,i,j,9)
    #     # 5-ring + 6-ring and 3 5-rings are allowed 
    #     if len(rings_9)>0:
    #         for ring9_temp in rings_9:
    #             has_5ring = False
    #             has_6ring = False
    #             if len(rings_5)>0:
    #                 for ring5_temp in rings_5:
    #                     if len(set(ring9_temp).intersection(set(ring5_temp)))==5:
    #                         has_5ring=True
    #             if len(rings_6)>0:
    #                  for ring6_temp in rings_6:
    #                     if len(set(ring9_temp).intersection(set(ring6_temp)))==6:
    #                         has_6ring=True
    #             if has_5ring==False and has_6ring==False:
    #                 cost += 100
    #                 return cost

    #     rings_10 = find_n_ring(state.pairs,i,j,10)
    #     # 6-ring + 6-ring and 5-ring + 7-ring and two 5-rings and one 6-ring  are allowed
    #     if len(rings_10)>0:
    #         for ring10_temp in rings_10:
    #             has_5ring = False
    #             has_6ring = False
    #             has_7ring = False
    #             if len(rings_5)>0:
    #                 for ring5_temp in rings_5:
    #                     if len(set(ring10_temp).intersection(set(ring5_temp)))==5:
    #                         has_5ring=True
    #             if len(rings_6)>0:
    #                  for ring6_temp in rings_6:
    #                     if len(set(ring10_temp).intersection(set(ring6_temp)))==6:
    #                         has_6ring=True
    #             if len(rings_7)>0:
    #                  for ring7_temp in rings_7:
    #                     if len(set(ring10_temp).intersection(set(ring7_temp)))==7:
    #                         has_7ring=True
    #             if has_5ring==False and has_6ring==False and has_7ring==False:
    #                 cost += 200
    #                 return cost
    #     rings_11 = find_n_ring(state.pairs,i,j,11)
    #     # 6-ring + 7-ring is allowed and 5-ring + 5-ring + 5 ring and two 6-rings + one 5-ring is allowed
    #     if len(rings_11)>0:
    #         for ring11_temp in rings_11:
    #             has_5ring = False
    #             has_6ring = False
    #             has_7ring = False
    #             if len(rings_5)>0:
    #                 for ring5_temp in rings_5:
    #                     if len(set(ring11_temp).intersection(set(ring5_temp)))==5:
    #                         has_5ring=True
    #             if len(rings_6)>0:
    #                  for ring6_temp in rings_6:
    #                     if len(set(ring11_temp).intersection(set(ring6_temp)))==6:
    #                         has_6ring=True
    #             if len(rings_7)>0:
    #                  for ring7_temp in rings_7:
    #                     if len(set(ring11_temp).intersection(set(ring7_temp)))==7:
    #                         has_7ring=True
    #             if has_5ring==False and has_6ring==False and has_7ring==False:
    #                 cost += 200
    #                 return cost
    #     rings_12 = find_n_ring(state.pairs,i,j,12)
    #     # 5-ring + 5-ring + 6 ring  and 3 6-rings are allowed
    #     if len(rings_12)>0:
    #         for ring12_temp in rings_12:
    #             has_5ring = False
    #             has_6ring = False
    #             if len(rings_5)>0:
    #                 for ring5_temp in rings_5:
    #                     if len(set(ring12_temp).intersection(set(ring5_temp)))==5:
    #                         has_5ring=True
    #             if len(rings_6)>0:
    #                  for ring6_temp in rings_6:
    #                     if len(set(ring12_temp).intersection(set(ring6_temp)))==6:
    #                         has_6ring=True
    #             if has_5ring==False and has_6ring==False:
    #                 cost += 100
    #                 return cost
    #     rings_13 = find_n_ring(state.pairs,i,j,13)
    #     # 5-ring + 6-ring + 6 ring is allowed
    #     if len(rings_13)>0:
    #         for ring13_temp in rings_13:
    #             has_5ring = False
    #             has_6ring = False
    #             if len(rings_5)>0:
    #                 for ring5_temp in rings_5:
    #                     if len(set(ring13_temp).intersection(set(ring5_temp)))==5:
    #                         has_5ring=True
    #             if len(rings_6)>0:
    #                  for ring6_temp in rings_6:
    #                     if len(set(ring13_temp).intersection(set(ring6_temp)))==6:
    #                         has_6ring=True
    #             if has_5ring==False and has_6ring==False:
    #                 cost += 100
    #                 return cost
    #     rings_14 = find_n_ring(state.pairs,i,j,14)
    #     # 5-ring + 6-ring + 6 ring is allowed
    #     if len(rings_14)>0:
    #         for ring14_temp in rings_14:
    #             has_6ring = False
    #             if len(rings_6)>0:
    #                  for ring6_temp in rings_6:
    #                     if len(set(ring14_temp).intersection(set(ring6_temp)))==6:
    #                         has_6ring=True
    #             if has_6ring==False:
    #                 cost += 100
    #                 return cost
    #     rings = rings_4+rings_5+rings_6+rings_7
    #     n_rings = len(rings)
    #     if n_rings>=2:
    #         for m in range(n_rings):
    #             for n in range(m+1,n_rings):
    #                 inter = set(rings[m]).intersection(set(rings[n]))
    #                 if len(inter)>2:
    #                     cost+=100
    #     return cost


    def ring_priority(self,state:State):
        '''
        calculate the priority of the state based on the rings
        '''
        p_3_ring = 0.005
        p_4_ring = 0.01
        p_5_ring = 0.12
        p_6_ring = 0.8
        p_7_ring = 0.05
        p_8_ring = 0.02
        p_large_ring = 0.05
        p_extra_large_ring = 0.01
        # def check_intersect(loops1,loops2):
        #     '''
        #     loops1 are smaller loops
        #     check wether all loops2 has intersect with loops1
        #     only if all loops2 dose not have intersect with loops1, return False
        #     '''
        #     is_subset = []
        #     for loop2 in loops2:
        #         for loop1 in loops1:
        #             if set(loop1).issubset(loop2):
        #                 is_subset.append(True)
        #                 break
        #         is_subset.append(False)
        #     if is_subset.count(False)==len(loops2):
        #         return False
        #     else:
        #         return True
        cost = 0
        i,j = state.pairs[-1]
        rings_3 = find_n_ring(state.pairs,i,j,3)
        if len(rings_3)>0:
            if len(rings_3)>1:
                cost = 1000
                return cost
            if self.accept_choice(p_3_ring):
                cost += (20+20*np.random.rand())
                state.priority_3rings += 20*len(rings_3)
            else:
                cost = 1000
                return cost
        rings_4 = find_n_ring(state.pairs,i,j,4) 
        # 4-ring is not encouraged
        # Two 3-ring is not allowed       
        if len(rings_4)>0:
            if len(rings_4)>1:
                cost = 1000
                return cost
            if self.accept_choice(p_4_ring):
                cost += (45+75*np.random.rand())
            else:
                cost = 1000
                return cost
            if len(rings_3)>0:
                cost = 1000
                return cost
        rings_5 = find_n_ring(state.pairs,i,j,5)
        rings_6 = find_n_ring(state.pairs,i,j,6)
        rings_7 = find_n_ring(state.pairs,i,j,7)
        # promote 5-ring and 6-ring
        if len(rings_5)>0:
            # 3-ring and 4-ring are not allowed
            if not self.accept_choice(p_5_ring):
                cost = 1000
                return cost
            cost = cost + 12 + 45 * len(rings_5) * np.random.rand()
            for ring5_temp in rings_5:
                if len(rings_3)>0:
                    cost+=300
                    return cost
                if len(rings_4)>0:
                    for ring4_temp in rings_4:
                        if len(set(ring5_temp).intersection(set(ring4_temp)))==4:
                            cost+=200
                            return cost
        if len(rings_5)>1:
            for m in range(len(rings_5)):
                for n in range(m+1,len(rings_5)):
                    inter = set(rings_5[m]).intersection(set(rings_5[n]))
                    if len(inter)>=3:
                        cost = cost + 50 + 70 * np.random.rand()
        if len(rings_6)>0:
            cost = cost - 40 - 50 * np.random.rand()
            # two 4-ring is not allowed and 
            # 3-ring + 5-ring is not encouraged
            for ring6_temp in rings_6:
                if len(rings_5)>0:
                    for ring5_temp in rings_5:
                        if len(set(ring6_temp).intersection(set(ring5_temp)))==5:
                            cost+=200
                            return cost
                        if len(set(ring6_temp).intersection(set(ring5_temp)))==4:
                            cost = cost + 50 + 80 * np.random.rand()
                        if len(set(ring6_temp).intersection(set(ring5_temp)))==2:
                            cost = cost + 50 + 60 * np.random.rand()
                if len(rings_4)>0:
                    for ring4_temp in rings_4:
                        if len(set(ring6_temp).intersection(set(ring4_temp)))==4:
                            cost+=300
                            return cost
                        if len(set(ring6_temp).intersection(set(ring4_temp)))==2:
                            cost = cost + 100 + 50 * np.random.rand()
                if len(rings_3)>0:
                    for ring3_temp in rings_3:
                        if len(set(ring6_temp).intersection(set(ring3_temp)))==3:
                            cost+=500
                            return cost
                        if len(set(ring6_temp).intersection(set(ring3_temp)))==2:
                            cost+=500
                            return cost
            # bridged two 6 ring is not encouraged (intersect == 2)
            if len(rings_6)>1:
                for m in range(len(rings_6)):
                    for n in range(m+1,len(rings_6)):
                        inter = set(rings_6[m]).intersection(set(rings_6[n]))
                        if len(inter)>=3:
                            cost = cost + 80 + 80*np.random.rand()
        # 3-ring + 6-ring or 4-ring + 5-ring or 7-ring are all allowed
        if len(rings_7)>0:
            if not self.accept_choice(p_7_ring):
                cost = 1000
                return cost
            cost+= 45 + 45 * np.random.rand()
            for ring7_temp in rings_7:
                if len(rings_3)>0:
                    for ring3_temp in rings_3:
                        if len(set(ring7_temp).intersection(set(ring3_temp)))==3:
                            cost+=500
                            return cost
                if len(rings_6)>0:
                    for ring6_temp in rings_6:
                        if len(set(ring7_temp).intersection(set(ring6_temp)))==6:
                            cost+=500
                            return cost
                        if len(set(ring7_temp).intersection(set(ring6_temp)))==2:
                            cost = cost + 50 + 50*np.random.rand()
        rings_8 = find_n_ring(state.pairs,i,j,8)
        # Only Two 5-rings are allowed
        # eight ring is not encoraged
        if len(rings_8)>0:
            for ring8_temp in rings_8:
                has_5ring = False
                if len(rings_5)>0:
                    for ring5_temp in rings_5:
                        if len(set(ring8_temp).intersection(set(ring5_temp)))==5:
                            has_5ring=True
                        # if len(set(ring8_temp).intersection(set(ring5_temp)))==1:
                        #     cost = cost + 20 + 20 * np.random.rand()
                if has_5ring==False:
                    if self.accept_choice(p_8_ring):
                        cost = cost + 50 + 50 * np.random.rand()
                    else:
                        cost = 1000
                        return cost
                else:
                    cost= cost +  30 + 20 * np.random.rand()
                if len(rings_6)>0:
                    for ring6_temp in rings_6:
                        if len(set(ring8_temp).intersection(set(ring6_temp)))==6:
                            cost+=300
                            return cost
                        if len(set(ring8_temp).intersection(set(ring6_temp)))==5:
                            cost = cost + 80 + 80 * np.random.rand()
                        
        rings_9 = find_n_ring(state.pairs,i,j,9)
        # 5-ring + 6-ring and 3 5-rings are allowed 
        if len(rings_9)>0:
            for ring9_temp in rings_9:
                has_5ring = False
                has_6ring = False
                if len(rings_5)>0:
                    for ring5_temp in rings_5:
                        if len(set(ring9_temp).intersection(set(ring5_temp)))==5:
                            has_5ring=True
                        if len(set(ring9_temp).intersection(set(ring5_temp)))==2:
                            cost = cost + 80 + 30 * np.random.rand()
                if len(rings_6)>0:
                     for ring6_temp in rings_6:
                        if len(set(ring9_temp).intersection(set(ring6_temp)))==6:
                            has_6ring=True
                        if len(set(ring9_temp).intersection(set(ring6_temp)))==2:
                            cost = cost + 10 + 30 * np.random.rand()
                if has_5ring==False and has_6ring==False:
                    cost = 1000
                    return cost
                else:
                    cost+= 50 + 50 * np.random.rand()

        rings_10 = find_n_ring(state.pairs,i,j,10)
        # 6-ring + 6-ring and 5-ring + 7-ring and two 5-rings and one 6-ring  are allowed
        if len(rings_10)>0:
            for ring10_temp in rings_10:
                has_5ring = False
                has_6ring = False
                has_7ring = False
                if len(rings_5)>0:
                    for ring5_temp in rings_5:
                        if len(set(ring10_temp).intersection(set(ring5_temp)))==5:
                            has_5ring=True
                if len(rings_6)>0:
                     for ring6_temp in rings_6:
                        if len(set(ring10_temp).intersection(set(ring6_temp)))==6:
                            has_6ring=True
                            cost -= 20
                if len(rings_7)>0:
                     for ring7_temp in rings_7:
                        if len(set(ring10_temp).intersection(set(ring7_temp)))==7:
                            has_7ring=True
                if has_5ring==False and has_6ring==False and has_7ring==False:
                    cost = 1000
                    return cost
                else:
                    cost+=30 + 30 * np.random.rand()
        rings_11 = find_n_ring(state.pairs,i,j,11)
        # 6-ring + 7-ring is allowed and 5-ring + 5-ring + 5 ring and two 6-rings + one 5-ring is allowed
        if len(rings_11)>0:
            for ring11_temp in rings_11:
                has_5ring = False
                has_6ring = False
                has_7ring = False
                if len(rings_5)>0:
                    for ring5_temp in rings_5:
                        if len(set(ring11_temp).intersection(set(ring5_temp)))==5:
                            has_5ring=True
                if len(rings_6)>0:
                     for ring6_temp in rings_6:
                        if len(set(ring11_temp).intersection(set(ring6_temp)))==6:
                            has_6ring=True
                if len(rings_7)>0:
                     for ring7_temp in rings_7:
                        if len(set(ring11_temp).intersection(set(ring7_temp)))==7:
                            has_7ring=True
                if has_5ring==False and has_6ring==False and has_7ring==False:
                    cost = 1000
                    return cost
                else:
                    cost+= 30 + 30 * np.random.rand()
        rings_12 = find_n_ring(state.pairs,i,j,12)
        # 5-ring + 5-ring + 6 ring  and 3 6-rings are allowed
        if len(rings_12)>0:
            for ring12_temp in rings_12:
                has_5ring = False
                has_6ring = False
                if len(rings_5)>0:
                    for ring5_temp in rings_5:
                        if len(set(ring12_temp).intersection(set(ring5_temp)))==5:
                            has_5ring=True
                if len(rings_6)>0:
                     for ring6_temp in rings_6:
                        if len(set(ring12_temp).intersection(set(ring6_temp)))==6:
                            has_6ring=True
                if has_5ring==False and has_6ring==False:
                    cost = 1000
                    return cost
                else:
                    cost = cost + 25 + 25 * np.random.rand()
        rings_13 = find_n_ring(state.pairs,i,j,13)
        # 5-ring + 6-ring + 6-ring is allowed
        if len(rings_13)>0:
            for ring13_temp in rings_13:
                has_5ring = False
                has_6ring = False
                if len(rings_5)>0:
                    for ring5_temp in rings_5:
                        if len(set(ring13_temp).intersection(set(ring5_temp)))==5:
                            has_5ring=True
                if len(rings_6)>0:
                     for ring6_temp in rings_6:
                        if len(set(ring13_temp).intersection(set(ring6_temp)))==6:
                            has_6ring=True
                if has_5ring==False and has_6ring==False:
                    cost = 1000
                    return cost
                else:
                    cost+=15
        rings_14 = find_n_ring(state.pairs,i,j,14)
        # 5-ring + 6-ring + 6 ring is allowed
        if len(rings_14)>0:
            for ring14_temp in rings_14:
                has_6ring = False
                if len(rings_6)>0:
                     for ring6_temp in rings_6:
                        if len(set(ring14_temp).intersection(set(ring6_temp)))==6:
                            has_6ring=True
                if has_6ring==False:
                    # cost += 100
                    cost = 1000
                    return cost
                else:
                    cost+=10
        # large_rings = []
        for n in range(15,30):
            large_ring_temp = find_n_ring(state.pairs,i,j,n)
            if len(large_ring_temp)>0:
                cost = 1000
                return cost
                # large_rings+=large_ring_temp
        # penalize bridge rings
        rings = rings_4+rings_5+rings_6+rings_7
        n_rings = len(rings)
        if n_rings>=2:
            for m in range(n_rings):
                for n in range(m+1,n_rings):
                    inter = set(rings[m]).intersection(set(rings[n]))
                    if len(inter)>2:
                        cost+=100
        rings = rings+rings_3
        # penalize long chain
        if len(rings)==0:
            cost+=8
        else:
            cost-=10
        return cost



    def cross_priority(self,state:State):
        if state.id==0:
            return 0
        m1,m2 = state.pairs[-1]
        mid1 = (self.pos[m1]+self.pos[m2])/2
        for pair in state.pairs[:-1]:
            n1,n2 = pair
            mid2 = (self.pos[n1]+self.pos[n2])/2
            if sum(abs(mid1-mid2))<0.01:
                return 200
        return 0          
    
    def reach_goal(self,state:State):
        # goal test, check if the state is a goal state
        # reach goal if the number of atoms larger than n_min and smaller than n_max
        pairs = state.pairs
        atom_list = list()
        for pair in pairs:
            atom_list+=pair
        atom_set = set(atom_list)
        n_atoms = len(atom_set)
        if n_atoms>=self.n_min and n_atoms<=self.n_max:
            return True
        # rand = np.random.rand()
        # n_min = self.N_atom//2-5
        # if n_atoms>n_min:
        #    p = n_atoms/self.N_atom
        #    if rand>p:
        #        return True
        return False
       
    def step(self,show:bool=False):
        if len(self.frontiers)==0:
            return False,None
        priority,current_state = heappop(self.frontiers)
        if show:
            current_state.show()
        if priority>150:
            print('priority too high')
            # current_state.show()
            # p_state = self.states[current_state.parent_id]
            # while p_state.parent_id!=0:
            #     p_state.show()
            #     p_state = self.states[p_state.parent_id]
            # p_state.show()
            return False,None
        # print('check2')
        # print(current_state.id)
        # print(current_state.priority)
        # print(current_state.pairs)
        self.get_successors(current_state)

        # current_state.show()
        new_frontiers = self.get_frontiers(current_state)
        for frontier_temp in new_frontiers:
            heappush(self.frontiers,(frontier_temp.priority,frontier_temp))    
        if self.reach_goal(current_state):
            # print(f'reach goal, current priority: {current_state.priority} ring: {current_state.priority_ring}')
            return True,current_state
        else:
            return True,None
    
    def search(self,n_max=200):
        state0 = self.init_head()
        self.update_priority(state0)
        self.get_successors(state0)
        new_frontiers = self.get_frontiers(state0)
        # print(new_frontiers)
        for frontier_temp in new_frontiers:
            heappush(self.frontiers,(frontier_temp.priority,frontier_temp))
        run,state_temp = self.step()
        count = 0
        while run and count<n_max:
            # print('check##########')
            # for state_check in self.states:
            #     print('id',state_check.id)
            #     print('parent',state_check.parent_id)
            #     print('successors',state_check.successors)
            #     print('pairs',state_check.pairs)
            #     print('---------')
            # print(frontier_temp.priority)
            run,state_temp = self.step()
            if state_temp!=None:
                # print('find one result')
                if not self.check_duplicate_results(state_temp):
                    self.results.append(state_temp)
            count+=1
        # if len(self.results)!=0:
        #     for result_temp in self.results:
        #         result_temp.show()
        return self.results
    
    def update_index(self,state:State):
        index_temp = 0
        index_dict = dict()
        new_edge_pairs = list()
        new_pos = list()
        for pair_temp in state.pairs:
            i,j = pair_temp
            if i not in index_dict.keys():
                index_dict[i]=index_temp
                index_temp+=1
                new_pos.append(self.pos[i,:])
            if j not in index_dict.keys():
                index_dict[j]=index_temp
                index_temp+=1
                new_pos.append(self.pos[j,:])
            new_edge_pairs.append((index_dict[i],index_dict[j]))
        new_pos = np.array(new_pos)
        # print(index_dict)
        return new_pos,new_edge_pairs
        
                        
    def convert_SDF(self,f_sdf,pos,pairs):
        N_atoms = len(pos)
        N_bonds = len(pairs)
        ele = ['C' for i in range(self.N_atom)]
        f = open(f_sdf,'w')
        f.write('XXX\nConverted From Graph\n\n')
        f.write(f'{N_atoms:<3}{N_bonds:<3}  0     0  0  0  0  0999 V2000\n')
        for element,coor in zip(ele,pos):
            line = f'{coor[0]:10.4f}{coor[1]:10.4f}{coor[2]:10.4f} {element:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n'
            f.write(line)
        for edge in pairs:
            e1 = edge[0]+1
            e2 = edge[1]+1
            line = f'{e1:3}{e2:3}  1  0  0  0  0\n'
            f.write(line)
        f.close()
        return None

    # def remove_redudant(self,pairs):
    #     new_pairs = list()
    #     for pair1 in pairs:
    #         i,j = pair1
    #         pair2 = (j,i)
    #         if (pair1 not in new_pairs) and (pair2 not in new_pairs):
    #             new_pairs.append(pair1)
    #     return new_pairs
        

   
    # def angle_penalty(self,vecs):
    #     #random_term = 0.1*np.random.randn()
    #     random_term=0
    #     penalty = 0
    #     N = len(vecs)
    #     if N>4:
    #         return 100
    #     for i in range(N):
    #         for j in range(i+1,N):
    #             vec1 = vecs[i]
    #             vec2 = vecs[j]
    #             cos_temp = np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    #             if cos_temp>0:
    #                 penalty+=2
    #             if cos_temp>0.2:
    #                 penalty+=5
    #     return penalty+random_term 
        
    
