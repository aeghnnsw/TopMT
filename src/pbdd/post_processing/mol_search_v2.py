import numpy as np
from heapq import *
import torch
import torch_geometric
import copy
from copy import deepcopy
import time
import networkx as nx
import rdkit

class State():
    def __init__(self,head,id1,id2,pairs):
        self.head = head
        self.id = id1
        self.parent_id = id2
        self.pairs = pairs
        self.successors = None
        self.frontier = None
        self.edge_priority = None
        self.freq_priority = None
        self.priority = None

    @classmethod
    def get_2nd_neighbor(cls,nx_graph,node):
        assert node in nx_graph.nodes,'node not in graph'
        first_neighbors = list(nx_graph.neighbors(node))
        second_neighbors = list()
        for node_temp in first_neighbors:
            second_neighbors+=list(nx_graph.neighbors(node_temp))
        neighbors = list(set(first_neighbors+second_neighbors))
        return neighbors


    def add_successors(self,successors):
        self.successors = successors
        
    def add_fontiers(self,frontiers):
        self.frontier = frontiers

    def show(self):
        # show info of the state
        print('id',self.id)
        print('pairs',self.pairs)
        print(f'number of paris: {len(self.pairs)}')
        print('priority',self.priority)
        print('priority_edge',self.priority_edge)
        print('priority_freq',self.priority_freq)
        print('successors',self.successors)

class MolSearch():
    def __init__(self,pos,edge_index,edge_w,n_min:int,n_max:int,top_freq_dict:dict=None):
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
        self.freq_dict = top_freq_dict
        # self.state_pairs = list()
    
    def check_duplicate_results(self,state:State):
        pairs1 = state.pairs
        for state_temp in self.results:
            pairs2 = state_temp.pairs
            if self.compare_pairs(pairs1,pairs2):
                return True
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
    
    def get_frontiers(self,state:State):
        assert state.successors is not None,'Get Successors first'
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

    def get_successors(self,state:State):
        head = state.head
        successors = list()
        if state.id!=0:
            parent_state = self.states[state.parent_id]
            successors = copy.deepcopy(parent_state.successors)
            # print('parent successors',successors)
            # print('pair',state.pairs[-1])
            successors.remove(state.pairs[-1])
        for i in range(self.N_edge):
            if head==self.edge_index[0][i]:
                id_temp = int(self.edge_index[1][i])
                if (id_temp,head) not in state.pairs and (head,id_temp) not in state.pairs:
                    pair_temp = (head,id_temp)
                    if pair_temp not in successors:
                        successors.append(pair_temp)
        state.add_successors(successors)  
        return successors 

    def priority_edge(self,state:State):
        if state.id ==0:
            return 0
        i,j = state.pairs[-1]
        cost = 10*(1-self.adj_w[i,j])
        # if w>0.5 cost<0, if w<0.5 cost>0
        cost = cost - 5
        random_term = 0.5*np.random.rand()
        return cost+random_term

    def priority_freq(self,state:State,cost_max=10,no_match_penalty=100):
        # construct networkx graph       
        g = nx.Graph()
        for pair in state.pairs:
            g.add_edge(pair[0],pair[1])
        n_nodes_g = len(g.nodes)
        if n_nodes_g<4:
            return 0
        node_i,node_j = state.pairs[-1]
        # get all second neighbors of node_i and node_j
        neighbors_i = State.get_2nd_neighbor(g,node_i)
        neighbors_j = State.get_2nd_neighbor(g,node_j)
        neighbors = list(set(neighbors_i+neighbors_j))
        # get the graph and freq of all neighbors
        freq_list = list()
        for neighbor in neighbors:
            neighbor_temp = State.get_2nd_neighbor(g,neighbor)
            subgrph_temp = g.subgraph(neighbor_temp)
            n_nodes = len(subgrph_temp.nodes)
            if n_nodes>3:
                match = False
                for top_temp in self.freq_dict[n_nodes]:
                    if nx.is_isomorphic(subgrph_temp,top_temp[0]):
                        freq_list.append(top_temp[1])
                        match = True
                        break
                if not match:
                    cost = no_match_penalty
        min_freq = min(freq_list)
        cost = cost_max*(1-min_freq)
        return cost


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
        return False
    
    def search(self,n_max=200):
        assert self.freq_dict is not None,'Please provide freq_dict'
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
        return self.results

    def step(self):
        if len(self.frontiers)==0:
            return False,None
        priority,current_state = heappop(self.frontiers)
        if priority>100:
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
            return True,current_state
        else:
            return True,None
    
    def update_priority(self,state:State):
        if state.id == 0:
            state.edge_priority = 0
            state.freq_priority = 0
            state.priority = 0
            return 0
        parent_id = state.parent_id
        parent_state = self.states[parent_id]
        edge_priority_new = self.priority_edge(state)
        state.edge_priority = parent_state.edge_priority+edge_priority_new
        state.freq_priority = self.priority_freq(state)
        state.priority = state.edge_priority+state.freq_priority
        return state.priority