from constraint import Problem
from constraint import MaxSumConstraint
from constraint import (BacktrackingSolver,
                       RecursiveBacktrackingSolver,
                        MinConflictsSolver)
import numpy as np


class MolCSP():
    def __init__(self,pos,edge_pairs):
        '''
        pos:    N_atom * 3
        '''
        self.onehot_dict = {0:'C-4',1:'C-3',2:'C-2',3:'C-1',\
                            4:'N-3',5:'N-2',6:'N-1',\
                            7:'O-2',8:'O-1',\
                            9:'F-1',\
                            10:'S-4',11:'S-2',12:'S-1',\
                            13:'P-4',\
                            14:'Cl-1'}
        self.N_atom = len(pos)
        self.edge_pairs = edge_pairs
        # self.P = Problem(BacktrackingSolver())
        # self.P = Problem(MinConflictsSolver(10000))
        self.P = Problem(RecursiveBacktrackingSolver())
        self.var = list()
        D_atoms = list(self.onehot_dict.keys())
        D_edges = [1,2,3]
        #self.filter_edge()
        for edge in edge_pairs:
            m,n = edge
            self.P.addVariable(f'edge_{m}_{n}',D_edges)
        print(self.P._variables)
        self.add_variables()
    
    def get_connected_edges(self,i:int):
        edge_list = list()
        for edge in self.edge_pairs:
            if i in edge:
                m,n = edge
                edge_list.append(f'edge_{m}_{n}')
        return edge_list

    def add_variables(self):
        def valence_constraint(*args):
            if args[0]==0:
                # 'C-4' constraint
                if len(args)==5:
                    if args[1]==1 and args[2]==1 and args[3]==1 and args[4]==1:
                        return True
                return False
            if args[0]==1:
                # 'C-3' constraint
                if len(args)==4:
                    if args[1]+args[2]+args[3]<=4:
                        return True
                return False
            if args[0]==2:
                # 'C-2' constraint
                if len(args)==3:
                    if args[1]+args[2]<=4:
                        return True
                return False
            if args[0]==3:
                # 'C-1' constraint
                if len(args)==2:
                    return True
                return False
            if args[0]==4:
                # 'N-3' constraint
                if len(args)==4:
                    if args[1]+args[2]+args[3]<=3:
                        return True
                return False
            if args[0]==5:
                # 'N-2' constraint
                if len(args)==3:
                    if args[1]+args[2]<=3:
                        return True
                return False
            if args[0]==6:
                # 'N-1' constraint
                if len(args)==2:
                    return True
                return False
            if args[0]==7:
                # 'O-2' constraint
                if len(args)==3:
                    if args[1]==1 and args[2]==1:
                        return True
                return False
            if args[0]==8:
                # 'O-1' constraint
                if len(args)==2:
                    if args[1]<=2:
                        return True
                return False
            if args[0]==9:
                # 'F-1' constraint
                if len(args)==2 and args[1]==1:
                    return True
                return False
            if args[0]==10:
                # 'S-4' constraint
                if len(args)==5:
                    if sum(args[1:])==6 and max(args[1:])<3:
                        return True
                return False
            if args[0]==11:
                # 'S-2' constraint
                if len(args)==3:
                    if args[1]==1 and args[2]==1:
                        return True
                return False
            if args[0]==12:
                # 'S-1' constraint
                if len(args)==2 and args[1]==1:
                    return True
                return False
            if args[0]==13:
                # 'P-4' constraint
                if len(args)==5:
                    if sum(args[1:])==5 and max(args[1:])<3:
                        return True
                return False
            if args[0]==14:
                # 'Cl-1' constraint
                if len(args)==2 and args[1]==1:
                    return True
                return False
        for i in range(self.N_atom):
            connected_edges = self.get_connected_edges(i)
            atom_var = f'atom{i}'
            var_list = [f'atom{i}']
            if len(connected_edges)==1:
                self.P.addVariable(atom_var,[3,6,8,9,12,14])
            elif len(connected_edges)==2:
                self.P.addVariable(atom_var,[2,5,7,11])
            elif len(connected_edges)==3:
                self.P.addVariable(atom_var,[1,4])
                # for edge in connected_edges:
                    # self.P._variables[edge] = [1,2]
            elif len(connected_edges)==4:
                self.P.addVariable(atom_var,[0,10,13])
                # for edge in connected_edges:
                    # self.P._variables[edge] = [1]
            var_list.extend(connected_edges)
            self.P.addConstraint(valence_constraint,var_list)
            print(var_list)


        
    # def add_angle_constraint(self):
    #     def calc_vectors(i,e1,e2,pos):
    #         vec_list = list()
    #         var_list = list()
    #         for m,n in zip(e1,e2):
    #             p1 = pos[m,:]
    #             p2 = pos[n,:]
    #             if m==i:
    #                 vec_temp = p2-p1
    #             if n==i:
    #                 vec_temp = p1-p2
    #             vec_list.append(vec_temp)
    #             var_list.append(f'edge{m}_{n}')
    #         return vec_list,var_list
    #     def angle_constraint1(var1,var2):
    #         # when cos>0, or angle<90
    #         if var1>0 and var2>0:
    #             return var1==1 and var2==1
    #         else:
    #             return True
    #     def angle_constraint2(var1,var2):
    #         # when cos>0.7 or angle<45
    #         return (var1==0) or (var2==0)
    #     def angle_constraint3(var1,var2):
    #         # when cos<-0.87 or angle>150
    #         return not ((var1==1) and (var2==1))
    #     def angle_constraint4(*vars):
    #         N = 0
    #         for var in vars:
    #             if var>0:
    #                 N+=1
    #         return N<=2
    #     for i in range(self.N_atom):
    #         e1,e2 = self.get_connected_edges(i)
    #         vec_list,var_list = calc_vectors(i,e1,e2,self.pos)
    #         L = len(vec_list)
    #         sm_angel_var = set()
    #         for j in range(L):
    #             for k in range(j+1,L):
    #                 vec1 = np.array(vec_list[j])
    #                 vec2 = np.array(vec_list[k])
    #                 cos_temp = np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    #                 var_names = [var_list[j],var_list[k]]
    #                 if cos_temp>0.7:
    #                     self.P.addConstraint(angle_constraint2,var_names)
    #                 elif cos_temp>=0:
    #                     self.P.addConstraint(angle_constraint1,var_names)
    #                     sm_angel_var.add(var_list[j])
    #                     sm_angel_var.add(var_list[k])
    #                 elif cos_temp<-0.87:
    #                     self.P.addConstraint(angle_constraint3,var_names)
    #         if len(sm_angel_var)>2:
    #             sm_angel_var = list(sm_angel_var)
    #             self.P.addConstraint(angle_constraint4,sm_angel_var)
    #     return None
    
    def get_solutions(self):
        self.solution = self.P.getSolutions()
        return self.solution
        