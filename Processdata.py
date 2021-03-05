from ase import neighborlist
import ase.io
import numpy as np
# from Utility import Spherical_hamonics_1To6
import igraph as ig
import pandas as pd
from Utility import CollectInputNames, Adsorption_energy
import os 
import glob
import pickle
from scipy.integrate import simps
from sklearn.preprocessing import scale

  
        

def CollectDatatoPandas(graphs,target,name=None):
    column_names = ['graphs','target','name']
    datas = pd.DataFrame(columns=column_names)
    datas['graphs'] = graphs
    datas['target'] = target
    datas['name']   = name
    return datas

def get_node_attribute(atom_index, feature_i, attributes_pool):
    
    if np.isnan(attributes_pool[feature_i][atom_index]) == True:
        #raise ValueError('Node_attribute is NAN type')
        node_attribute = np.nan
    else:
        node_attribute = attributes_pool[feature_i][atom_index]
        
    return node_attribute

class GraphBase():
    #ToDo weighted edge
    def __init__(self,
                atoms=None,
                cutoff_mult = 1,
                weighted = False,
                skin = 0.1
                ):
        
        self.cutoff_mult = cutoff_mult
        self.weighted    = weighted
        self.skin        = skin
        self.atoms       = atoms
        
    #This part should in initilize section
    @property
    def atoms(self):  #each graph only have a atoms object
        return self.atoms
    @atoms.setter
    def atoms(self, input_atoms):
        GraphBase.atoms = input_atoms
        
    def _node_representation(self): #adjacency matrix
        cutoff       = neighborlist.natural_cutoffs(self.atoms, mult=self.cutoff_mult)
        neighborList = neighborlist.NeighborList(cutoff, self_interaction=False, bothways=True, skin=self.skin)
        neighborList.update(self.atoms)
        node_matrix  = neighborList.get_connectivity_matrix(sparse=False).astype(float)
        return node_matrix
    
    def Create_graph(self):
        g = ig.Graph.Adjacency(self._node_representation().tolist(), mode='undirected')
        # assert (np.array(g.vs.degree()) < 13).all() == True  #the maximum bond should not exceeding 12
        assert (np.array(g.vs.degree()) > 0 ).all() == True  #there shoule be no isolated atom (degree == 0)
        return g
    
    def Node_attributes(self, feature_list, attributes_pool, file_i):
        #Todo local order parameter
        atoms_indexs = self.atoms.get_atomic_numbers()
        node_attributes = {}
        feature_attributes = {}
        for feature in feature_list:
            feature_attributes[feature] = []
            if feature in attributes_pool.keys():
                for i, atom_index in enumerate(atoms_indexs):
                    feature_attributes[feature].append(get_node_attribute(atom_index, feature, attributes_pool))
            else:
                #this will return a list of feature vector of all atoms in the structure
                # feature_attributes[feature] = get_feature_attribute(self.atoms, atoms_indexs, feature, file_i)
                feature_attributes[feature] = FeatureAttribute(self.atoms, atoms_indexs, feature, file_i).calculate()
            # else:
            #     raise NameError('UNKNOWN FEATURE')
        return np.array(list(map(list,zip(*feature_attributes.values()))))
        
        
        # for i, atom_index in enumerate(atoms_indexs):
        #     node_attributes[i] = []
        #     for feature in feature_list:
        #         if feature in attributes_pool.keys():
        #             node_attributes[i].append(get_node_attribute(atom_index, feature, attributes_pool))
        #         # elif feature == 'Local_parameter':
        #         #     node_attributes[i].extend(Spherical_hamonics_1To6(i, atoms, graph, file_i))
        #         else:
        #             raise NameError('UNKNOWN FEATURE')
        #         #print(node_attributes[i])
        
        # return np.array(list(node_attributes.values()))
        
class GraphGroup(GraphBase):
    
    def __init__(self, database_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.database_path = database_path
        self.file_list     = CollectInputNames(database_path)
        
    #node_attributes are indepent from graph
    def Collect_graph(self):
        graphs = []
        # file_list = CollectInputNames(self.database_path)
        for i, file_i in enumerate(self.file_list):
            self.atoms = ase.io.read(self.database_path + file_i)
            graph_i = self.Create_graph()
        #   if weighted == True:
        #      graph = Assign_Edge_weight(atoms,graph,database_path+file_i)
        #      graphs.append(graph)
            graphs.append(graph_i)
        return graphs
    
    def Collect_node_attributes(self, feature_list, attributes_pool, normalize=True):
        sum_node_attributes = []
        # file_list = CollectInputNames(database_path)
        for i, file_i in enumerate(self.file_list):
            self.atoms = ase.io.read(self.database_path + file_i)
            #atoms = DeleteFixAtoms(atoms) #wenbin
            attribute_graph_i = self.Node_attributes(feature_list=feature_list, attributes_pool=attributes_pool, file_i=file_i)
            sum_node_attributes.append(attribute_graph_i)
            
        if normalize:
            return self.__class__.NormalizeAttribute(sum_node_attributes)
        else:
            return np.array(sum_node_attributes) 
        
    def Collect_ads_energies(self):
        ads_energies = []
        for i, file_i in enumerate(self.file_list):
            ads_energy = Adsorption_energy(self.database_path, file_i)
            ads_energies.append(ads_energy)
        return ads_energies
    
    # #* build-in normalize method
    # @staticmethod
    # def NormalizeAttribute(node_features):
        
    #     node_features_data = scale(np.concatenate(node_features, axis=0), axis = 0)
    #     splits_idx = np.cumsum(list(map(lambda node_feature: len(node_feature), node_features)))
    #     node_features_split = np.vsplit(node_features_data,splits_idx)		
    #     node_features = node_features_split[:-1]
        
    #     return np.array(node_features)
    
    @staticmethod
    def NormalizeAttribute(node_attributes):
        conc_attributes = np.concatenate(tuple(i for i in node_attributes),axis=0)
        normed_conc_attributes = conc_attributes / np.nanmax(np.abs(conc_attributes),axis=0) # [-1:1] #conc_attributes / conc_attributes.max(axis=0) [0:1]
        indexlist = np.cumsum(list(map(lambda node_attribute: len(node_attribute), node_attributes)))
        #indexlist = np.cumsum(lengthlist)
        # normed_conc_attributes[np.isnan(normed_conc_attributes)] = 2 #sign an outlier value for nan 
        normed_node_attributes = np.array_split(normed_conc_attributes, indexlist[:-1])
        return np.array(normed_node_attributes)
    
class FeatureAttribute(GraphBase):
    def __init__(self, atoms=None, atoms_indexs=None, feature=None, file_i=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atoms = atoms
        self.atoms_indexs = atoms_indexs
        self.feature = feature
        self.file_i  = file_i
        self.feature_i_list = []
        self.dos_pathway = '/home/wenbin/Calculation/graph_project/dos/'

        self.slab_name = self.file_i.split('#')[0].split('-')[1]
        self.ads_name  = self.file_i.split('#')[1].split('-')[0]
        self.slab_index = np.where(self.atoms_indexs > 18)[0]
        self.ads_index = np.where(self.atoms_indexs < 18)[0]

        
    def calculate(self):
        # if self.feature in self.__dir__():   #check whether the feature is the function of class
        if self.feature == 'd_band_center':
            return self.d_band_center()
        if self.feature == 'surface_band_center':
            return self.surface_band_center()
        if self.feature == 'HOMO':
            return self.HOMO()[0]
        if self.feature == 'GAP':
            return self.HOMO()[1]
        if self.feature == 'label_sites':
            return self.label_sites()
        else:
            raise TypeError('Unknown type of feature')
    # def __initialize__(self):
    #     if not self._initialized:
    #         if self.feature in self.__dict__.keys():
    #             return self.self.feature
    
    def HOMO(self):
        #! the scf should only include one scf calculation
        readline=0
        nkpoints=0
        bulk=[]
        with open(glob.glob(os.path.join(self.dos_pathway,f'{self.ads_name}-*.scf'))[0], 'r') as input_file:
            lines = input_file.readlines()
            
        for line in lines:
            if readline == 0 and " k =" in line and " bands " in line:
                readline=1
                nkpoints+=1
            elif "the Fermi energy is" in line:
                EF=float(line.split()[4])
                readline=0
            elif "occupation numbers" in line:
                readline=0
            elif "SPIN" in line:
                1
            elif readline == 1 and len(line) > 5:
                if " k =" in line :
                    nkpoints+=1
                else:
                    for i in range(len(line.split())):
                        bulk.append(float(line.split()[i]))
        bulk=sorted(bulk)
        for x in bulk:
            if x > EF:
                lumo=float(x)
                break
        homo = bulk[bulk.index(lumo)-1]
        gap = lumo -  bulk[bulk.index(lumo)-1]
        
        #* global feature
        feature_list_homo = len(self.atoms_indexs)*[homo]
        feature_list_gap  = len(self.atoms_indexs)*[gap]
        return feature_list_homo, feature_list_gap
        
    def surface_band_center(self):
        # atoms.get_tags()
        # atoms.get_positions()[:,2]

        top_k_idx=self.atoms.get_positions()[:,2].argsort()[::-1] 
        top_index_slab=[] 
        for index in top_k_idx: 
            if self.atoms[index].number > 18: 
                top_index_slab.append(index) 
       
        if self.slab_name[-7:] == '211_334':
            surface_index = top_index_slab[:9]
        elif self.slab_name[-7:] == '111_334':
            surface_index = top_index_slab[:9]
        elif self.slab_name[-7:] == '211_323':
            surface_index = top_index_slab[:6]
            
        else:
            raise ValueError('Unknown type of surface')
        
        #! check the matching of sorted index and tag 
        #there is no tag for slab 211_323
        # if self.slab_name[-7:] == '211_334' or self.slab_name[-7:] == '111_334':
        #     for i in surface_index:
        #         if self.atoms[i].tag !=1:
        #             print(self.file_i)
                #assert self.atoms[i].tag == 1

        with open(glob.glob(os.path.join(self.dos_pathway,f'{self.slab_name}*.pickle'))[0], 'rb') as input_file:
            dos_energies, dos_total, pdos = pickle.load(input_file)
        
        d_band_center = 0        
        # print(surface_index)
        for atom_id in surface_index:

            states = 'd'
            #! whether Ni Fe Co in the system, the others should also consider as spin like Cu
            symbols = [atom.symbol for atom in self.atoms]
            if 'Fe' in symbols or 'Co' in symbols or 'Ni' in symbols:
            # if atoms[atom_id].symbol in ['Fe', 'Co', 'Ni']:
                summed_pdos = pdos[atom_id][states][0] + pdos[atom_id][states][1]
            else:
                summed_pdos = pdos[atom_id][states][0]

            dbc = simps(summed_pdos*dos_energies,dos_energies) / simps(summed_pdos,dos_energies)
            d_band_center += dbc/len(surface_index)
        self.feature_i_list = len(self.atoms_indexs) *[d_band_center]
        # print(surface_index)
        return self.feature_i_list
            
    
    def d_band_center(self):

        #slab feature
        with open(glob.glob(os.path.join(self.dos_pathway,f'{self.slab_name}*.pickle'))[0], 'rb') as input_file:
            dos_energies, dos_total, pdos = pickle.load(input_file)
        for atom_id in self.slab_index:
            assert self.atoms_indexs[atom_id] > 18
            states = 'd'
            #! whether Ni Fe Co in the system, the others should also consider as spin like Cu
            symbols = [atom.symbol for atom in self.atoms]
            if 'Fe' in symbols or 'Co' in symbols or 'Ni' in symbols:
            # if atoms[atom_id].symbol in ['Fe', 'Co', 'Ni']:
                summed_pdos = pdos[atom_id][states][0] + pdos[atom_id][states][1]
            else:
                summed_pdos = pdos[atom_id][states][0]
            
            band_center = simps(summed_pdos*dos_energies,dos_energies) / simps(summed_pdos,dos_energies)
            self.feature_i_list.append(band_center)
        
        #molecule feature setting to be 0
        for atom_id in self.ads_index:
            assert self.atoms_indexs[atom_id] < 18
            self.feature_i_list.append(0)

        #* this is for the calculation of band center of atom in the molecule
        # with open(glob.glob(os.path.join(dos_pathway,f'{ads_name}-*.pickle'))[0], 'rb') as input_file:
        #     dos_energies, dos_total, pdos = pickle.load(input_file)
        #     #TODO whether atom index of molecule corresponds to surface after minus constant
        #     for atom_id in ads_index:
                
        #         assert self.atoms_indexs[atom_id] < 18
        #         if self.atoms[atom_id].symbol == 'C' or self.atoms[atom_id].symbol == 'O':
        #             states = 'p'
        #         elif self.atoms[atom_id].symbol == 'H':
        #             states = 's'
        #         #! change to the index in molecule
        #         atom_id = atom_id - len(slab_index)
        #         print(pdos[atom_id]['s'].max())
        #         print(atom_id)
        #         summed_pdos = pdos[atom_id][states][0] + pdos[atom_id][states][1]
        #         band_center = simps(summed_pdos*dos_energies,dos_energies) / simps(summed_pdos,dos_energies)
        #         feature_i_list.append(band_center)
        return self.feature_i_list       
    
    def label_sites(self):
        graph = self.Create_graph()  #function from sub class
        ads_bond_index = [] 
        metal_bond_index = [] 
        for node_i in graph.vs.indices: 
            if self.atoms[node_i].number < 18:   
                for edge_i in graph.es[graph.incident(node_i)]: 
                    if  self.atoms[edge_i.source].number > 18 and self.atoms[edge_i.source].number != node_i: 
                        metal_bond_index.append(edge_i.source)  
                        ads_bond_index.append(node_i) 
                    if  self.atoms[edge_i.target].number > 18 and self.atoms[edge_i.source].number != node_i: 
                        metal_bond_index.append(edge_i.target) 
                        ads_bond_index.append(node_i)
        bonding_indexs = list(set(ads_bond_index)) + list(set(metal_bond_index))
        self.feature_i_list = len(self.atoms_indexs)*[0]
        for bonding_index in bonding_indexs:
            self.feature_i_list[bonding_index] = 1
        
        return self.feature_i_list
   
#Todo
def Cal_Edge_weight(atoms, atom_i, atom_j):
    if atom_i == atom_j:
        return 1
    else:
        bond_length = atoms.get_distance(atom_i,atom_j,mic=True)
        covalent_r = covalent_radii[atoms[atom_i].number] + covalent_radii[atoms[atom_j].number]
        return covalent_r/bond_length

def Define_ads_bond(atoms, graph):
    ads_bond_index = []
    for node_i in graph.vs.indices:
        if atoms[node_i].number < 18:  
           for edge_i in graph.es[graph.incident(node_i)]:
               if atoms[edge_i.source].number > 18 or  atoms[edge_i.target].number > 18 :
                   ads_bond_index.append(node_i)
    return list(set(ads_bond_index))

def Assign_Edge_weight(atoms, graph, file_name):
    tags=False
    edge_list = graph.get_edgelist()
    ads_bond_index = Define_ads_bond(atoms,graph)   #define bonding atoms
    for e_index, edge_i in enumerate(edge_list):
        #print(e_index, edge_i)
        weight_i_list = []
        for ads_atom_i in ads_bond_index:    #cal contribution of source and target of edge in reference with ads_atoms
            weight_i_list.append(Cal_Edge_weight(atoms,edge_i[0], ads_atom_i))
            weight_i_list.append(Cal_Edge_weight(atoms,edge_i[1], ads_atom_i))
        try:
            weight_i = sum(weight_i_list)/len(weight_i_list)
        except:
            tags = True
            weight_i = 0
        graph.es[e_index]['weight'] = weight_i
    if tags == True:
        print(atoms)
        print(file_name)
        os.system('mv {} {}'.format('/home/wenbin/Software/VScode/wwl_test/'+ file_name,'/home/wenbin/Software/VScode/wwl_test/tmp/'))
    return graph


    
def DeleteFixAtoms(atoms):
    fix_indexs = atoms.constraints[0].get_indices()
    del atoms[fix_indexs]
    return atoms
