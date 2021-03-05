
class fruit():

    def __init__(self,name):
        self.name = name
    def __new__(cls):
        print('111')
        
        
class FeatureAttribute():
    def __init__(self, atoms=None, atoms_indexs=None, feature=None, file_i=None)
        self.atoms = atoms
        self.atoms_indexs = atoms_indexs
        self.feature = feature
        self.file_i  = file_i
        self._initialized = False
        #self._running = False
        #self.results = {}
    def __initialize__(self):
        if not self._initialized:
            if self.feature in self.__dict__.keys():
                return self.self.feature
                
                
            else:
                raise TypeError('Unknown type of feature')
    
    def d_band_center(self):
        feature_i_list = []
        slab_name = file_i.split('#')[0].split('-')[1]
        ads_name  = file_i.split('#')[1].split('-')[0]
        slab_index = np.where(atoms_indexs > 18)[0]; ads_index = np.where(atoms_indexs < 18)[0]
        dos_pathway = '/home/wenbin/Calculation/graph_project/dos/'
        #slab feature
        with open(glob.glob(os.path.join(dos_pathway,f'{slab_name}*.pickle'))[0], 'rb') as input_file:
            dos_energies, dos_total, pdos = pickle.load(input_file)
            for atom_id in slab_index:
                assert atoms_indexs[atom_id] > 18
                states = 'd'
                #! whether Ni Fe Co in the system, the others should also consider as spin like Cu
                symbols = [atom.symbol for atom in atoms]
                if 'Fe' in symbols or 'Co' in symbols or 'Ni' in symbols:
                # if atoms[atom_id].symbol in ['Fe', 'Co', 'Ni']:
                    summed_pdos = pdos[atom_id][states][0] + pdos[atom_id][states][1]
                else:
                    summed_pdos = pdos[atom_id][states][0]
                
                band_center = simps(summed_pdos*dos_energies,dos_energies) / simps(summed_pdos,dos_energies)
                feature_i_list.append(band_center)
                
        with open(glob.glob(os.path.join(dos_pathway,f'{ads_name}-*.pickle'))[0], 'rb') as input_file:
            dos_energies, dos_total, pdos = pickle.load(input_file)
            #TODO whether atom index of molecule corresponds to surface after minus constant
            for atom_id in ads_index:
                
                assert atoms_indexs[atom_id] < 18
                if atoms[atom_id].symbol == 'C' or atoms[atom_id].symbol == 'O':
                    states = 'p'
                elif atoms[atom_id].symbol == 'H':
                    states = 's'
                #! change to the index in molecule
                atom_id = atom_id - len(slab_index)
                print(pdos[atom_id]['s'].max())
                print(atom_id)
                summed_pdos = pdos[atom_id][states][0] + pdos[atom_id][states][1]
                band_center = simps(summed_pdos*dos_energies,dos_energies) / simps(summed_pdos,dos_energies)
                feature_i_list.append(band_center)
        return feature_i_list        
        
        
        
    def __call__(self):
        return self.__initialize_()
    
    

def get_feature_attribute(atoms, atoms_indexs, feature, file_i):
    print(file_i)
    feature_i_list = []
    slab_name = file_i.split('#')[0].split('-')[1]
    ads_name  = file_i.split('#')[1].split('-')[0]
    
    slab_index = np.where(atoms_indexs > 18)[0]; ads_index = np.where(atoms_indexs < 18)[0]
    dos_pathway = '/home/wenbin/Calculation/graph_project/dos/'
    
    #slab feature
    with open(glob.glob(os.path.join(dos_pathway,f'{slab_name}*.pickle'))[0], 'rb') as input_file:
        dos_energies, dos_total, pdos = pickle.load(input_file)
        for atom_id in slab_index:
            assert atoms_indexs[atom_id] > 18
            states = 'd'
            #! whether Ni Fe Co in the system, the others should also consider as spin like Cu
            symbols = [atom.symbol for atom in atoms]
            if 'Fe' in symbols or 'Co' in symbols or 'Ni' in symbols:
            # if atoms[atom_id].symbol in ['Fe', 'Co', 'Ni']:
                summed_pdos = pdos[atom_id][states][0] + pdos[atom_id][states][1]
            else:
                summed_pdos = pdos[atom_id][states][0]
            
            band_center = simps(summed_pdos*dos_energies,dos_energies) / simps(summed_pdos,dos_energies)
            feature_i_list.append(band_center)
    
    with open(glob.glob(os.path.join(dos_pathway,f'{ads_name}-*.pickle'))[0], 'rb') as input_file:
        dos_energies, dos_total, pdos = pickle.load(input_file)
        #TODO whether atom index of molecule corresponds to surface after minus constant
        for atom_id in ads_index:
            
            assert atoms_indexs[atom_id] < 18
            if atoms[atom_id].symbol == 'C' or atoms[atom_id].symbol == 'O':
                states = 'p'
            elif atoms[atom_id].symbol == 'H':
                states = 's'
            #! change to the index in molecule
            atom_id = atom_id - len(slab_index)
            print(pdos[atom_id]['s'].max())
            print(atom_id)
            summed_pdos = pdos[atom_id][states][0] + pdos[atom_id][states][1]
            band_center = simps(summed_pdos*dos_energies,dos_energies) / simps(summed_pdos,dos_energies)
            feature_i_list.append(band_center)
    return feature_i_list    