#%%
import plotly.express as px
import pandas as pd
import plotly.io as pio
import numpy as np
import pkg_resources
import json
path = pkg_resources.resource_filename('catkit', 'data/properties.json')
with open(path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

feature_list = ['atomic_number', 
                'atomic_radius', 
                #'atomic_radius_rahm',
                'atomic_volume', 
                #'atomic_weight', 
                'boiling_point', 
                #'c6', 
                #'c6_gb', 
                #'covalent_radius_bragg', 
                #'covalent_radius_cordero', 
                #'covalent_radius_pyykko', 
                'covalent_radius_pyykko_double',
                #'covalent_radius_pyykko_triple',
                #'covalent_radius_slater', 
                'dband_center_bulk', 
                #'dband_center_slab', 
                'dband_kurtosis_bulk',
                #'dband_kurtosis_slab', 
                'dband_skewness_bulk', 
                #'dband_skewness_slab', 
                'dband_width_bulk', 
                #'dband_width_slab', 
                'density',
                'dipole_polarizability',
                'electron_affinity', 
                #'en_allen', 
                'en_ghosh',
                'en_pauling', 
                #'evaporation_heat',
                'fusion_heat', 
                #'gas_basicity', 
                'group_id', 
                'heat_of_formation',
                'lattice_constant', 
                #'melting_point', 
                #'metallic_radius', 
                #'metallic_radius_c12', 
                #'period',
                'proton_affinity', 
                'specific_heat', 
                'thermal_conductivity', 
                'vdw_radius', 
                #'vdw_radius_alvarez',
                #'vdw_radius_batsanov', 
                #'vdw_radius_bondi', 
                #'vdw_radius_dreiding', 
                #'vdw_radius_mm3', 
                'vdw_radius_rt', 
                #'vdw_radius_truhlar',
                'vdw_radius_uff']
df = df[feature_list]
df2 = df.corr()
# %%
