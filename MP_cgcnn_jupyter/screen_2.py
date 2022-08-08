import os
import sys
from pymatgen.ext.matproj import MPRester
from copy import deepcopy
from pymatgen.core.composition import Composition
import logging
from monty.serialization import loadfn,dumpfn
from pymatgen.io.cif import CifWriter

#filename = sys.argv[1]
#m = eval(sys.argv[2])
#f = eval(sys.argv[3])

qe = MPRester(sys.argv[1])

directory = sys.argv[2]
os.mkdir(directory)

criteria = {'nelements':1}
criteria['icsd_ids.0']={'$exists':True}
entries = qe.get_entries(criteria, inc_structure=True, \
                         property_data=['icsd_id', 'icsd_ids', 'e_above_hull', \
                                        'material_id', 'tag','formation_energy_per_atom'])
id_prop = {}
for entry in entries:
    name = entry.data["material_id"][3:]
    name_cif = directory + '/' + name +'.cif'
    print(name_cif)
    w = CifWriter(entry.structure)
    w.write_file(name_cif)
    id_prop[name] = entry.data['formation_energy_per_atom']
csv_name = directory + '/' +'id_prop.csv'
with open(csv_name, 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in id_prop.items()]