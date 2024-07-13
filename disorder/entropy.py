import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from disorder.disorder import Disorder


class Entropy:
    """
    data is a row of a dataframe with the output from disorder classification
    """
    def __init__(self, file: str, radius_file: str = 'data/all_radii.csv', 
                 cutoff: float = 0.5, occ_tol: float = 1.05, merge_tol: float = 0.01, \
                 pymatgen_dist_matrix: bool = False, dist_tol: float = 0.01):
        
        self.material=Disorder(file,radius_file=radius_file, cutoff=cutoff,occ_tol=occ_tol,merge_tol=merge_tol,\
                               pymatgen_dist_matrix=pymatgen_dist_matrix,dist_tol=dist_tol)
        self.data=self.material.classify()
        self.formula=self.material.material.read_formula
        self.z=self.material.material.z
        self.occ_tol=occ_tol

    def orbit_entropy_s_sv_v(self, multiplicity, occupancies):
        '''
        Function calculates entropy for one simple orbit S, SV
        Input: multiplicity - multiplicity of the orbit
               occupancies - occupancies of species on the orbit
        '''
        entropy=0
        sum_occ = np.sum(occupancies)
        occupancies.append(1-sum_occ)
        for occ in occupancies:
            if occ > 0:
                entropy += -multiplicity * occ * np.log(occ)
        return entropy     
    
    def orbit_entropy_internal(self, multiplicity, occupancies, intersect_index):
        '''
        Function calculates entropy for one P, VP, SP, SVP orbits with no external intersection
        Input: multiplicity - multiplicity of the orbit
               occupancies - occupancies of species on the orbit
        '''
        mixing_entropy = 0
        conf_entropy = 0

        for occ in occupancies:
            if occ > 0:
                mixing_entropy += -multiplicity * occ * np.log(occ*intersect_index)
                conf_entropy+= -multiplicity * occ * np.log(occ)

        if(np.sum(occupancies)*intersect_index < 1):
            mixing_entropy += -multiplicity/intersect_index * (1-np.sum(occupancies)*intersect_index) * np.log(1-np.sum(occupancies)*intersect_index)
            conf_entropy += -multiplicity/intersect_index * (1-np.sum(occupancies)*intersect_index) * np.log(1-np.sum(occupancies)*intersect_index)

        return mixing_entropy, conf_entropy

    
    def calculate_entropy(self, entropy_type: str = 'configurational') -> float:
        '''
        Function calculates entropy for compound
        input: entropy_type is : 'mixing' or 'configurational'
        output: float value of entropy
        '''
        
        calculated_orbits = np.zeros(len(self.data['label']))
        mixing_entropy = 0
        conf_entropy = 0
        
        for i, orb in self.data['label'].items():
            disorder_type = self.data['orbit_disorder'][i]
            mult = self.data['multiplicity'][i]
            _, occ = self.data['species'][i].keys(), list(self.data['species'][i].values())
            
            if disorder_type == 'O':
                calculated_orbits[i] = 1
            elif disorder_type in {'S', 'V', 'SV'}:
                add_entropy = self.orbit_entropy_s_sv_v(mult, occ)
                mixing_entropy += add_entropy
                conf_entropy += add_entropy
                calculated_orbits[i] = 1
            elif disorder_type == 'COM':
                mixing_entropy = np.nan
                conf_entropy = np.nan
            elif disorder_type in {'SP', 'VP', 'P', 'SVP'}:
                if not calculated_orbits[i]:
                    if self.data['internal_intersection'][i] and len(self.data['intersect_orbit_connected'][i]) == 1:
                        add_entropy = self.orbit_entropy_internal(mult, occ, self.data['internal_intersect_index'][i][0])
                        mixing_entropy += add_entropy[0]
                        conf_entropy += add_entropy[1]
                        calculated_orbits[i] = 1
                    elif len(self.data['intersect_orbit_connected'][i]) > 1:
                        mixing_occ = {}
                        for orb1 in self.data['intersect_orbit_connected'][i]:
                            k = self.data.loc[self.data['label'] == orb1].index[0]
                            if not self.data['internal_intersection'][k]:
                                if not calculated_orbits[k]:
                                    spec_k, occ_k = self.data['species'][k].keys(), list(self.data['species'][k].values())
                                    for j in occ_k:
                                        conf_entropy += -self.data['intersect_orbit_connected_mult'][k] * j * np.log(j)
                                    for ind, name in enumerate(spec_k):
                                        mixing_occ[name] = mixing_occ.get(name, 0) + occ_k[ind]
                                    calculated_orbits[k] = 1
                            else:
                                if not calculated_orbits[k]:
                                    spec_k, occ_k = self.data['species'][k].keys(), list(self.data['species'][k].values())
                                    for j in occ_k:
                                        conf_entropy += -mult * j * np.log(j)
                                    for ind, name in enumerate(spec_k):
                                        mixing_occ[name] = mixing_occ.get(name, 0) + occ_k[ind] * self.data['internal_intersect_index'][k][0]
                                    calculated_orbits[k] = 1
                        if np.sum(list(mixing_occ.values()))<1:
                            mixing_occ['VAC']=1-np.sum(list(mixing_occ.values()))
                            conf_entropy += -self.data['intersect_orbit_connected_mult'][i] * mixing_occ['VAC'] * np.log(mixing_occ['VAC'])
                        
                        for name in mixing_occ.keys():
                            mixing_entropy += -self.data['intersect_orbit_connected_mult'][k] * mixing_occ[name] * np.log(mixing_occ[name])
                        if self.data['intersect_orbit_connected_occ'][i] > 1:
                            mixing_entropy = np.nan
                            conf_entropy = np.nan
                            break
                    else:
                        mixing_entropy = np.nan
                        conf_entropy = np.nan
                        break
            
        
        comp = Composition(self.formula)
        natoms = np.sum(list(comp.as_dict().values()))
        mixing_entropy /= (self.z * natoms)
        conf_entropy /= (self.z * natoms)
        
        if(entropy_type == 'mixing'):
            return mixing_entropy
        elif(entropy_type == 'configurational'):
            return conf_entropy
