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

        self.formula=self.material.material.read_formula
        self.z=self.material.material.z
        self.occ_tol=occ_tol
    
    def get_data(self):
        self.data=self.material.classify()
        return self.data

    def orbit_entropy_s_sv_v(self, multiplicity, occupancies):
        '''
        Function calculates entropy for one simple orbit S, SV, or V
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
        input: entropy_type is : 'mixing', 'configurational', or 'mc_configurational'
        output: float value of entropy
        '''
        
        if not hasattr(self, 'data'):
            self.data=self.material.classify()
        calculated_orbits = np.zeros(len(self.data['label']))
        mixing_entropy = 0
        conf_entropy = 0
        
        if(entropy_type!='mc_configurational'):
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
        elif(entropy_type=='mc_configurational'):
            p=self.material.positions   
            new_orb_list=[]
            for ind,orb in enumerate(self.data['intersect_orbit_connected'].values):
                new_orb=tuple(np.sort(orb))
                new_orb_list.append(new_orb)
            for orb in list(set(new_orb_list)):
                for sub_orb in orb:
                    internal_intersection=self.data.loc[self.data['label']==sub_orb]['internal_intersection'].values[0]
                    disorder_type=self.data.loc[self.data['label']==sub_orb]['orbit_disorder'].values[0]
                if(len(orb)>1 or internal_intersection==True):
                    x=orb
                    n_exp=100000
                    orbit=p.loc[p['atom_site_label'].isin(x)]
                    index=orbit.index.values
                    occupancies=orbit['atom_site_occupancy'].values
                    intersections=orbit['intersecting_sites'].values
                    non_reject=np.ones(n_exp)
                    num_conf=np.ones(n_exp)
                    total_occ=np.zeros(n_exp)
                    for exp in range(n_exp):
                        prob=np.random.rand(len(orbit))
                        atoms=np.array(prob<occupancies,dtype=int)
                        occup_sites=atoms*index
                        total_occ[exp]=np.sum(atoms)
                        # if(total_occ[exp]!=2):
                        #     num_conf[exp]=0
                        occupied_indices = np.where(occup_sites != 0)[0]
                        for j in occupied_indices:
                            for site in intersections[j]:
                                if(site in occup_sites):
                                    non_reject[exp]=0      
                    conf_entropy+=np.log(np.sum(non_reject*num_conf)/np.sum(num_conf))
                    for sub_orb in x:
                        total_occ=0
                        mult=p.loc[p['atom_site_label']==sub_orb]['atom_site_symmetry_multiplicity'].values[0]
                        orbit_content=p.loc[p['atom_site_label']==sub_orb]['atom_site_type_symbol'].values[0]
                        for elem,occ in orbit_content.items():
                            total_occ+=occ
                            conf_entropy+=-mult*(occ*np.log(occ))
                        if(total_occ<1):    
                            conf_entropy+=-mult*((1-total_occ)*np.log(1-total_occ))
                elif(disorder_type in {'V','S','SV'}):
                    total_occ=0
                    mult=p.loc[p['atom_site_label']==orb[0]]['atom_site_symmetry_multiplicity'].values[0]
                    orbit_content=p.loc[p['atom_site_label']==orb[0]]['atom_site_type_symbol'].values[0]
                    for elem,occ in orbit_content.items():
                        total_occ+=occ
                        conf_entropy+=-mult*(occ*np.log(occ))
                    if(total_occ<1):    
                        conf_entropy+=-mult*((1-total_occ)*np.log(1-total_occ))
            
        comp = Composition(self.formula)
        natoms = np.sum(list(comp.as_dict().values()))
        mixing_entropy = round(mixing_entropy/(self.z * natoms),3)
        conf_entropy = round(conf_entropy/(self.z * natoms),3)

        if(entropy_type == 'mixing'):
            return mixing_entropy
        elif(entropy_type == 'configurational'):
            return conf_entropy
        elif(entropy_type == 'mc_configurational'):
            return conf_entropy
