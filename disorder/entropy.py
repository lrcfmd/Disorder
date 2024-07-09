import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core.composition import Composition
from disorder.disorder import Disorder
import networkx as nx


class Entropy:
    """
    data is a row of a dataframe with the output from disorder classification
    """
    def __init__(self, file, radius_file='data/all_radii.csv', cutoff=0.5,occ_tol=1.05,merge_tol=0.005,pymatgen_dist_matrix=False,dist_tol=1e-3):
        
        self.material=Disorder(file,radius_file=radius_file, cutoff=cutoff,occ_tol=occ_tol,merge_tol=merge_tol,\
                               pymatgen_dist_matrix=pymatgen_dist_matrix,dist_tol=dist_tol)
        self.data=self.material.classify()
        self.data['formula']=self.material.material.read_formula
        self.data['Z']=self.material.material.z
       
    def mixing_entropy(self):
        calculated_orbits=np.zeros(len(self.data['label']))
        mixing_entropy=0
        conf_entropy=0
        
        for i,orb in self.data['label'].items():
            # for orbits without positional disorder
            if(self.data['orbit_disorder'][i]=='O'):
                calculated_orbits[i]=1
            elif(self.data['orbit_disorder'][i]=='S'):
                mult=self.data['multiplicity'][i]
                _,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                for j in occ:
                    mixing_entropy+=-mult*j*np.log(j)
                    conf_entropy+=-mult*j*np.log(j)
                calculated_orbits[i]=1
            elif(self.data['orbit_disorder'][i]=='V'):
                mult=self.data['multiplicity'][i]
                _,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                sum_occ=np.sum(occ)
                occ.append(1-sum_occ)
                for j in occ:
                    mixing_entropy+=-mult*j*np.log(j)
                    conf_entropy+=-mult*j*np.log(j)
                calculated_orbits[i]=1
            elif(self.data['orbit_disorder'][i]=='SV'):
                mult=self.data['multiplicity'][i]
                _,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                sum_occ=np.sum(occ)
                occ.append(1-sum_occ)
                for j in occ:
                    mixing_entropy+=-mult*j*np.log(j)
                    conf_entropy+=-mult*j*np.log(j)
                calculated_orbits[i]=1
            # for COM orbits
            elif(self.data['orbit_disorder'][i]=='COM'):
                mixing_entropy=np.nan
                conf_entropy=np.nan
                
            # for orbits with positional disorder expressions for mixing and configurational entropy 
            # will be different    
            elif(self.data['orbit_disorder'][i]=='SP'):
                # as intersecting orbits appear >1 time in the list of orbits
                if(calculated_orbits[i]==0):
                    if(self.data['internal_intersection'][i]==True and len(self.data['intersect_orbit_connected'][i])==1):
                    # internal intersection, no external intersection
                        calculated_orbits[i]=1
                        _,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                        for j in occ:
                            mixing_entropy+=-self.data['multiplicity'][i]*j*np.log(j*self.data['internal_intersect_index'][i][0])
                            conf_entropy+=-self.data['multiplicity'][i]*j*np.log(j)
                            
                    elif(len(self.data['intersect_orbit_connected'][i])>1 and self.data['intersect_orbit_connected_occ'][i]<=1.0):
                    # external intersection, maybe internal intersection
                        mixing_occ={}
                        for orb1 in self.data['intersect_orbit_connected'][i]:
                            k=list(self.data['label'].keys())[list(self.data['label'].values()).index(orb1)]
                            if(self.data['internal_intersection'][k]==False):
                            # no internal intersection
                                if(calculated_orbits[k]==0):
                                    calculated_orbits[k]=1
                                    names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                    for j in occ:
                                        conf_entropy+=-self.data['intersect_orbit_connected_mult'][k]*j*np.log(j)
                                # we need mixing_occ because different intersecting orbits may have the same element
                                # and for calculation of mixing entropy we need to sum all occupancies for it on different orbits
                                    for ind,name in enumerate(names):
                                        if(name in mixing_occ.keys()):
                                            mixing_occ[name]+=occ[ind]
                                        else:
                                            mixing_occ[name]=occ[ind]
                            else:
                            # both external and internal intersection
                                if(calculated_orbits[k]==0):
                                    calculated_orbits[k]=1
                                    names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                    for j in occ:
                                        conf_entropy+=-self.data['multiplicity'][k]*j*np.log(j)
                                    for ind,name in enumerate(names):
                                        if(name in mixing_occ.keys()):
                                            mixing_occ[name]+=occ[ind]*self.data['multiplicity'][k]/self.data['intersect_orbit_connected_mult'][k]
                                        else:
                                            mixing_occ[name]=occ[ind]*self.data['multiplicity'][k]/self.data['intersect_orbit_connected_mult'][k]
                            
                        for name in mixing_occ.keys():
                            # note in this case multiplicities of intersecting orbits are the same
                            mixing_entropy+=-self.data['intersect_orbit_connected_mult'][k]*mixing_occ[name]*np.log(mixing_occ[name])

                    else:
                    # Combination of external and internal intersection produces structures, 
                    # for which entropy depends on particular configuration. If total occupancy of intersecting 
                    # sites > 1 sometimes it is possible, but it will be correlated disorder
                        conf_entropy=np.nan
                        mixing_entropy=np.nan
                        break

            elif(self.data['orbit_disorder'][i]=='VP'):
                if(calculated_orbits[i]==0):
                    if(self.data['internal_intersection'][i]==True and len(self.data['intersect_orbit_connected'][i])==1):
                    # internal intersection, no external intersection
                        calculated_orbits[i]=1
                        names,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                        for j in occ:
                            mixing_entropy+=-self.data['multiplicity'][i]*j*np.log(j*self.data['internal_intersect_index'][i][0])
                            conf_entropy+=-self.data['multiplicity'][i]*j*np.log(j)    
                        mixing_entropy+=-self.data['multiplicity'][i]/self.data['internal_intersect_index'][i][0]*\
                            (1-self.data['internal_intersect_index'][i][0]*j)*np.log(1-j*self.data['internal_intersect_index'][i][0])
                        conf_entropy+=-self.data['multiplicity'][i]/self.data['internal_intersect_index'][i][0]*\
                            (1-self.data['internal_intersect_index'][i][0]*j)*np.log(1-j*self.data['internal_intersect_index'][i][0])
                    elif(len(self.data['intersect_orbit_connected'][i])>1 and self.data['intersect_orbit_connected_occ'][i]<=1):
                        # external intersection
                        mixing_occ=0
                        for orb1 in self.data['intersect_orbit_connected'][i]:
                            k=list(self.data['label'].keys())[list(self.data['label'].values()).index(orb1)]
                            if(self.data['internal_intersection'][k]==False):
                                # no internal intersection
                                if(calculated_orbits[k]==0):
                                    calculated_orbits[k]=1
                                    names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                    for j in occ:
                                        conf_entropy+=-self.data['multiplicity'][k]*j*np.log(j)
                                        mixing_occ+=j
                            else:
                                # internal intersection
                                if(calculated_orbits[k]==0):
                                    calculated_orbits[k]=1
                                    names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                    for j in occ:
                                        conf_entropy+=-self.data['multiplicity'][k]*j*np.log(j)
                                        mixing_occ+=j*self.data['multiplicity'][k]/self.data['intersect_orbit_connected_mult'][k]
                                    
                        if(1-mixing_occ>0):
                            conf_entropy+=-self.data['multiplicity'][k]*(1-mixing_occ)*np.log(1-mixing_occ)
                            mixing_entropy+=-self.data['multiplicity'][k]*mixing_occ*np.log(mixing_occ)-self.data['multiplicity'][k]*(1-mixing_occ)*np.log(1-mixing_occ)

                    else:
                        # Combination of external and internal intersection produces structures, 
                        # for which entropy depends on particular configuration
                        mixing_entropy=np.nan
                        conf_entropy=np.nan
                        break

                       
            elif(self.data['orbit_disorder'][i]=='P'):
                if(self.data['internal_intersection'][i]==True and len(self.data['intersect_orbit_connected'][i])==1):
                    # only internal intersection
                    calculated_orbits[i]=1
                    name,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                    for j in occ:
                        conf_entropy+=-self.data['multiplicity'][i]*j*np.log(j)

                elif(len(self.data['intersect_orbit_connected'][i])>1 and self.data['intersect_orbit_connected_occ'][i]<=1.02):
                    # only external intersection
                    for orb1 in self.data['intersect_orbit_connected'][i]:
                        k=list(self.data['label'].keys())[list(self.data['label'].values()).index(orb1)]
                        if(self.data['internal_intersection'][k]==False):
                            if(calculated_orbits[k]==0):
                                calculated_orbits[k]=1
                                names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                for j in occ:
                                    conf_entropy+=-self.data['intersect_orbit_connected_occ'][i]*j*np.log(j)
                        else:
                            if(calculated_orbits[k]==0):
                                calculated_orbits[k]=1
                                names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                for j in occ:
                                    conf_entropy+=-self.data['multiplicity'][k]*j*np.log(j)
                                       
                else:
                    # Combination of external and internal intersection produces structures, 
                    # for which entropy depends on particular configuration
                    mixing_entropy=np.nan
                    conf_entropy=np.nan
                    break
        
        comp=Composition(self.data['formula'])
        natoms=np.sum(list(comp.as_dict().values()))
        z=self.data['Z']
        mixing_entropy=mixing_entropy/z/natoms

        return mixing_entropy
    
    
    def configurational_entropy(self):
        calculated_orbits=np.zeros(len(self.data['label']))
        mixing_entropy=0
        conf_entropy=0
        
        
        for i,orb in self.data['label'].items():
                    # for orbits without positional disorder
                    if(self.data['orbit_disorder'][i]=='O'):
                        calculated_orbits[i]=1
                    elif(self.data['orbit_disorder'][i]=='S'):
                        mult=self.data['multiplicity'][i]
                        _,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                        for j in occ:
                            mixing_entropy+=-mult*j*np.log(j)
                            conf_entropy+=-mult*j*np.log(j)
                        calculated_orbits[i]=1
                    elif(self.data['orbit_disorder'][i]=='V'):
                        mult=self.data['multiplicity'][i]
                        _,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                        sum_occ=np.sum(occ)
                        occ.append(1-sum_occ)
                        for j in occ:
                            mixing_entropy+=-mult*j*np.log(j)
                            conf_entropy+=-mult*j*np.log(j)
                        calculated_orbits[i]=1
                    elif(self.data['orbit_disorder'][i]=='SV'):
                        mult=self.data['multiplicity'][i]
                        _,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                        sum_occ=np.sum(occ)
                        occ.append(1-sum_occ)
                        for j in occ:
                            mixing_entropy+=-mult*j*np.log(j)
                            conf_entropy+=-mult*j*np.log(j)
                        calculated_orbits[i]=1
                    # for COM orbits
                    elif(self.data['orbit_disorder'][i]=='COM'):
                        mixing_entropy=np.nan
                        conf_entropy=np.nan
                        
                    # for orbits with positional disorder expressions for mixing and configurational entropy 
                    # will be different    
                    elif(self.data['orbit_disorder'][i]=='SP'):
                        # as intersecting orbits appear >1 time in the list of orbits
                        if(calculated_orbits[i]==0):
                            if(self.data['internal_intersection'][i]==True and len(self.data['intersect_orbit_connected'][i])==1):
                            # internal intersection, no external intersection
                                calculated_orbits[i]=1
                                _,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                                for j in occ:
                                    mixing_entropy+=-self.data['multiplicity'][i]*j*np.log(j*self.data['internal_intersect_index'][i][0])
                                    conf_entropy+=-self.data['multiplicity'][i]*j*np.log(j)
                                    
                            elif(len(self.data['intersect_orbit_connected'][i])>1 and self.data['intersect_orbit_connected_occ'][i]<=1.0):
                            # external intersection, maybe internal intersection
                                mixing_occ={}
                                for orb1 in self.data['intersect_orbit_connected'][i]:
                                    k=list(self.data['label'].keys())[list(self.data['label'].values()).index(orb1)]
                                    if(self.data['internal_intersection'][k]==False):
                                    # no internal intersection
                                        if(calculated_orbits[k]==0):
                                            calculated_orbits[k]=1
                                            names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                            for j in occ:
                                                conf_entropy+=-self.data['intersect_orbit_connected_mult'][k]*j*np.log(j)
                                        # we need mixing_occ because different intersecting orbits may have the same element
                                        # and for calculation of mixing entropy we need to sum all occupancies for it on different orbits
                                            for ind,name in enumerate(names):
                                                if(name in mixing_occ.keys()):
                                                    mixing_occ[name]+=occ[ind]
                                                else:
                                                    mixing_occ[name]=occ[ind]
                                    else:
                                    # both external and internal intersection
                                        if(calculated_orbits[k]==0):
                                            calculated_orbits[k]=1
                                            names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                            for j in occ:
                                                conf_entropy+=-self.data['multiplicity'][k]*j*np.log(j)
                                            for ind,name in enumerate(names):
                                                if(name in mixing_occ.keys()):
                                                    mixing_occ[name]+=occ[ind]*self.data['multiplicity'][k]/self.data['intersect_orbit_connected_mult'][k]
                                                else:
                                                    mixing_occ[name]=occ[ind]*self.data['multiplicity'][k]/self.data['intersect_orbit_connected_mult'][k]
                                    
                                for name in mixing_occ.keys():
                                    # note in this case multiplicities of intersecting orbits are the same
                                    mixing_entropy+=-self.data['intersect_orbit_connected_mult'][k]*mixing_occ[name]*np.log(mixing_occ[name])

                            else:
                            # Combination of external and internal intersection produces structures, 
                            # for which entropy depends on particular configuration. If total occupancy of intersecting 
                            # sites > 1 sometimes it is possible, but it will be correlated disorder
                                conf_entropy=np.nan
                                mixing_entropy=np.nan
                                break

                    elif(self.data['orbit_disorder'][i]=='VP'):
                        if(calculated_orbits[i]==0):
                            if(self.data['internal_intersection'][i]==True and len(self.data['intersect_orbit_connected'][i])==1):
                            # internal intersection, no external intersection
                                calculated_orbits[i]=1
                                names,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                                for j in occ:
                                    mixing_entropy+=-self.data['multiplicity'][i]*j*np.log(j*self.data['internal_intersect_index'][i][0])
                                    conf_entropy+=-self.data['multiplicity'][i]*j*np.log(j)    
                                mixing_entropy+=-self.data['multiplicity'][i]/self.data['internal_intersect_index'][i][0]*\
                                    (1-self.data['internal_intersect_index'][i][0]*j)*np.log(1-j*self.data['internal_intersect_index'][i][0])
                                conf_entropy+=-self.data['multiplicity'][i]/self.data['internal_intersect_index'][i][0]*\
                                    (1-self.data['internal_intersect_index'][i][0]*j)*np.log(1-j*self.data['internal_intersect_index'][i][0])
                            elif(len(self.data['intersect_orbit_connected'][i])>1 and self.data['intersect_orbit_connected_occ'][i]<=1):
                                # external intersection
                                mixing_occ=0
                                for orb1 in self.data['intersect_orbit_connected'][i]:
                                    k=list(self.data['label'].keys())[list(self.data['label'].values()).index(orb1)]
                                    if(self.data['internal_intersection'][k]==False):
                                        # no internal intersection
                                        if(calculated_orbits[k]==0):
                                            calculated_orbits[k]=1
                                            names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                            for j in occ:
                                                conf_entropy+=-self.data['multiplicity'][k]*j*np.log(j)
                                                mixing_occ+=j
                                    else:
                                        # internal intersection
                                        if(calculated_orbits[k]==0):
                                            calculated_orbits[k]=1
                                            names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                            for j in occ:
                                                conf_entropy+=-self.data['multiplicity'][k]*j*np.log(j)
                                                mixing_occ+=j*self.data['multiplicity'][k]/self.data['intersect_orbit_connected_mult'][k]
                                            
                                if(1-mixing_occ>0):
                                    conf_entropy+=-self.data['multiplicity'][k]*(1-mixing_occ)*np.log(1-mixing_occ)
                                    mixing_entropy+=-self.data['multiplicity'][k]*mixing_occ*np.log(mixing_occ)-self.data['multiplicity'][k]*(1-mixing_occ)*np.log(1-mixing_occ)

                            else:
                                # Combination of external and internal intersection produces structures, 
                                # for which entropy depends on particular configuration
                                mixing_entropy=np.nan
                                conf_entropy=np.nan
                                break

                            
                    elif(self.data['orbit_disorder'][i]=='P'):
                        if(self.data['internal_intersection'][i]==True and len(self.data['intersect_orbit_connected'][i])==1):
                            # only internal intersection
                            calculated_orbits[i]=1
                            name,occ=self.data['species'][i].keys(),list(self.data['species'][i].values())
                            for j in occ:
                                conf_entropy+=-self.data['multiplicity'][i]*j*np.log(j)

                        elif(len(self.data['intersect_orbit_connected'][i])>1 and self.data['intersect_orbit_connected_occ'][i]<=1.02):
                            # only external intersection
                            for orb1 in self.data['intersect_orbit_connected'][i]:
                                k=list(self.data['label'].keys())[list(self.data['label'].values()).index(orb1)]
                                if(self.data['internal_intersection'][k]==False):
                                    if(calculated_orbits[k]==0):
                                        calculated_orbits[k]=1
                                        names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                        for j in occ:
                                            conf_entropy+=-self.data['intersect_orbit_connected_occ'][i]*j*np.log(j)
                                else:
                                    if(calculated_orbits[k]==0):
                                        calculated_orbits[k]=1
                                        names,occ=self.data['species'][k].keys(),list(self.data['species'][k].values())
                                        for j in occ:
                                            conf_entropy+=-self.data['multiplicity'][k]*j*np.log(j)
                                            
                        else:
                            # Combination of external and internal intersection produces structures, 
                            # for which entropy depends on particular configuration
                            mixing_entropy=np.nan
                            conf_entropy=np.nan
                            break

        comp=Composition(self.data['formula'])
        natoms=np.sum(list(comp.as_dict().values()))
        z=self.data['Z']
        conf_entropy=conf_entropy/z/natoms

        return conf_entropy