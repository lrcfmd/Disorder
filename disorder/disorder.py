import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core.composition import Composition
from disorder.cifreader import Read_CIF
import networkx as nx


class Disorder: 
    """ 
    Disorder class aims to assign disorder labels to each orbit in the structure spacified by the input CIF
    (1) cutoff specify how the threshold for sites intersection is determined
    (2) occ_tol determines the threshold of toleration of occupancy > 1. If occ_tol >= occupancy > 1, 
    then occupancy is rescaled to 1
    (3) merge_tol determines the threshold for the distance between the sites to be assumed 
    having the same location. In this case, they are merged, which generally leads to sites occupied 
    by more than one element
    (4) pymatgen_dist_matrix specifies which distance matrix to use. If True, than Structure.distance_matrix
    from pymatgen is used. Otherwise, the hand-written one is used.
    """
    def __init__(self, file, radius_file='data/all_radii.csv', cutoff=0.5,occ_tol=1.05,merge_tol=0.005,pymatgen_dist_matrix=False,dist_tol=1e-3):
        
        self.radius=pd.read_csv(radius_file,dtype={
            'symbol': str,
            'charge': int,
            'CN': str,
            'Spin State': str,
            'crystal radius': float,
            'ionic radius': float,
            'Key*': str,
            'species': str,
            'covalent': float,
            'empirical': float,
            'metallic': float
        })
        self.cutoff=cutoff
        self.occ_tol=occ_tol
        self.pymatgen_dist_matrix=pymatgen_dist_matrix
        self.errors=[]
        self.list_el=list(set(self.radius['symbol'].values))
    
        self.material=Read_CIF(file=file)
        
        try:
            o=self.material.orbits()
            s=self.material.symmetry()
            
            self.positions,self.struct = self.material.positions(o,s,pystruct=True,
                                                             merge_sites=True,merge_tol=merge_tol,dist_tol=dist_tol)
            
            self.errors=self.material.return_errors()
        except: 
            self.errors.append('IStructure_from_CIF not reading CIF')
    
                
    def return_errors(self):
        return self.errors
    
    def vacancy_number(self, orbit_label):
        db=self.positions.loc[self.positions['atom_site_label']==orbit_label]
        vac_number=sum(db['atom_site_occupancy'].values)
        return vac_number
    
    def create_cell(self):
        """This method creates a cell as input format in spglib"""
        lattice=self.struct.lattice.as_dict()['matrix']
        positions=[]
        elements=[]
        for s in self.struct.sites:
            positions.append(list(s.frac_coords))
            if(len(s.as_dict()['species'])==1):
                elements.append(s.as_dict()['species'][0]['element'])
            else:
                el=[]
                for x in s.as_dict()['species']:
                    el.append(x['element'])
                el.sort()
                elements.append(el)
        
        num_el=[]
        ext={}
        counter=100
        for el in elements:
            if(type(el)==str and el!='D'):
                num_el.append(self.dict_el[el])
            elif(type(el)==str and el=='D'):
                num_el.append(99)
                self.error.append('D in the structure')
            else:
                new_el=el[0]
                for e in el[1:]:
                    new_el=new_el+e
                if(new_el in ext.keys()):
                    num_el.append(ext[new_el])
                else:
                    num_el.append(counter)
                    ext[new_el]=counter
                    counter+=1
        self.dict_el.update(ext)
        return (lattice,positions,num_el)
  
    def determine_intersections(self):
        """ This function determines the intersections between sites.
            Returns a list of list of sites which intersect with a given one"""
        
        distance_template=np.zeros((len(self.positions),len(self.positions)))
        site_element_radiuses=self.element_radiuses()

        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                if(i!=j):
                    distance_template[i,j]=max([1,(site_element_radiuses[i]+site_element_radiuses[j])*self.cutoff])

        if(self.pymatgen_dist_matrix==True):
            dist=self.struct.distance_matrix
        else:
            dist=self.distance_matrix()
            
        a=dist-distance_template
        close=[]
        for i in range(len(self.positions)):
            cl=[]
            for j in range(len(self.positions)):
                if(a[i,j]<0):
                    cl.append(j)
            # we assume that for 'H1+' ad 'D1+' positional disorder is not possible        
            # for sp in Composition(self.positions.iloc[i]['atom_site_type_symbol']).elements:
            #     if(str(sp.element)=='H' or str(sp.element)=='D'):
            #         if(sp.oxi_state==1):
            #             cl=[]
            close.append(np.array(cl,dtype=int))
        # 1. Note that self.struct and self.positions created by IStructure_from_CIF() are ordered in the same way
        # 2. Write a case when IStructure_from_CIF() is not working, but self.struct is created by pymatgen from CIF
        # In this case self.positions should be created. Correct labels of orbits should be collected from CIF
        # or with the self of spglib.get_symmetry()
        self.positions['intersecting_sites']=close
        
        return self.positions
    
    def determine_orbits(self):
        """ This function determines the equivalent positions (orbits) from
            the list of fractional coordinates"""
        
        columns=['label','species','multiplicity','Wyckoff_symbol','occupancy']
        orbits_labels=list(set(self.positions['atom_site_label']))
        arr=[]
        for orb in orbits_labels:
            dx=self.positions.loc[self.positions['atom_site_label']==orb]
            arr.append([orb,dx['atom_site_type_symbol'].values[0],dx['atom_site_symmetry_multiplicity'].values[0],
                       dx['atom_site_Wyckoff_symbol'].values[0],dx['atom_site_occupancy'].values[0]])
        
        self.orbits=pd.DataFrame(arr,columns=columns)
        
        return self.orbits
    
    def determine_intersecting_orbits(self):
        """ 
        This function determines internal and external intersection 
        for functions based on the lists of intersection between sites 
        """

        self.orbits=self.determine_orbits()
        
        intersect_index=[]
        internal_intersection=[]
        external_intersection=[]
        intersect_orbits=[]
        vacancy_number=[]
        intersect_orbit_connected=[]
        intersect_orbit_connected_mult=[]
        intersect_orbit_connected_occ=[]
        
        # fast way to do it, if there is only ordered and substitutionally disordered orbits
        VPorbits=False
        for i,orb in enumerate(self.orbits['label'].values):
            if(len(self.orbits.iloc[i]['species'])==1):
                if(self.orbits.iloc[i]['occupancy']<1.0):
                    VPorbits=True
            elif(len(self.orbits.iloc[i]['species'])>1):
                if(self.orbits.iloc[i]['occupancy']<0.989):
                    VPorbits=True

        if(VPorbits==False):
            for i,orb in enumerate(self.orbits['label'].values):
                internal_intersection.append(False)
                external_intersection.append(False)
                intersect_index.append([1])
                intersect_orbits.append([])
                vacancy_number.append(self.vacancy_number(orb))
                intersect_orbit_connected.append([orb])
                intersect_orbit_connected_mult.append(self.orbits.iloc[i]['multiplicity'])
                intersect_orbit_connected_occ.append(self.orbits.iloc[i]['occupancy'])

            self.orbits['internal_intersection']=internal_intersection
            self.orbits['internal_intersect_index']=intersect_index
            self.orbits['vacancy_number']=vacancy_number
            self.orbits['external_intersection']=external_intersection
            self.orbits['intersecting_orbits']=intersect_orbits
            self.orbits['intersect_orbit_connected']=intersect_orbit_connected
            self.orbits['intersect_orbit_connected_mult']=intersect_orbit_connected_mult
            self.orbits['intersect_orbit_connected_occ']=intersect_orbit_connected_occ

        else:
            self.positions=self.determine_intersections()
        
            for orb in self.orbits['label'].values:
                db=self.positions.loc[self.positions['atom_site_label']==orb]
                ib=[]
                switch=0
                for j in db['intersecting_sites'].values:
                    for i in j:
                    # so if there are intersection and (1) there is intersection with the site form the same orbit
                    # then internal_intersection = True
                    # (2) there is intersection with site from another orbit, the name of that orbit is added to 
                    # the list of intersecting orbits 
                        if(i not in db.index.values):
                            ib.append(self.positions.iloc[i]['atom_site_label'])
                        elif(i in db.index.values):
                            switch=1
                if(switch==0 and len(ib)==0):
                # no internal or external intersection
                    internal_intersection.append(False)
                    external_intersection.append(False)
                    intersect_index.append([1])   # all orbits with no internal intersection have self-intersection
                                                # index 1
                elif(switch==0 and len(ib)!=0):
                # only external intersection
                    internal_intersection.append(False)
                    external_intersection.append(True)
                    intersect_index.append([1])   # all orbits with no internal intersection have self-intersection
                                                # index 1
                elif(switch!=0 and len(ib)==0):
                # only internal intersection
                    internal_intersection.append(True)
                    external_intersection.append(False)
                    # index = indexes of atoms on the orb which has internal intersection
                    index=self.positions.loc[self.positions['atom_site_label']==orb].index.values
                    # the list of lists of intersecting sites on orb
                    intersections=self.positions.loc[self.positions['atom_site_label']==orb]['intersecting_sites'].values
                    # building a praph of nodes
                    G = nx.Graph()
                    for ie in range(len(index)):
                        G.add_node(index[ie])
                        for je in intersections[ie]:
                            G.add_node(je)
                            G.add_edge(index[ie],je)
                    inter_sizes=list(set([len(i) for i in nx.connected_components(G)])) 
                    intersect_index.append(inter_sizes) # sometimes orbits with the internal intersection sites which form 
                                                        # groups of different size, that is why it is a list here
                else:
                # internal and external intersection
                    # in this case the list of itersecting orbits contains the orbit itself
                    internal_intersection.append(True)
                    external_intersection.append(True)
                    index=self.positions.loc[self.positions['atom_site_label']==orb].index.values
                    intersections=self.positions.loc[self.positions['atom_site_label']==orb]['intersecting_sites'].values
                    G = nx.Graph()
                    for ie in range(len(index)):
                        G.add_node(index[ie])
                        for je in intersections[ie]:
                            if(je in index): # because in case of external intersection there will 
                                            # be indexes belonging to different orbits
                                G.add_node(je)
                                G.add_edge(index[ie],je)
                    inter_sizes=list(set([len(i) for i in nx.connected_components(G)]))
                    intersect_index.append(inter_sizes)
                    
                ib=list(set(ib))
                intersect_orbits.append(ib)
                vacancy_number.append(self.vacancy_number(orb))
            
            
            self.orbits['internal_intersection']=internal_intersection
            self.orbits['internal_intersect_index']=intersect_index
            self.orbits['vacancy_number']=vacancy_number
            self.orbits['external_intersection']=external_intersection
            self.orbits['intersecting_orbits']=intersect_orbits
            
            # to refine the intersecting orbit determination we constract a graph where orbits are nodes 
            # and there is an edge between any two intersecting orbits. The connected components within this graph
            # show the list of 'effective' orbits in the compound.

            G=nx.Graph()
            for i,orb in enumerate(self.orbits['label'].values):
                G.add_node(orb)
                for je in intersect_orbits[i]:
                    G.add_edge(orb,je)
            connected_comp=[]
            for i in nx.connected_components(G):
                connected_comp.append(list(i)) # flag 1

            for orb in self.orbits['label'].values:
                for jj in connected_comp:
                    if(orb in jj):
                        intersect_orbit_connected.append(jj)
            
            self.orbits['intersect_orbit_connected']=intersect_orbit_connected
            
            # calculating multiplicities and total_occupancies of intersect_orbit_connected
            

            for ind,s_orb in enumerate(intersect_orbit_connected):
                s_orb=list(s_orb)
                index=self.positions.loc[self.positions['atom_site_label']==s_orb[0]].index.values
                intersections=self.positions.loc[self.positions['atom_site_label']==s_orb[0]]['intersecting_sites'].values
                for ind in range(1,len(s_orb)):
                    index=np.append(index,self.positions.loc[self.positions['atom_site_label']==s_orb[ind]].index.values)
                    intersections=np.append(intersections,self.positions.loc[self.positions['atom_site_label']==s_orb[ind]]['intersecting_sites'].values)

                G = nx.Graph()
                for ie in range(len(index)):
                    G.add_node(index[ie])
                    for je in intersections[ie]:
                        G.add_node(je)
                        G.add_edge(index[ie],je)
                cc_comp_mult=[]
                cc_comp_occ=[]
                for i in nx.connected_components(G):
                    cc_comp_mult.append(len(i))
                    m=0
                    for j in i:
                        m+=self.positions.iloc[j]['atom_site_occupancy']
                    cc_comp_occ.append(round(m,2))
                intersect_orbit_connected_mult.append(len(cc_comp_mult))
                if(len(set(cc_comp_occ))==1):
                    if(cc_comp_occ[0]>1 and cc_comp_occ[0]<self.occ_tol):
                        cc_comp_occ[0]=1.0
                    intersect_orbit_connected_occ.append(cc_comp_occ[0])
                    if(cc_comp_occ[0] > self.occ_tol):
                        self.errors.append('intersect_orbit_connected_occ > occ_tol')
                else:
                    intersect_orbit_connected_occ.append(list(set(cc_comp_occ)))
                    self.errors.append('no single value for intersect_orbit_connected_occ')
            
            self.orbits['intersect_orbit_connected_mult']=intersect_orbit_connected_mult
            self.orbits['intersect_orbit_connected_occ']=intersect_orbit_connected_occ
        
        return self.orbits, VPorbits
        
    
    def element_radiuses(self):
        """Crystal radiuses are given only for integer oxidation numbers, but
           sometimes CIF files contain float oxidations. This function rounds 
           oxidation states to use later in radius lookup table
           
           If element has oxidation state not present in the table of ionic radii, or
           its oxidation is 0, empirical radius is taken""" 
        
        element_radiuses=[]
        
        for el in self.positions['atom_site_type_symbol'].values:
            rad=[]
            species=Composition(el).elements
            # if several elements occupy the site, the average of the radiuses is taken
            # if oxidation state is non-integer, it is rounded to the smaller side to get the crystal radius
            for spec in species:
                elem=spec.element
                oxi=int(round(spec.oxi_state,0))
                if str(elem) in self.list_el:
                    rs=self.radius.loc[self.radius['symbol']==str(elem)].loc[self.radius['charge']==oxi]['ionic radius'].values
                    if(len(rs)>0):
                        rad.append(min(rs))
                    else:
                        rs=min(self.radius.loc[self.radius['symbol']==str(elem)]['empirical'].values)
                        rad.append(rs)
                else:
                    self.errors.append('Element '+str(elem)+' does not have radius data')
                    rad=np.nan
            
            element_radiuses.append(np.mean(rad))
        
        self.positions['radiuses']=element_radiuses
        return element_radiuses
    
    def distance(self,x1,x2,lattice,space_num):
        if(space_num < 16 or (space_num > 142 and space_num < 195)):
            x1c=np.dot(x1,lattice)
            x2c=np.dot(x2,lattice)
            x1images=np.zeros((27,3))
            x2images=np.zeros((27,3))

            for i in range(3):
                vec=np.array([[0, lattice[0][i], -lattice[0][i], lattice[1][i], \
                               -lattice[1][i],lattice[2][i],-lattice[2][i],\
                              lattice[0][i]+lattice[1][i],-lattice[0][i]-lattice[1][i],\
                              lattice[0][i]-lattice[1][i],-lattice[0][i]+lattice[1][i],\
                              lattice[2][i]+lattice[1][i],-lattice[2][i]-lattice[1][i],\
                              lattice[2][i]-lattice[1][i],-lattice[2][i]+lattice[1][i],\
                              lattice[0][i]+lattice[2][i],-lattice[0][i]-lattice[2][i],\
                              lattice[0][i]-lattice[2][i],-lattice[0][i]+lattice[2][i],\
                              lattice[0][i]+lattice[1][i]+lattice[2][i], \
                              -lattice[0][i]-lattice[1][i]-lattice[2][i],\
                              -lattice[0][i]+lattice[1][i]+lattice[2][i],\
                               lattice[0][i]-lattice[1][i]+lattice[2][i],\
                               lattice[0][i]+lattice[1][i]-lattice[2][i],\
                              -lattice[0][i]-lattice[1][i]+lattice[2][i],\
                               -lattice[0][i]+lattice[1][i]-lattice[2][i],\
                               lattice[0][i]-lattice[1][i]-lattice[2][i]]])

                x1images[:,i]=x1c[i]*np.ones(27)+vec
                x2images[:,i]=x2c[i]*np.ones(27)+vec

            d=cdist(x1images,x2images,'euclidean')
            dist=np.min(d)

        else:
            d=np.zeros(3)
            for i in range(3):
                if(x1[i]-x2[i]>0.5):
                    d[i]=(x1[i]-x2[i]-1.0)
                elif(x1[i]-x2[i]<-0.5):
                    d[i]=(x1[i]-x2[i]+1.0)
                else:
                    d[i]=x1[i]-x2[i]
            dc=np.dot(d,lattice)
            dist=np.sqrt(dc[0]**2+dc[1]**2+dc[2]**2)

        return dist 
    
    def distance_matrix(self):
        space_num=self.material.space_group
        lattice=self.struct.lattice.as_dict()['matrix']
        # the code assumes that self.positions and self.struct are ordered in the same way
        dist_matrix=np.zeros((len(self.positions),len(self.positions)))
        if(space_num < 16 or (space_num > 142 and space_num < 195)):
            for i in range(len(self.positions)):
                for j in range(i+1,len(self.positions)):
                    x1=np.array([self.positions.iloc[i]['atom_site_fract_x'],\
                        self.positions.iloc[i]['atom_site_fract_y'],\
                        self.positions.iloc[i]['atom_site_fract_z']])
                    x2=np.array([self.positions.iloc[j]['atom_site_fract_x'],\
                        self.positions.iloc[j]['atom_site_fract_y'],\
                        self.positions.iloc[j]['atom_site_fract_z']])

                    dist_matrix[i,j]=self.distance(x1,x2,lattice,space_num)
                    dist_matrix[j,i]=dist_matrix[i,j]
        else:
            dist_matrix=self.struct.distance_matrix
                    
        return dist_matrix
    
    def classify(self):
        self.orbits, _=self.determine_intersecting_orbits()
        orbit_disorder=[]
        for i,orb in enumerate(self.orbits['label'].values):
            # orbits without positional disorder component
            if(self.orbits.iloc[i]['external_intersection']==False and self.orbits.iloc[i]['internal_intersection']==False):
                if(len(self.orbits.iloc[i]['species'].keys())==1 and self.orbits.iloc[i]['occupancy']==1.0):
                    orbit_disorder.append('O') #1
                elif(len(self.orbits.iloc[i]['species'].keys())==1 and self.orbits.iloc[i]['occupancy']<1.0):
                    orbit_disorder.append('V') #2
                elif(self.orbits.iloc[i]['occupancy']>0.989 and len(self.orbits.iloc[i]['species'].keys())>1):
                    orbit_disorder.append('S') #3
                elif(self.orbits.iloc[i]['occupancy']<0.989 and len(self.orbits.iloc[i]['species'].keys())>1):
                    orbit_disorder.append('SV') #4
            # orbits with positional disorder component: have only internal intersection  
            elif(self.orbits.iloc[i]['internal_intersection']==True and self.orbits.iloc[i]['external_intersection']==False):
                if(len(self.orbits.iloc[i]['species'].keys())>1):
                    if(abs(self.orbits.iloc[i]['vacancy_number']-round(self.orbits.iloc[i]['vacancy_number'],0))<0.011):
                        orbit_disorder.append('SP') #7
                    else:
                        orbit_disorder.append('SVP') #8
                else:
                    if(abs(self.orbits.iloc[i]['vacancy_number']-round(self.orbits.iloc[i]['vacancy_number'],0))<0.005):
                        orbit_disorder.append('P') #5
                    else:
                        orbit_disorder.append('VP') #6
            # orbits with positional disorder componenet: have external intersection and/no internal intersection             
            elif(len(self.orbits.iloc[i]['intersect_orbit_connected'])>1):
                intersect_el=[]
                for intersect_orb in self.orbits.iloc[i]['intersect_orbit_connected']:
                    for j,orbj in enumerate(self.orbits['label'].values):
                        if(orbj==intersect_orb):
                            for el in self.orbits.iloc[j]['species'].keys():
                                intersect_el.append(el)
                intersect_el=set(intersect_el)

                if(self.orbits.iloc[i]['species'].keys()==intersect_el):
                    if(len(self.orbits.iloc[i]['species'].keys())>1):
                        if(type(self.orbits.iloc[i]['intersect_orbit_connected_occ'])!=list):
                            if(abs(self.orbits.iloc[i]['intersect_orbit_connected_occ']-round(self.orbits.iloc[i]['intersect_orbit_connected_occ']))<0.011):
                                orbit_disorder.append('SP') #11
                            else:
                                orbit_disorder.append('SVP') #12
                        else:
                            orbit_disorder.append('COM') #13                             
                           
                    else:
                        if(type(self.orbits.iloc[i]['intersect_orbit_connected_occ'])!=list):
                            if(abs(self.orbits.iloc[i]['intersect_orbit_connected_occ']-round(self.orbits.iloc[i]['intersect_orbit_connected_occ']))<0.011):
                                orbit_disorder.append('P') #9
                            else:
                                orbit_disorder.append('VP') #10
                        else:
                            orbit_disorder.append('COM') #13 
                else:
                    if(type(self.orbits.iloc[i]['intersect_orbit_connected_occ'])!=list):
                        if(abs(self.orbits.iloc[i]['intersect_orbit_connected_occ']-round(self.orbits.iloc[i]['intersect_orbit_connected_occ']))<0.011):
                            orbit_disorder.append('SP') #11
                        else:
                            orbit_disorder.append('SVP') #12
                    else:
                        orbit_disorder.append('COM') #13
                    
                          
        self.orbits['orbit_disorder']=orbit_disorder 
        return self.orbits
    
    def print_error(self):
        print(self.error)
        return
    
    def float_brackets(self,s):
        s_new=''
        for sym in s:
            if(sym=='(' or sym==')' or sym=="'"):
                pass
            else:
                s_new+=sym
        return float(s_new)
    