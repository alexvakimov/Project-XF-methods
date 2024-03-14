#!/usr/bin/env python

import sys
import cmath
import math
import os
import h5py
import matplotlib.pyplot as plt   # plots
import numpy as np
import time
import warnings

from liblibra_core import *
import util.libutil as comn
from libra_py import units
import libra_py.models.Holstein as Holstein
import libra_py.models.Tully as Tully
import libra_py.models.Subotnik as Subotnik
import libra_py.models.Esch_Levine as Esch_Levine
from libra_py import dynamics_plotting
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot
import libra_py.data_savers as data_savers

import libra_py.dynamics.exact.compute as dvr
import libra_py.dynamics.exact.save as dvr_save

warnings.filterwarnings('ignore')

colors = {}
colors.update({"11": "#8b1a0e"})  # red       
colors.update({"12": "#FF4500"})  # orangered 
colors.update({"13": "#B22222"})  # firebrick 
colors.update({"14": "#DC143C"})  # crimson   
colors.update({"21": "#5e9c36"})  # green
colors.update({"22": "#006400"})  # darkgreen  
colors.update({"23": "#228B22"})  # forestgreen
colors.update({"24": "#808000"})  # olive      
colors.update({"31": "#8A2BE2"})  # blueviolet
colors.update({"32": "#00008B"})  # darkblue  
colors.update({"41": "#2F4F4F"})  # darkslategray
colors.update({"51": "#000000"}) 

clrs_index = ["11", "21", "31", "41", "12", "22", "32", "13","23", "14", "24"]

def compute_model(q, params, full_id):

    model = params["model"]
    res = None
    
    if model==1:        
        res = Holstein.Holstein2(q, params, full_id) 
    elif model==2:
        res = Tully.Tully3(q, params, full_id)
    elif model==3:
        res = Subotnik.double_arch_geometry(q, params, full_id)
    elif model==4:
        res = Esch_Levine.JCP_2020(q, params, full_id)
    else:
        pass            

    return res


# Here, we define several sets of parameters:
# 
# * Sets 1 to 4 - for the 2-level Holstein Hamiltonians. These are just parabolas with constant coupling.
# * Sets 5 - for the ECWR (Extended Coupling With Reflection) Tully model
# * Sets 6 - for the DAG (Double Arch Geometry) or symmetrized ECWR Tully model
# * Sets 7 to 10 - for the n-level Esch-Levine Hamiltonians. These are just lines or bundles of lines crossing with yet another line and having a constant coupling. These models are for 2- to 5-state problems. 


model_params1 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.000}
model_params2 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.001}
#model_params3 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.01}
model_params3 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0,  0.0], "x_n":[0.0,  2.5],"k_n":[0.002, 0.005],"V":0.05}
model_params4 = {"model":1, "model0":1, "nstates":2, "E_n":[0.0, -0.01], "x_n":[0.0,  0.5],"k_n":[0.002, 0.008],"V":0.001}

model_params5 = {"model":2, "model0":2, "nstates":2} # ECR
model_params6 = {"model":3, "model0":3, "nstates":2} # DAG

model_params7 = {"model":4, "model0":4, "nstates":2, 
                 "w0":0.015, "w1":0.005, "V":0.005, "eps":0.0, "i_crit":2, "delta":0.01 } # Esch-Levine

model_params8 = {"model":4, "model0":4, "nstates":3, 
                 "w0":0.015, "w1":0.005, "V":0.005, "eps":0.0, "i_crit":3, "delta":0.01 } # Esch-Levine

model_params9 = {"model":4, "model0":4, "nstates":5, 
                 "w0":0.015, "w1":0.005, "V":0.005, "eps":0.0, "i_crit":4, "delta":0.01 } # Esch-Levine

model_params10 = {"model":4, "model0":4, "nstates":5, 
                 "w0":0.015, "w1":0.005, "V":0.005, "eps":0.02, "i_crit":3, "delta":0.01 } # Esch-Levine

all_model_params = [model_params1, model_params2, model_params3, model_params4, 
                    model_params5, model_params6, 
                    model_params7, model_params8, model_params9, model_params10
                   ]


# Choose the model to simulate here by setting `model_indx`.

# 0 - Holstein, trivial crossing, 2 level
# 1 - Holstein, strong nonadiabatic, 2 level
# 2 - Holstein, adiabatic, 2 level
# 3 - Holstein, double crossing, strong nonadiabatic, 2 level
# 4 - Tully, extended crossing with reflection, 2 level
# 5 - Double arch geometry or symmetrized ECWR, 2 level
# 6 - Esch-Levine, LZ-like, 2 level
# 7 - Esch-Levine, 1 crosses 2 parallel, 3 level
# 8 - Esch-Levine, 1 crosses 4 evenly-spaced parallel, 5 level
# 9 - Esch-Levine, 1 crosses 4 parallel split into 2 groups, 5 level

#################################
# Give the model used an index
model_indx = 7
################################

model_params = all_model_params[model_indx]

# For setting nsteps
list_nsteps = []
for i in range(len(all_model_params)):
    if all_model_params[i]["model"] == 1: #Holstein
        list_nsteps.append(8000)
    elif all_model_params[i]["model"] == 2: #ECR
        list_nsteps.append(4000)
    elif all_model_params[i]["model"] == 3: #DAG
        list_nsteps.append(3000)
    elif all_model_params[i]["model"] == 4: #Esch-Levine
        list_nsteps.append(8001)

NSTATES = model_params["nstates"]

# * `ndof` - number of nuclear degrees of freedom
# * `q` - nuclear coordinates, should be of length `ndof`
# * `p` - nuclear momenta, should be of length `ndof`
# * `mass` - nuclear masses, should be of length `ndof`
# * `force_constant` - should be of length `ndof`; this is the force constant of the harmonic potential that defines the width of the Gaussian wavepacket (that is the ground-state solution for such potential)
# * `init_type` - how to sample (or not) momenta and coordinates
# 
# For electronic variables:
# * `ndia` - the number of diabatic states
# * `nadi` - the number of adiabatic states
# * `rep` - representation in which we initialize the electronic variables:
#   - 0 - diabatic wfc;
#   - 1 - adiabatic wfc;
#   - 2 - diabatic density matrix;
#   - 3 - adiabatic density matrix;
# * `istates` - the populations of all `rep` states, should be of length `nadi`
# * `init_type` - how to sample amplitudes


#*********************** This is for the initial condition type **************************
#============== How nuclear DOFs are initialized =================
#icond_nucl = 0  # Coords and momenta are set exactly to the given value
#icond_nucl = 1  # Coords are set, momenta are sampled
#icond_nucl = 2  # Coords are sampled, momenta are set
icond_nucl = 3  # Both coords and momenta are sampled

nucl_params = { "ndof":1, "q":[-4.0], "p":[0.0], 
                "mass":[2000.0], "force_constant":[0.01], 
                "init_type":icond_nucl }

#============= How electronic DOFs are initialized ==================
#icond_elec = 2  # amplitudes all have the same phase
icond_elec = 3  # amplitudes gain random phase 

#============= Also select the representation ========================
# rep = 0 # diabatic wfc
rep = 1 # adiabatic wfc

istates = []
for i in range(NSTATES):
    istates.append(0.0)    
    
elec_params = {"verbosity":2, "init_dm_type":0,
               "ndia":NSTATES, "nadi":NSTATES, 
               "rep":rep, "init_type":icond_elec, "istates":istates
              }

#******************** This is for specific values in initial conditions *******************
#####################################
# Select a specific initial condition
icond_indx = 0
#####################################    
if model_indx in [0, 1]: # Holstein model
    if icond_indx==0:
        nucl_params["q"] = [-4.0]
        elec_params["istates"][0] = 1.0          
    elif icond_indx==1:
        nucl_params["q"] = [-1.0]
        elec_params["istates"][1] = 1.0 
elif model_indx in [2]: # Holstein model
    if icond_indx==0:
        nucl_params["q"] = [-4.0]
        elec_params["istates"][0] = 1.0          
    elif icond_indx==1:
        nucl_params["q"] = [-2.0]
        elec_params["istates"][1] = 1.0 
elif model_indx in [3]: # Holstein model
    if icond_indx==0:
        nucl_params["q"] = [-4.0]
        elec_params["istates"][0] = 1.0
    elif icond_indx==1:
        nucl_params["q"] = [-3.0]
        elec_params["istates"][1] = 1.0
elif model_indx in [4]: # Tully, ECR
    if icond_indx==0:
        nucl_params["q"] = [-15.0]
        nucl_params["p"] = [25.0]
        elec_params["istates"][0] = 1.0
elif model_indx in [5]: # Tully, DAG
    if icond_indx==0:
        nucl_params["q"] = [-20.0]
        nucl_params["p"] = [20.0]
        elec_params["istates"][0] = 1.0
elif model_indx in [6]: # Esch-Levine 2-level
    if icond_indx==0:
        nucl_params["q"] = [-1.0]
        nucl_params["p"] = [10.0]
        elec_params["istates"][1] = 1.0
elif model_indx in [7]: # Esch-Levine 3-level
    if icond_indx==0:
        nucl_params["q"] = [-1.0]
        nucl_params["p"] = [10.0]
        elec_params["istates"][2] = 1.0
elif model_indx in [8,9]: # Esch-Levine 5-level, two types
    if icond_indx==0:
        nucl_params["q"] = [-1.0]
        nucl_params["p"] = [10.0]
        elec_params["istates"][4] = 1.0

# For setting the initial state
state_indx = [i for i in range(len(elec_params["istates"])) if elec_params["istates"][i] > 0.5][0]


def potential(q, params):
    full_id = Py2Cpp_int([0,0]) 
    
    return compute_model(q, params, full_id)

exact_params = { "nsteps":8005, "dt":1.0, "progress_frequency":1/8000,
                 "rmin":[-100.0], "rmax":[300.0], "dx":[0.020], "nstates":model_params["nstates"],
                  "x0":nucl_params["q"], "p0":nucl_params["p"], "istate":[1,state_indx], "masses":[2000.0], "k":[0.01],
                  "integrator":"SOFT",
                  "wfc_prefix":F"wfc{model_indx}", "wfcr_rep":1, "wfcr_states":[0,1,2], "wfck_params":[0,0,0],
                  "mem_output_level":0, "txt_output_level":0, "txt2_output_level":0, "hdf5_output_level":3, 
                  "properties_to_save":[ "timestep", "time", "Epot_dia", "Ekin_dia", "Etot_dia",
                                         "Epot_adi", "Ekin_adi", "Etot_adi", "norm_dia", "norm_adi",
                                         "pop_dia", "pop_adi", "q_dia", "q_adi", "p_dia", "p_adi",
                                         "coherence_adi"],
                  "prefix":F"exact-model{model_indx}-icond{icond_indx}", "prefix2":F"exact-model{model_indx}-icond{icond_indx}",
                  "use_compression":0, "compression_level":[0, 0, 0]
               }

wfc = dvr.init_wfc(exact_params, potential, model_params)
savers = dvr_save.init_tsh_savers(exact_params, model_params, exact_params["nsteps"], wfc)
dvr.run_dynamics(wfc, exact_params, model_params, savers)
