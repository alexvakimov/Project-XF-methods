#!/usr/bin/env python
# coding: utf-8

# # Nonadiabatic dynamics based on exact factorization in Libra
# 
# 
# ## Table of Content: <a name="TOC"></a>
# 
# 1. [Generic setups](#1)
# 
# 2. [Theoretical Background](#2)
# 
#    2.1. [Decoherence-Induced Surface Hopping based on XF (DISH-XF) or SHXF](#2.1)
# 
# 3. [Model Hamiltonians](#3)
# 
# 4. [Choosing the Nonadiabatic Dynamics Methodology](#4)
# 
# 5. [Choosing initial conditions: Nuclear and Electronic](#5)
# 
# 6. [Running the calculations](#6)
# 
# 7. [Plotting the results](#7)
# 
# 8. [Comparison with quantum dynamics](#8)
# 

# ## 1. Generic setups
# <a name="1"></a>[Back to TOC](#TOC)
# 
# Here, we import all necessary libraries, set up some definitions (e.g. colors), and define the function that would be calling model Hamiltonians also defined within Libra package.
# 
# Packages that set dynamic parameters are imported from the recipes directory. Their names specify their corresponding nonadiabatic dynamics methods. Specifically, nonadiabatic dynamics methods based on exact factorization (XF) contain 'xf' in their names. Try to run other methods for comparison.
# 
#     from recipes import shxf, mqcxf, ehxf
#     from recipes import ehrenfest_adi_ld, ehrenfest_dia, mfsd
#     from recipes import fssh, sdm, bcsh

# In[1]:


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

from recipes import shxf, mqcxf, ehxf
from recipes import ehrenfest_adi_ld, ehrenfest_adi_nac, ehrenfest_dia, mfsd
from recipes import fssh, sdm, bcsh

import libra_py.dynamics.exact.compute as dvr
import libra_py.dynamics.exact.save as dvr_save

#from matplotlib.mlab import griddata
#%matplotlib inline 
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


# ## 3. Model Hamiltonians
# <a name="3"></a>[Back to TOC](#TOC)
# 
# First, let's define the `compute_model` function that returns all the necessary objects and properties for the dynamics. 

# In[2]:


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

# In[3]:


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

# In[4]:


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


# Here, visualization of each model system is done. Change parameters accordingly.

# In[5]:


# New plotting:
# Common setups
plot_params = {"figsize":[24, 6], "titlesize":24, "labelsize":28, "fontsize": 28, "xticksize":26, "yticksize":26,
               "colors": colors, "clrs_index": clrs_index,
               "prefix":F"case", "save_figures":0, "do_show":1,
               "plotting_option":1, "nac_idof":0 }


# In[6]:


list_states = [x for x in range(model_params["nstates"])]

# Holstein
#plot_params.update( { "xlim":[-4, 5], "ylim":[-0.01*5, 0.03*5], "ylim2":[-2, 2], "show_nac_abs":1 })
#dynamics_plotting.plot_surfaces(compute_model, [ model_params ], list_states, -4.0, 5.0, 0.05, plot_params)

# Tully
#plot_params.update( { "xlim":[-15.0, 15.0], "ylim":[-0.4, 0.4], "ylim2":[-0.4, 0.4], "show_nac_abs":1 })
#dynamics_plotting.plot_surfaces(compute_model, [ model_params ], list_states, -15.0, 30.0, 0.05, plot_params)

# Esch-Levine
#plot_params.update( { "xlim":[-4, 4], "ylim":[-0.06, 0.06], "ylim2":[-3, 3], "show_nac_abs":1 })
#dynamics_plotting.plot_surfaces(compute_model, [ model_params ], list_states, -4.0, 8.0, 0.05, plot_params)


# #### 4. Choosing the Nonadiabatic Dynamics Methodology 
# <a name="4"></a>[Back to TOC](#TOC)
# 
# In this section, we go over parameters to set up a computational methodology. 
# 
# Let's start with the simulation-specific parameters:
# 
# * `nsteps` -  how many steps of dynamics to compute
# * `nstaj` - how many trajectories to use
# * `nstates:2` - all our models are 2-level systems
# * `dt:1` - nuclear integration timestep in a.u. of time
# * `num_electronic_substeps` - do multiple steps of electronic integration per nuclear step
# * `isNBRA` and `is_nbra` - is set to `1`, will turn on some simplifications and optimization for NBRA type of calculations. Here, we are doing the non-NBRA case
# * `frogress_frequency:0.1` - printing out a message evry `2500 x 0.1 = 250` steps
# * `which_adi_states` - properties of which adiabatic states to save, we only have 2
# * `which_dia_states` - properties of which diabatic states to save, we only have 2
# * `mem_output_level:4` - how much data to save into the hdf5 output files. This is the most intensive output, usually needed only for some special cases (extra-analysis, debugging, new methods, demonstration like htis, etc.)
# * `properties_to_save` - list of properties to be computed on the fly and saved into hdf5 output file.

# In[7]:


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


# In[8]:


NSTATES = model_params["nstates"]
#list_nsteps[model_indx]
dyn_general = { "nsteps":list_nsteps[model_indx], "ntraj":2000, "nstates":NSTATES,
                "dt":1.0, "num_electronic_substeps":1, "isNBRA":0, "is_nbra":0,
                "progress_frequency":0.1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),      
                "mem_output_level":5,
                "properties_to_save":[ "timestep", "time", "q", "p", "f", "Cadi", "Cdia", "Epot_ave", "Ekin_ave", "Etot_ave",
                "states", "se_pop_adi", "se_pop_dia", "sh_pop_adi", "dc1_adi"],
                "prefix":"adiabatic_md", "prefix2":"adiabatic_md"
              }

#[ "timestep", "time", "q", "p", "f", "Cadi", "Cdia", "Epot_ave", "Ekin_ave", "Etot_ave",
#                "se_pop_adi", "se_pop_dia", "sh_pop_adi", "hvib_adi", "hvib_dia", "St", "basis_transform", "D_adi" ]


# Now, it is time to select the type of calculations we want to do. Keep in mind that some options are related to each other, so usually one would need to correlate the choices. For methods based on surface hopping, default options are used for frustrated hops and how to rescale momenta on hops.

# In[9]:


#################################
# Give the recipe above an index
method_indx = 7
#################################

if method_indx == 0:
    ehrenfest_dia.load(dyn_general)  # Ehrenfest, dia
elif method_indx == 1:
    ehrenfest_adi_nac.load(dyn_general)  # Ehrenfest, adi with nac
elif method_indx == 2:
    mfsd.load(dyn_general)  # MFSD
    
elif method_indx == 3:
    fssh.load(dyn_general)  # FSSH
elif method_indx == 4:
    sdm.load(dyn_general)  # SDM with default EDC parameters
elif method_indx == 5:
    bcsh.load(dyn_general)  # BCSH 

elif method_indx == 6:
    shxf.load(dyn_general)  # SHXF
elif method_indx == 7:
    mqcxf.load(dyn_general)  # MQCXF
elif method_indx == 8:
    ehxf.load(dyn_general)  # EhXF 

#Update properties to save in the case of XF-based methods
if method_indx in [6]:
    dyn_general["properties_to_save"] += ["p_quant", "VP", "q_aux", "p_aux", "nab_phase"]
elif method_indx in [7,8]:
    dyn_general["properties_to_save"] += ["p_quant", "VP", "f_xf", "q_aux", "p_aux", "nab_phase"]    


# ## 5. Choosing initial conditions: Nuclear and Electronic
# <a name="5"></a>[Back to TOC](#TOC)
# 
# The setup of the parameters below is rather intuitive:
# 
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

# In[10]:


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

## (time-depdent) wave packet width

#dyn_general["wp_width"] = 0.3



#dyn_general.update({"use_td_width": 0})

#dyn_general.update({"wp_v":0.0})
#dyn_general["wp_v"] = 1./np.sqrt(4*2000.)/dyn_general["wp_width"] # free Gaussian



# ## 6. Running the calculations
# <a name="6"></a>[Back to TOC](#TOC)

# In[52]:
dyn_general.update({"use_td_width": 0})
wp_width = MATRIX(1,1); wp_width.set(0,0,0.3)
dyn_general["wp_width"] = wp_width

#wp_v = MATRIX(1,1); wp_v.set(0,0,0.0004)
#dyn_general["wp_v"] = wp_v

dyn_general.update({"tp_algo": 1})
dyn_general.update({"project_out_aux":1})
#dyn_general["electronic_integrator"]=6
#nucl_params["force_constant"] = [0.001]
#nucl_params["p"] = [10.0]
#integ = dyn_general["electronic_integrator"]

dyn_params = dict(dyn_general)
dyn_params.update({ "prefix":F"model{model_indx}-method{method_indx}-icond{icond_indx}", 
  "prefix2":F"model{model_indx}-method{method_indx}-icond{icond_indx}" })
  
print(F"Computing model{model_indx}-method{method_indx}-icond{icond_indx}")    
#dyn_params.update({ "prefix":F"model{model_indx}-method{method_indx}-icond{icond_indx}-v{wp_v.get(0,0)}", 
#  "prefix2":F"model{model_indx}-method{method_indx}-icond{icond_indx}-v{wp_v.get(0,0)}" })
#  
#print(F"Computing model{model_indx}-method{method_indx}-icond{icond_indx}-v{wp_v.get(0,0)}")    

rnd = Random()
res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params, elec_params, nucl_params, rnd)


# ## 7. Plotting the results
# <a name="7"></a>[Back to TOC](#TOC)

# In[53]:


#============ Plotting ==================
pref = F"model{model_indx}-method{method_indx}-icond{icond_indx}"
#pref = F"model{model_indx}-method{method_indx}-icond{icond_indx}-v{wp_v.get(0,0)}"

plot_params = { "prefix":pref, "filename":"mem_data.hdf", "output_level":3,
                "which_trajectories":[0, 1, 2, 3, 4], "which_dofs":[0], "which_adi_states":list(range(NSTATES)), 
                "which_dia_states":list(range(NSTATES)), 
                "frameon":True, "linewidth":3, "dpi":300,
                "axes_label_fontsize":(8,8), "legend_fontsize":8, "axes_fontsize":(8,8), "title_fontsize":8,
                "what_to_plot":["coordinates", "momenta",  "forces", "energies", "phase_space", "se_pop_adi",
                                "se_pop_dia", "sh_pop_adi" ], 
                "which_energies":["potential", "kinetic", "total"],
                "save_figures":1, "do_show":0
              }

#"what_to_plot":["coordinates", "momenta",  "forces", "energies", "phase_space", "se_pop_adi",
#                                "se_pop_dia", "sh_pop_adi", "traj_resolved_adiabatic_ham", "traj_resolved_diabatic_ham", 
#                                "time_overlaps", "basis_transform"
#                               ], 

tsh_dynamics_plot.plot_dynamics(plot_params)


# In[13]:


#Check the norm conservation
#with h5py.File(F"model{model_indx}-method{method_indx}-icond{icond_indx}/mem_data.hdf", 'r') as f:
#    #plt.title("Monitoring the norm")
#    plt.plot(f["time/data"][:]/41.0, f["se_pop_adi/data"][:, 0]+f["se_pop_adi/data"][:, 1], label="norm",
#linewidth=2, color = colors["11"])
#    #plt.savefig(pref + "/norm.png")

#Check the energy conservation
#with h5py.File(F"model{model_indx}-method{method_indx}-icond{icond_indx}/mem_data.hdf", 'r') as f:
    #plt.title("Monitoring the norm")
#    plt.plot(f["time/data"][:]/41.0, f["Etot_ave/data"][:]-f["Etot_ave/data"][0], label="E_tot",
#linewidth=2, color = colors["11"])    

#with h5py.File(F"model{model_indx}-method{method_indx}-icond{icond_indx}/mem_data.hdf", 'r') as f:
#    #plt.title("Monitoring the norm")
#    plt.plot(f["time/data"][:]/41.0, f["dc1_adi/data"][:, 0, 0, 0, 1], linewidth=2, color = colors["11"])
#    #plt.plot(f["time/data"][:]/41.0, f["dc1_adi/data"][:, 0, 0, 1, 0], linewidth=2, color = colors["21"])    


# In[14]:


if method_indx in [6, 8]:
    plot_params["what_to_plot"] = ["q_aux", "p_aux", "nab_phase", "phase_space_aux", "p_quant"]
elif method_indx in [7]:
    plot_params["what_to_plot"] = ["q_aux", "p_aux", "nab_phase", "phase_space_aux", "p_quant", "VP", "f_xf"]

tsh_dynamics_plot.plot_dynamics_xf(plot_params)


# In[30]:



## ## 8. Comparison with quantum dynamics
## <a name="8"></a>[Back to TOC](#TOC)
#
## In[11]:
#
#
## For setting the initial state
#state_indx = [i for i in range(len(elec_params["istates"])) if elec_params["istates"][i] > 0.5][0]
#
#
#def potential(q, params):
#    full_id = Py2Cpp_int([0,0]) 
#    
#    return compute_model(q, params, full_id)
#
#
## In[13]:
#
#
#exact_params = { "nsteps":int(list_nsteps[model_indx]), "dt":1.0, "progress_frequency":1/8000,
#                 "rmin":[-25.0], "rmax":[25.0], "dx":[0.025], "nstates":model_params["nstates"],
#                  "x0":nucl_params["q"], "p0":nucl_params["p"], "istate":[1,state_indx], "masses":[2000.0], "k":[0.01],
#                  "integrator":"SOFT",
#                  "wfc_prefix":F"wfc{model_indx}", "wfcr_rep":1, "wfcr_states":[0,1], "wfck_params":[0,0,0],
#                  "mem_output_level":0, "txt_output_level":0, "txt2_output_level":0, "hdf5_output_level":3, 
#                  "properties_to_save":[ "timestep", "time", "Epot_dia", "Ekin_dia", "Etot_dia",
#                                         "Epot_adi", "Ekin_adi", "Etot_adi", "norm_dia", "norm_adi",
#                                         "pop_dia", "pop_adi", "q_dia", "q_adi", "p_dia", "p_adi",
#                                         "coherence_adi"],
#                  "prefix":F"exact-model{model_indx}-icond{icond_indx}", "prefix2":F"exact-model{model_indx}-icond{icond_indx}",
#                  "use_compression":0, "compression_level":[0, 0, 0]
#               }
#
#
## In[14]:
#
#
#wfc = dvr.init_wfc(exact_params, potential, model_params)
#savers = dvr_save.init_tsh_savers(exact_params, model_params, exact_params["nsteps"], wfc)
#dvr.run_dynamics(wfc, exact_params, model_params, savers)
#
#
## In[15]:
#
#
#plt.figure(figsize=(2*6.42, 2*2.41))
#plt.xticks(fontsize=8)
#with h5py.File(F"exact-model{model_indx}-icond{icond_indx}/data.hdf", 'r') as f:        
#    plt.plot(f["time/data"][:]/41.0, f["pop_adi/data"][:, 0, 0], label="P0_adi", linewidth=2, color = colors["11"])
#    plt.plot(f["time/data"][:]/41.0, f["pop_adi/data"][:, 1, 0], label="P1_adi", linewidth=2, color = colors["21"])
#    #plt.plot(f["time/data"][:]/41.0, f["pop_adi/data"][:, 2, 0], label="P2_adi", linewidth=2, color = colors["31"])
#    #plt.plot(f["time/data"][:]/41.0, f["pop_adi/data"][:, 3, 0], label="P3_adi", linewidth=2, color = colors["41"])
#    #plt.plot(f["time/data"][:]/41.0, f["pop_adi/data"][:, 4, 0], label="P4_adi", linewidth=2, color = colors["51"])
#    #plt.plot(f["time/data"][:]/41.0, f["pop_dia/data"][:, 0, 0], label="P0_adi", linewidth=2, color = colors["11"])
#    #plt.plot(f["time/data"][:]/41.0, f["pop_dia/data"][:, 1, 0], label="P1_adi", linewidth=2, color = colors["21"])
#    #plt.plot(f["time/data"][:]/41.0, f["pop_adi/data"][:, 3, 0], label="P3_adi", linewidth=2, color = colors["41"])
#    #plt.plot(f["time/data"][:]/41.0, f["pop_adi/data"][:, 4, 0], label="P4_adi", linewidth=2, color = colors["32"])
#    plt.legend()



