#IMPORT PACKAGES
from matflow import load_workflow
import numpy as np
import matplotlib.pyplot as plt
from defdap import quat
from defdap import ebsd
from defdap import hrdic
import pickle
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#################################################################################################################
#SAMPLES PARAMETERS
region = 'B2_zone1'
output = '../results/{}/'.format(region)   
plot_maps = True
WorkFlow = {'A3_zone1':'E:/Matflow_exports/RVE_extrusion_2023-09-15-135907',
           'A3_zone2':'E:/Matflow_exports/RVE_extrusion_2023-04-27-001619',
           'A3_zone3':'E:/Matflow_exports/RVE_extrusion_2023-04-27-205744',
           'B2_zone1':'E:/Matflow_exports/RVE_extrusion_2023-10-18-155222'}
crop_data = {'A3_zone1':[292,363,335,195],
             'A3_zone2':[220,210,100,80],
             'A3_zone3':[190,271,168,135]}

##################################################################################################################
#LOAD SIMULATION DATA
workflow = load_workflow(WorkFlow[region])
sim_task = workflow.tasks.simulate_volume_element_loading
vol_elem_resp = sim_task.elements[0].outputs.volume_element_response

mean_strain = vol_elem_resp['phase_data']['vol_avg_strain']['data']
mean_stress = vol_elem_resp['phase_data']['vol_avg_stress']['data']
np.save('{}_mean_strain.npy'.format(region),mean_strain)
np.save('{}_mean_stress.npy'.format(region),mean_stress)

strain = vol_elem_resp['grain_data']['epsilon_U^0(F)']['data']
####################################################################################################################
#ANALYSIS DATA
c=3
SIM_data={}
for step in range(0,len(strain)):
    strain2 = strain[step]
    shear_strain =[]
    e11_strain=strain[step][:,0,0]
    
    for i in range(0,len(strain2)):
        shear_strain.append( ( ( strain2[i, 0, 0] - strain2[i,1,1] )**2/2 + strain2[i,1,0]**2  )**0.5 )
    
    file_name = "{}_strain_DIC.pickle".format(region)
    with open(file_name, "rb") as pickle_file:
        DIC_data = pickle.load(pickle_file)
    
    grain_id=[]
    for n in range(0,len(DIC_data['step_{}'.format(step+c)][:,0])):
        grain_id.append(int(DIC_data['step_{}'.format(step+c)][:,0][n]))  
    
    SIM_ESS = []
    SIM_e11 = []
    for m in grain_id:
        SIM_ESS.append(shear_strain[m])
        SIM_e11.append(e11_strain[m])
        
    e_11_list = []  # Initialize inside the outer loop for each step
    e_22_list = []  # Initialize inside the outer loop for each step
    e_12_list = []  # Initialize inside the outer loop for each step
    
    for i in grain_id:
        strain3 = strain[step]
        e_11_list.append(strain3[i, 0, 0])
        e_22_list.append(strain3[i, 1, 1])
        e_12_list.append(strain3[i, 1, 0])
    
    shear_strain = []
    for i in range(len(grain_id)):
        shear_strain.append(((e_11_list[i] - e_22_list[i]) ** 2 / 2 + e_12_list[i] ** 2) ** 0.5)
########################################################################################################################
#PLOT MAPS
    if plot_maps:
        strain3 = vol_elem_resp['grain_data']['epsilon_U^0(F)']['data']
        strain2 = strain3[step]
        shear_strain2 =[]
        for i in range(0,len(strain2)):
            shear_strain2.append( ( ( strain2[i, 0, 0] - strain2[i,1,1] )**2/2 + strain2[i,1,0]**2  )**0.5 )
   
        grain_map = vol_elem_resp['field_data']['grain']['data']
        shear_strain_map = np.zeros_like(grain_map,dtype=float) # making a grain map with zero strains
        for i_grain in np.unique(grain_map):
            shear_strain_map[grain_map == i_grain] = shear_strain2[i_grain]
        
        z_layer = 7
        buff_size_xy = (crop_data[region][0],crop_data[region][1],crop_data[region][2],crop_data[region][3])
        map_slice = (slice(buff_size_xy[0], -buff_size_xy[1]), slice(buff_size_xy[2], -buff_size_xy[3]), z_layer)
    
        plt.figure(dpi=1200) #figsize=(3,2),dpi=1600
        plt.imshow( (shear_strain_map[map_slice].T),vmin=0,vmax=0.015)
        #plt.imshow( (shear_strain[map_slice].T)*scaling,vmin=0,vmax=0.01)
        plt.colorbar(label=' Strain')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(output+ 'CP_grain_avg_ESS_{}.png'.format(step))
        plt.close()
########################################################################################################################    
#EXPORT DATA
    #strain_comparison['step_{}'.format(str(step))]=np.column_stack((grain_ID, grain_size, DIC_grain_strain))
    SIM_data['step_{}'.format(str(step+c))]=np.column_stack([SIM_e11,shear_strain])

    
strain_comparison={}
for step in range(0,len(strain)):
    strain_comparison['step_{}'.format(str(step+c))] = np.column_stack((DIC_data['step_{}'.format(step+c)], SIM_data['step_{}'.format(step+c)] ))
#strain_comparison  

file_name = "{}_strain_comparison.pickle".format(region)
with open(file_name, "wb") as pickle_file:
    pickle.dump(strain_comparison, pickle_file)
