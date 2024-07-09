#IMPORT PACKAGES
import matflow as mf
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
#SAMPLES PARAMET1RS
region = 'A3_zone2'
local_validated = 'dev'
plot_maps = False

if local_validated==True:
    output = '../results/local_validated/{}/'.format(region)   
elif local_validated==False:
    output= '../results/macro_validated/{}/'.format(region)   
elif local_validated=='dev':
    output= '../results/dev_validated/{}/'.format(region)   


if local_validated==True: #local validated
    WorkFlow = {'A3_zone1':'E:/Matflow_exports/RVE_extrusion_2024-05-26-111128',
               'A3_zone2':'D:/Matflow_exports/RVE_extrusion_2024-05-26-111014',
               'A3_zone3':'D:/Matflow_exports/RVE_extrusion_2024-05-26-110927',
               'B2_zone1':'E:/Matflow_exports/'}
elif local_validated=='dev': #macro validated
    print('dev')
    wk_path = {'A3_zone1':'E:/Matflow_exports/RVE_extrusion_EBSD_2024-06-29_220803.zip',
              'A3_zone2':'E:/Matflow_exports/RVE_extrusion_EBSD_2024-06-29_220908.zip',
              'A3_zone3':'E:/Matflow_exports/RVE_extrusion_EBSD_2024-06-29_220627.zip'}

    
crop_data = {'A3_zone1':[292,363,335,195],
             'A3_zone2':[220,210,100,80],
             'A3_zone3':[190,271,168,135]}

##################################################################################################################
#LOAD SIMULATION DATA
wk = mf.Workflow(wk_path[region])
sim_elem = wk.tasks.simulate_VE_loading_damask.elements[0].outputs.VE_response

mean_strain = sim_elem.value['phase_data']['vol_avg_strain']['data']
mean_stress = sim_elem.value['phase_data']['vol_avg_stress']['data']

np.save(output+ '{}_mean_strain.npy'.format(region),mean_strain) #---------------------------------------------edit
np.save(output+ '{}_mean_stress.npy'.format(region),mean_stress) #---------------------------------------------edit

strain = sim_elem.value['grain_data']['epsilon_U^0(F)']['data']

file_name = "{}_strain_DIC.pickle".format(region)
with open(file_name, "rb") as pickle_file:
    DIC_data = pickle.load(pickle_file)
grain_id=DIC_data['step_1']['grain_ID']
####################################################################################################################
#ANALYSIS STRAIN
k=5 # name contant
SIM_data={}
for step in range(0,len(strain)):
    strain2 = strain[step]
    shear_strain =[]
    e11_strain=strain[step][:,0,0]
    for i in range(0,len(strain2)):
        shear_strain.append( ( ( strain2[i, 0, 0] - strain2[i,1,1] )**2/2 + strain2[i,1,0]**2  )**0.5 )
    
    #file_name = "{}_strain_DIC.pickle".format(region)
    #with open(file_name, "rb") as pickle_file:
    #    DIC_data = pickle.load(pickle_file)
    
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
        strain_field = sim_elem.value['field_data']['epsilon_U^0(F)']['data']
        z_layer = 7
        strain_field2 = strain_field[step]
        buff_size_xy = crop_data[region]#(292,363,335,195)
        map_slice = (slice(buff_size_xy[0], -buff_size_xy[1]), slice(buff_size_xy[2], -buff_size_xy[3]), z_layer) 
        cropped_strain = strain_field2[map_slice]
        
        rows = len(cropped_strain)
        cols = len(cropped_strain[0,:])
        shear_strain_field = np.zeros((rows, cols)) 
        
        for i in range(rows):
            for j in range(cols):
                values = ((cropped_strain[i,j,0,0]-cropped_strain[i,j,1,1])**2/2 + cropped_strain[i,j,1,0]**2)**0.5
                shear_strain_field[i, j] = values
        
        plt.figure(figsize=(4,3),dpi=1600) #figsize=(3,2),dpi=1600
        #plt.imshow( (strain[map_slice].T),vmin=0,vmax=0.015)
        plt.imshow( (shear_strain_field.T),vmin=0,vmax=0.015)
        plt.colorbar(label= 'Effective shear strain')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(output+ 'CP_ESS_{}.png'.format(step+k))
        plt.close()
###############################################################################################################################        
        strain3 = sim_elem.value['grain_data']['epsilon_U^0(F)']['data']
        strain2 = strain3[step]
        shear_strain2 =[]
        for i in range(0,len(strain2)):
            shear_strain2.append( ( ( strain2[i, 0, 0] - strain2[i,1,1] )**2/2 + strain2[i,1,0]**2  )**0.5 )
        
        grain_map = sim_elem.value['field_data']['grain']['data']
        shear_strain_map = np.zeros_like(np.array(grain_map),dtype=float) # making a grain map with zero strains
        for i_grain in np.unique(np.array(grain_map)):
            shear_strain_map[np.array(grain_map) == i_grain] = shear_strain2[i_grain]
        
        plt.figure(figsize=(4,3),dpi=1600) #figsize=(3,2),dpi=1600
        plt.imshow( (shear_strain_map[map_slice].T),vmin=0,vmax=0.015)
        #plt.imshow( (shear_strain[map_slice].T)*scaling,vmin=0,vmax=0.01)
        plt.colorbar(label=' Strain')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(output+ 'CP_grain_avg_ESS_{}.png'.format(step+k))
        plt.close()
########################################################################################################################    
#EXPORT DATA
    #strain_comparison['step_{}'.format(str(step))]=np.column_stack((grain_ID, grain_size, DIC_grain_strain))
    SIM_data['step_{}'.format(str(step+k))]={'CP_e11':SIM_e11,'CP_shear_strain':shear_strain}

#ANLYSIS SLIP
slip = sim_elem.value['grain_data']['gamma_sl']
slip_data = slip['data']
np.save(output + '{}_slip_data.npy'.format(region),slip_data)    #------------------------------------------------------------------
#strain_comparison={}
#for step in range(0,len(strain)):
#    strain_comparison['step_{}'.format(str(step+k))] = {(DIC_data['step_{}'.format(step+k)], SIM_data['step_{}'.format(step+k)] }
    #strain_comparison  

file_name = (output+ "{}_CP_strain.pickle".format(region))
with open(file_name, "wb") as pickle_file:
    pickle.dump(SIM_data, pickle_file)