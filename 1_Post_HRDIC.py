#IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt5
from defdap import quat
from defdap import ebsd
from defdap import hrdic
import pandas as pd
import pickle

##################################################################################################################
#SAMPLE PARAMETERS
region='A3_zone2' #change file path
DIC_data = {}
resolution = 40/2048
DIC_mean_shear_strain = []
DIC_std_shear_strain  = []
output ="../results/{}/".format(region)
plot_maps = True
#slip activaiton
#A3_zone1 = 8
crop_area = {'A3_zone1':[30, 30, 20, 20],
            'A3_zone2':[30,40,10,10],
            'A3_zone3':[30,35,10,10]
            }

dic_homog = {'A3_zone1':[(1765, 335), (738, 215), (1373, 1167), (1550, 1579), (1816, 1790), (1055, 664), (309, 1141), (263, 681)],
           'A3_zone2':[(1120, 115), (2089, 703), (583, 1117), (944, 1458), (150, 869), (1508, 871)],
           'A3_zone3':[(610, 529), (2083, 216), (1571, 2080), (224, 1958), (243, 986), (850,1394), (1204, 1599)],
           'B2_zone1':[(1867, 112), (688, 1112), (1378, 293), (391, 796), (2755, 1534), (2730, 1035), (1933, 1727)]}

ebsd_homog = {'A3_zone1':[(402, 352), (335, 343), (376, 410), (388, 439), (405, 452), (356, 373), (307, 407), (303, 376)],
           'A3_zone2':[(280, 100), (342, 141), (244, 169), (266, 192), (215, 151), (305, 153)],
           'A3_zone3': [(236, 200), (334, 177), (299, 310), (211, 304), (213, 232), (245, 253), (274, 276)],
           'B2_zone1':[(292, 131), (218, 198), (261, 143), (198,177), (356, 225), (354, 191), (303, 242)]}

dicFilePath ={'A3_zone1':"../DIC_data/A3/zone_1/take 3/",
             'A3_zone2':"../DIC_data/A3/zone_2/take3/",
             'A3_zone3':"../DIC_data/A3/zone_3/take 3/",
             'B2_zone1':"../DIC_data/B2/take 2/"}

EbsdFilePath={'A3_zone1':'../EBSD_data/CP large area/A3_zone1',
              'A3_zone2':'../EBSD_data/CP large area/A3_zone2',
              'A3_zone3':'../EBSD_data/CP large area/A3_zone3',
              'B2_zone1':'../EBSD_data/CP large area/B2_zone1'}

EbsdFlip = {'A3_zone1':False,
              'A3_zone2':False,
              'A3_zone3':False,
              'B2_zone1':True}

min_grain ={'A3_zone1':50,
           'A3_zone2':50,
           'A3_zone3':50,
           'B2_zone1':150}
##################################################################################################################
#IMPORT DIC DATA
for step in np.arange(1,15): #index from 1
    if step <10:
        DicMap = hrdic.Map(dicFilePath[region], "B0000{}.txt".format(step))
    else:
        DicMap = hrdic.Map(dicFilePath[region], "B000{}.txt".format(step))
    
###################################################################################################################
#LINK DIC AND EBSD 
    DicMap.setPatternPath('01-1.BMP',1) 
    DicMap.setCrop(xMin=crop_area[region][0], xMax=crop_area[region][1], yMin=crop_area[region][2], yMax=crop_area[region][3])
    DicMap.setScale(micrometrePerPixel=resolution)

    EbsdMap = ebsd.Map(EbsdFilePath[region]) #delete 'cubic'
    
    if EbsdFlip[region]:
        EbsdMap.rotateData()

    EbsdMap.buildQuatArray()
    EbsdMap.findBoundaries(boundDef = 2) #degrees
    EbsdMap.findGrains(minGrainSize = min_grain[region]) #pixels
    EbsdMap.calcGrainMisOri(calcAxis = False)
    EbsdMap.calcAverageGrainSchmidFactors(loadVector=[1,0,0])
    
    DicMap.homogPoints = dic_homog[region]
    EbsdMap.homogPoints = ebsd_homog[region]

    #DicMap.linkEbsdMap(EbsdMap, transformType='polynomial', order = 2)
    DicMap.linkEbsdMap(EbsdMap, transformType='affine')

    DicMap.findGrains(algorithm='warp')
    EbsdMap.calcAverageGrainSchmidFactors(loadVector=np.array([1,0,0]))
#########################################################################################################################
#PLOT MAPS
    if plot_maps: 
        DicMap.plotGrainAvMaxShear(vmin=0,vmax=0.015,plotColourBar=True,plotScaleBar=True,plotGBs= True ,dilateBoundaries=True)
        plt.tight_layout()
        plt.savefig(output + 'Grain_avg_ESS_DIC_{}'.format(step),dpi=1600)
        plt.close()
        DicMap.plotMaxShear(plotGBs=True, dilateBoundaries=True, plotColourBar=True, plotScaleBar=True, vmin=0, vmax=0.015)
        plt.tight_layout()
        plt.savefig(output + 'ESS_DIC_{}'.format(step),dpi=1600)
        plt.close()
#######################################################################################################################
#ANALYSIS STRAINS
    grain_ID = []
    DIC_ESS = []
    grain_size = []
    DIC_e11 =[]
    DIC_e22 =[]
    DIC_e12 =[]
    for k in range(0,len(DicMap.grainList)):
        grain_ID.append(DicMap[k].ebsdGrain.grainID)
        grain_size.append(len(DicMap[k].maxShearList)*resolution)
    
    DIC_e11=(DicMap.calcGrainAv(DicMap.crop(DicMap.e11)))
    DIC_e22=(DicMap.calcGrainAv(DicMap.crop(DicMap.e22)))
    DIC_e12=(DicMap.calcGrainAv(DicMap.crop(DicMap.e12)))
    shear_strain=[]
    for i in range(len(DIC_e11)):    
        shear_strain.append( (( (DIC_e11[i]) - (DIC_e22[i]) )**2/2 + (DIC_e12[i])**2  )**0.5 )
    DIC_mean_shear_strain = np.mean(shear_strain)
    DIC_std_shear_strain = np.std(shear_strain)
#####################################################################################################################
#EXPORT DATA
    
    #DIC_data['step_{}'.format(str(step))]=np.column_stack((grain_ID, grain_size, DIC_e11,shear_strain))
    DIC_data['step_{}'.format(str(step))] = {'grain_ID':grain_ID,'grain_size ($\mu m^2$)':grain_size,'DIC_e11':DIC_e11,'DIC_shear_strain':shear_strain}
file_name = "{}_strain_DIC.pickle".format(region)
with open(file_name, "wb") as pickle_file:
    pickle.dump(DIC_data, pickle_file)