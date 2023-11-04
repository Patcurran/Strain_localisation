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
region='B2_zone1' #change file path
DIC_data = {}
resolution = 40/2048
DIC_mean_shear_strain = []
DIC_std_shear_strain  = []
output ="../results/{}/".format(region)
plot_maps = False
dic_homog = {'A3_zone1':[(335, 10), (348, 180), (253, 307), (331, 329), (99, 208), (66, 116), (83, 13)],
           'A3_zone2':[(412, 39), (775, 254), (219, 262), (126, 631), (347, 535), (59, 119)],
           'A3_zone3':[(775, 73), (180, 69), (752, 783), (88, 684), (273, 175), (248, 360)],
           'B2_zone1':[(201, 335), (123, 243), (470, 64), (706, 427), (75, 547), (744, 153)]}

ebsd_homog = {'A3_zone1':[(408, 337), (412, 398), (377, 445), (405, 453), (323, 409), (314, 374), (319, 337)],
           'A3_zone2':[(280, 101), (342, 140), (245, 141), (227, 209), (266, 193), (216, 115)],
           'A3_zone3':[(334, 176), (229, 177), (328, 312), (213, 294), (237, 199), (240, 230)],
           'B2_zone1':[(217, 198), (198, 178), (275, 136), (331, 217), (193, 244), (330, 157)]}
##################################################################################################################
#IMPORT DIC DATA
for step in np.arange(1,12): #index from 1
    dicFilePath = "../DIC_data/B2/"
    if step <10:
        DicMap = hrdic.Map(dicFilePath, "B0000{}.txt".format(step))
    else:
        DicMap = hrdic.Map(dicFilePath, "B000{}.txt".format(step))
    
###################################################################################################################
#LINK DIC AND EBSD 
    DicMap.setPatternPath('01-1.BMP',1) 
    DicMap.setCrop(xMin=10, xMax=10, yMin=10, yMax=10)
    DicMap.setScale(micrometrePerPixel=resolution)

    EbsdFilePath = "../EBSD_data/CP large area/{}".format(region)
    EbsdMap = ebsd.Map(EbsdFilePath) #delete 'cubic'
    #EbsdMap.rotateData()

    EbsdMap.buildQuatArray()
    EbsdMap.findBoundaries(boundDef = 2) #degrees
    EbsdMap.findGrains(minGrainSize = 5) #pixels
    EbsdMap.calcGrainMisOri(calcAxis = False)
    EbsdMap.calcAverageGrainSchmidFactors(loadVector=[1,0,0])
    
    DicMap.homogPoints = dic_homog[region]
    EbsdMap.homogPoints = ebsd_homog[region]

    #DicMap.linkEbsdMap(EbsdMap, transformType='polynomial', order = 2)
    DicMap.linkEbsdMap(EbsdMap, transformType='affine')

    DicMap.findGrains(algorithm='warp')
    EbsdMap.calcAverageGrainSchmidFactors(loadVector=np.array([1,0,0]))
    
    if plot_maps: 
        DicMap.plotGrainAvMaxShear(vmin=0,vmax=0.01,plotColourBar=True,plotScaleBar=True,plotGBs= True ,dilateBoundaries=True)
        plt.tight_layout()
        plt.savefig(output + 'Grain_avg_ESS_DIC_{}'.format(step),dpi=1600)
        plt.close()
        DicMap.plotMaxShear(plotGBs=True, dilateBoundaries=True, plotColourBar=True, plotScaleBar=True, vmin=0, vmax=0.01)
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
    
    DIC_data['step_{}'.format(str(step))]=np.column_stack((grain_ID, grain_size, DIC_e11,shear_strain))

file_name = "{}_strain_DIC.pickle".format(region)
with open(file_name, "wb") as pickle_file:
    pickle.dump(DIC_data, pickle_file)