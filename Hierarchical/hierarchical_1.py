# -*- coding: utf-8 -*-
#This file is compatible with python 3.7
"""
Created on Tue Oct 10 10:14:42 2019

@author: dharik - Neha
"""

import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import collections
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
import seaborn as sns
from IPython import embed as IP
from textwrap import wrap


pd.options.mode.chained_assignment = None  # default='warn'

#from scipy.spatial.distance import pdist, squareform

# %%  Input data: Load and Renewables profiles for EnergyVille

SysType = 'North' # North or South

if SysType =='North':
    
    EnergyVilleLoadbyZone = pd.read_csv('Load.csv')

    RenewProfiles = pd.read_csv('Renewables.csv')



AnnualSystemLoadTWh = EnergyVilleLoadbyZone.sum().sum()/1000 # Electricity Generation in TWh

AnalysisType ='Bulk' # Bulk or DER , Bulk - ignore PVFT, PVLat, DER - Include PVFT


# %%  Aggregating Renewables and Load data as a single data frame
#ColumnNames = ['Time_index','Load_MW_z1','Load_MW_z2','Load_MW_Rur','SolarPV_Urb','SolarPV_Sub','SolarPV_Rur','SolarPV_Re','Solano_Wind','Central_Valley_North_Los_Banos_Wind','Greater_Carrizo_Wind']

if AnalysisType =='Bulk':
    ColumnNames = EnergyVilleLoadbyZone.columns.tolist() + RenewProfiles.columns.tolist()


# Total number of hours of the year = 8760
Nhours = len(EnergyVilleLoadbyZone)

#  Create a new dataframe storing aggregated load and renewables time series
AnnualTSeries= pd.concat([EnergyVilleLoadbyZone,RenewProfiles], axis=1)

# Storing load data in a normalized fashion - normalized load to be 0 and 1
#AnnualTSeries['Time_index'] = [int(x)  for x in range(1, Nhours+1)]


# %% Slicing Input data into columns of length = 24 X Ndays x NTSeries for the entire year
# Final output data from the k-means clustering process
#OutputColumnNames =ColumnNames + ['HourlyWeight']


def GetHierarchicalClusteringOutputs (InputData,NumGrpDays,NClusters, PeakDayinCluster, LoadWeight):
    # InputData - Annual timeseries of load and renewables data 
    ##- this could be either for a single year (8760 rows)or multiple year (e.g. 2 year data should have 17520 rows)
    # NumGrpDays- number of days in each subperiod
    # NClusters - number of subperiods to be generated as part of outputs
    # PeakDayinCluster - whether model should include subperiod with the peak system wide load ('yes' or 'no')
    # LoadWeight -  relative weight of load time series compared to renewables (default = 1)
    
    # Initialize dataframes to store final and intermediate data in
    OldColNames = InputData.columns.tolist()
    # CAUTION: Load Column lables should be named with the phrase "Load_"
    LoadColNames = InputData.columns[InputData.columns.str.contains('Load_')].tolist()
    
    # Columns to be reported in output files
    NewColNames = InputData.columns.tolist() + ['GrpWeight']
    # Dataframe storing final outputs
    FinalOutputData = pd.DataFrame(columns =NewColNames)
    
    # Dataframe storing normalized inputs
    AnnualTSeriesNormalized = pd.DataFrame(columns = OldColNames)
    
    Nhoursperyear = len(InputData)

   
    # Normalized all load and renewables data 0 and LoadWeight, All Renewables b/w 0 and 1
    for j in range(len(OldColNames)):
        AnnualTSeriesNormalized.loc[:,OldColNames[j]] = (InputData.loc[:,OldColNames[j]] -min(InputData.loc[:,OldColNames[j]]))/float(max(InputData.loc[:,OldColNames[j]])-min(InputData.loc[:,OldColNames[j]]))
        
        if OldColNames[j] in LoadColNames: 
            # If j corresponds to load columns scale them to be twice as important as Renewables
            AnnualTSeriesNormalized.loc[:,OldColNames[j]] =LoadWeight*AnnualTSeriesNormalized.loc[:,OldColNames[j]]
    
    # Identify hour with maximum system wide load
    hr_maxSysLoad = InputData.loc[:,['Load_MW_z1','Load_MW_z2','Load_MW_z3']].sum(axis=1).idxmax()
################################# pre-processing data to create concatenated column of load, pv and wind data
     
    # Number of such samples in a year - by avoiding division by float we are excluding a few days in each sample set
    # Hence annual generation for each zone will not exactly match up with raw data
    NumDataPoints =round(Nhoursperyear/24/NumGrpDays)

    DataPointAsColumns = ['p' + str(j) for j in range(1,NumDataPoints+1)]

    # Create a dictionary storing groups of time periods to average over for each hour
    HourlyGroupings = {i:[j for j in range(NumGrpDays*24*(i-1),NumGrpDays*24*i)] for i in range(1,NumDataPoints +1)}
    
    #  Create a new dataframe storing aggregated load and renewables time series
    ModifiedDataNormalized = pd.DataFrame(columns = DataPointAsColumns)
    # Original data organized in concatenated column
    ModifiedData = pd.DataFrame(columns = DataPointAsColumns)

    # Creating the dataframe with concatenated columns
    for j in range(0,NumDataPoints):
    
        if j==1: # Store  variable names for the concatenated column
            ConcatenatedRowNames = AnnualTSeriesNormalized.loc[HourlyGroupings[j+1],:].melt(id_vars=None)['variable']
    
        ModifiedDataNormalized[DataPointAsColumns[j]] = AnnualTSeriesNormalized.loc[HourlyGroupings[j+1],:].melt(id_vars=None)['value']
        ModifiedData[DataPointAsColumns[j]] = InputData.loc[HourlyGroupings[j+1],:].melt(id_vars=None)['value']

# Eliminate grouping including the hour with largest system laod (GW) - this group will be manually included in the outputs
    if PeakDayinCluster =='yes':
        #IP()
        GroupingwithPeakLoad =['p' + str(int(hr_maxSysLoad/24/NumGrpDays+1)) ]
        ClusteringInputDF = ModifiedDataNormalized.drop(GroupingwithPeakLoad,axis=1)
    else:
        ClusteringInputDF = ModifiedDataNormalized
        

################################## k-means clustering process
    # create Hierarchical clustering model and specify the number of clusters gathered
    # number of replications =100, squared euclidean distance
    
    if PeakDayinCluster =='yes': # If peak day in cluster, generate one less cluster
        NClusters = NClusters -1
    
    # Hierarchical clustering, need to write all arguments
    model = AgglomerativeClustering(n_clusters = 2, affinity )

    # Store clustered data
    # Create an empty list storing weight of each cluster
    EachClusterWeight  = [None]*NClusters 
    
    # Create an empty list storing name of each data point
    EachClusterRepPoint = [None]*NClusters

    for k in range(NClusters):
    # Number of points in kth cluster (i.e. label=0)
        EachClusterWeight[k] = len(model.labels_[model.labels_==k])

        # Compute Euclidean distance of each point from centroid of cluster k
        dist ={ClusteringInputDF.loc[:,model.labels_ ==k].columns[j]:np.linalg.norm(ClusteringInputDF.loc[:,model.labels_ ==k].values.transpose()[j] -model.cluster_centers_[k]) for j in range(EachClusterWeight[k])}

        # Select column name closest with the smallest euclidean distance to the mean
        EachClusterRepPoint[k] = min(dist, key = lambda k: dist[k])
    #IP()
# Storing selected groupings in a new data frame with appropriate dimensions (E.g. load in GW)  
    ClusterOutputDataTemp =  ModifiedData[EachClusterRepPoint]

# Selecting rows corresponding to Load in excluded subperiods and exclude them from scale factor calculation
    NRowsLoad = len(LoadColNames)
 # Excluding grouping with peak hr from scale factor calculation
    if PeakDayinCluster =='yes': 
         Actualdata = ModifiedData.loc[0:24*NumGrpDays*NRowsLoad-1,:].drop(GroupingwithPeakLoad,axis=1)
    else:
         Actualdata = ModifiedData.loc[0:24*NumGrpDays*NRowsLoad-1,:]

# Scale factor to adjust total generation in original data set to be equal to scaled up total generation in sampled data set    
    SampleweeksAnnualTWh = sum([ClusterOutputDataTemp.loc[0:24*NumGrpDays*NRowsLoad-1,EachClusterRepPoint[j]].sum()*EachClusterWeight[j] for j in range(NClusters)])    
    ScaleFactor =Actualdata.loc[0:24*NumGrpDays*NRowsLoad-1,:].sum().sum()/SampleweeksAnnualTWh
    
# Updated load values in GW
    ClusterOutputDataTemp.loc[0:24*NumGrpDays*NRowsLoad-1,:] = ScaleFactor*ClusterOutputDataTemp.loc[0:24*NumGrpDays*NRowsLoad-1,:]

    
# Add the grouping with the peak hour back into the cluster if that is excluded in the clustering
    if PeakDayinCluster =='yes':
       EachClusterRepPoint = EachClusterRepPoint + GroupingwithPeakLoad
       EachClusterWeight =EachClusterWeight +[1]
       ClusterOutputData =  pd.concat([ClusterOutputDataTemp, ModifiedData[GroupingwithPeakLoad]], axis=1, sort=False)
    else:
       ClusterOutputData =ClusterOutputDataTemp
       
    # Store weights for each selected hour  Number of days *24, for each week
    ClusteredWeights=pd.DataFrame(EachClusterWeight*np.ones([NumGrpDays*24,len(EachClusterWeight)]), columns = EachClusterRepPoint)

    # Storing weights in final output data column
    FinalOutputData['GrpWeight'] = ClusteredWeights.melt(id_vars=None)['value']

    # Regenerating data organized by time series (columns) and representative time periods (hours)
    for i in range(len(NewColNames)-1):
        FinalOutputData[NewColNames[i]] =ClusterOutputData.loc[ConcatenatedRowNames==NewColNames[i],:].melt(id_vars=None)['value']
    
   
    # Calculating error metrics and Annual profile
    FullLengthOutputs = FinalOutputData
    for j in range(len(EachClusterWeight)):
    # Selecting rows of the FinalOutputData dataframe to append
        df_try =FinalOutputData.truncate(before=NumGrpDays*24*j,after=NumGrpDays*24*(j+1) -1)
#        print(EachClusterWeight[j])
        if EachClusterWeight[j] >1: # Need to duplicate entries only weight is greater than 1
            FullLengthOutputs =FullLengthOutputs.append([df_try]*(EachClusterWeight[j]-1),ignore_index=True)
    
    #  Root mean square error between the duration curves of each time series 
    # Only conisder the points consider in the k-means clustering - ignoring any days dropped off from original data set  due to rounding
    RMSE = {OldColNames[j]:np.linalg.norm(np.sort(InputData.truncate(after=len(FullLengthOutputs)-1)[OldColNames[j]].values) 
    -np.sort(FullLengthOutputs[OldColNames[j]].values)) for j in range(len(OldColNames))}
    print(RMSE)


    return {'Data':FinalOutputData,                 # Scaled Output Load and Renewables profiles for the sampled representative groupings
            'ClusterWeights': EachClusterWeight,    # Weight of each for the representative groupings
            'AnnualGenScaleFactor':ScaleFactor,     # Scale factor used to adjust load output to match annual generation of original data
            'RMSE': RMSE,                           # Root mean square error between full year data and modeled full year data (duration curves)
           'AnnualProfile':FullLengthOutputs}, EachClusterRepPoint, EachClusterWeight       # Modeled duration curves GW
   

# %%  CLUSTERING OUTPUT ANALYSIS
    
############################## KEY INPUT INFORMATION
PossibleNumGrpDays =[1, 4, 7] # Defining number of consecutive days in each group
#PossibleNumGrpDays =[7] # Defining number of consecutive days in each group

 # Defining number of groupings to be selected
PossibleNClusters=[3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 21, 24, 27, 30, 35, 40, 45, 50, 52]  # Defining number of groupings to be selected

# Whether or not to include the grouping with the peak demand
#SelectPeakDayinCluster = ['yes','no']
SelectPeakDayinCluster = ['yes']

TruncateCF ='Yes'  # whether or not to truncate Capacity factors of Renewables <0.01

#for (i,j) in zip(range(len(PossibleNumGrpDays)), range(len(PossibleNClusters))):
# Storing clustering outputs by grouping size and number of clusters
OutputsbyGrpsbyNClusters= collections.OrderedDict() 


LoadWeightval =1 # Weighting load to be once or twice as important as Renewables - Energy paper reference

for i in range(len(PossibleNumGrpDays)):
    
    OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]] =collections.OrderedDict()
    
    for j in range(len(PossibleNClusters)):
#        print(PossibleNumGrpDays[i])
#        print(PossibleNClusters[j])
        
        OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]] = collections.OrderedDict()
         
        for k in range(len(SelectPeakDayinCluster)):
            print(PossibleNumGrpDays[i])
            print(PossibleNClusters[j])
                      
            OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]][SelectPeakDayinCluster[k]], EachClusterRepPoint, EachClusterWeight \
            = GetHierarchicalClusteringOutputs(AnnualTSeries,PossibleNumGrpDays[i],PossibleNClusters[j],
                                        SelectPeakDayinCluster[k],LoadWeightval)
            
#            # Adding code to truncate capacity factor values <0.01 to be either 0 or 0.01
#            if TruncateCF=='Yes':
#               df1 = OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]]['yes']['Data']
#
#               for r in ColumnNames[3:]:
#                   df1.loc[(df1[r]>0) & (df1[r]<0.01),r]=(df1.loc[ (df1[r]>0) & (df1[r]<0.01),r]).round(2)
#
#               OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]]['yes']['Data'] = df1
 
    
     
            
# %%  Exporting data as csv files simple case
NumGrpDays =7
peakhr ='yes'
#NClusters =22

for NClusters in PossibleNClusters:#[7]:#in range(2,52,5):
#for NClusters in range(2,53,1):
    df1 = OutputsbyGrpsbyNClusters[NumGrpDays][NClusters][peakhr]['Data']           
    
    df1['SubWtAsInGenXInputs'] =df1['GrpWeight']*NumGrpDays*24
    
    #df1.rename(columns ={"GrpWeight":"SubWtsAsInGenXInputs"}, inplace =True)
    
    
    if SysType =='North':
        #IP()
    
        df1.to_csv('hier_AZ_' +
                   'NumGrpDays_'+ str(NumGrpDays) +'_NClusters_' + str(NClusters) +
                   '_Peakhr_' + str(peakhr) + '_LoadWt_' + str(LoadWeightval) + '_'+ str(AnalysisType)+ '.csv', index=False) 
    


# Check on average capacity factor data       
#ReCols =['Northern_California_Solar', 'Solano_Solar','Central_Valley_North_Los_Banos_Solar', 'Westlands_Solar', 'Solano_Wind', 'Central_Valley_North_Los_Banos_Wind', 'Greater_Carrizo_Wind']   
#AvgCFDict ={j:np.multiply(df1[j].values,df1['GrpWeight']).sum()/df1['GrpWeight'].sum() for j in ReCols}
#
#AvgCFError = {j:(AnnualTSeries.mean()[j]-AvgCFDict[j])/AnnualTSeries.mean()[j]*100 for j in ReCols}

# %%  Plot RMSE error for load, solar, wind -subplot by region and legend by number of days
fontsizeval = 14

ForcePeakday ='yes'

# load profile
sns.set(font_scale=1.1, context ='talk',style ='darkgrid')
plt.figure(figsize=(6,10))
for k in range(3):  # three load profiles
 plt.subplot(3,1,k+1)
 for i in range(len(PossibleNumGrpDays)):
     plt.plot(PossibleNClusters,[OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]][ForcePeakday]['RMSE'][ColumnNames[k]] for j in range(len(PossibleNClusters))],label=str(PossibleNumGrpDays[i]) +'d')
     plt.title(ColumnNames[k] , fontsize=fontsizeval, fontweight='bold')
 if k==0:
     plt.legend(ncol=3,fontsize=fontsizeval, frameon=True)
 if k== 2:
     plt.xlabel('Number of Clusters',fontsize = fontsizeval, fontweight ='bold')
 if k==1:    
     plt.ylabel('Root Means Squared Error', fontsize = fontsizeval, fontweight ='bold')
 plt.subplots_adjust(hspace=0.3)
 plt.tick_params(axis='y',labelsize =fontsizeval)
 plt.tick_params(axis='x',labelsize =fontsizeval)
 plt.savefig('hier_LoadError_'+'MaxGrpDays_' + str(PossibleNumGrpDays[-1]) + 'MaxNumClusters_' + str(PossibleNClusters[-1]) +'Peakhr_'+ str(ForcePeakday) +'.png',bbox_inches="tight")

### Solar PV profile
##fontsizeval = 7
##sns.set(font_scale=1.1, context ='talk',style ='darkgrid')
##plt.figure(figsize=(6,10))
##for k in range(3,22):  # four PV profiles
## plt.subplot(5,4,k-2)
## for i in range(len(PossibleNumGrpDays)):
##     plt.plot(PossibleNClusters,[OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]][ForcePeakday]['RMSE'][ColumnNames[k]] for j in range(len(PossibleNClusters))],label=str(PossibleNumGrpDays[i]) +'d')
##     plt.title(ColumnNames[k] , fontsize=fontsizeval, fontweight='bold')
## if k==3:
##     plt.legend(bbox_to_anchor=(0.1, 0.99, 1., 0.802),ncol=5,fontsize=fontsizeval, frameon=True)
## if k== 21:
##     plt.xlabel('Number of Clusters',fontsize = fontsizeval, fontweight ='bold')
## if k==11:    
##     plt.ylabel('Root Means Squared Error', fontsize = fontsizeval, fontweight ='bold')
## plt.subplots_adjust(hspace=0.5)
## plt.tick_params(axis='y',labelsize =fontsizeval)
## plt.tick_params(axis='x',labelsize =fontsizeval)
## plt.savefig('PVError_'+'MaxGrpDays_' + str(PossibleNumGrpDays[-1]) + 'MaxNumClusters_' + str(PossibleNClusters[-1]) +'Peakhr_'+ str(ForcePeakday) +'.png',bbox_inches="tight")
##
### Wind profile
##fontsizeval = 7
##sns.set(font_scale=1.1, context ='talk',style ='darkgrid')
##plt.figure(figsize=(6,10))
##for k in range(22,34):  # four PV profiles
## #IP()
## plt.subplot(6,2,k-21)
## for i in range(len(PossibleNumGrpDays)):
##     plt.plot(PossibleNClusters,[OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]][ForcePeakday]['RMSE'][ColumnNames[k]] for j in range(len(PossibleNClusters))],label=str(PossibleNumGrpDays[i]) +'d')
##     plt.title(ColumnNames[k] , fontsize=fontsizeval, fontweight='bold')
## if k==22:
##     plt.legend(bbox_to_anchor=(0.1, 0.99, 1., 0.802),ncol=5,fontsize=fontsizeval, frameon=True)
## if k==33:
##     plt.xlabel('Number of Clusters',fontsize = fontsizeval, fontweight ='bold')
## if k==28:    
##     plt.ylabel('Root Means Squared Error', fontsize = fontsizeval, fontweight ='bold')
## plt.subplots_adjust(hspace=0.5)
## plt.tick_params(axis='y',labelsize =fontsizeval)
## plt.tick_params(axis='x',labelsize =fontsizeval)
## plt.savefig('WindError_'+'MaxGrpDays_' + str(PossibleNumGrpDays[-1]) + 'MaxNumClusters_' + str(PossibleNClusters[-1]) +'.png',bbox_inches="tight")


### Hydro profile
##fontsizeval = 7
##sns.set(font_scale=1.1, context ='talk',style ='darkgrid')
##plt.figure(figsize=(6,10))
##for k in range(34,37):#len(ColumnNames)):  # 34 load+solar+wind profiles
## #IP()
## plt.subplot(3,1,k-33)
## for i in range(len(PossibleNumGrpDays)):
##     plt.plot(PossibleNClusters,[OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]][ForcePeakday]['RMSE'][ColumnNames[k]] for j in range(len(PossibleNClusters))],label=str(PossibleNumGrpDays[i]) +'d')
##     plt.title(ColumnNames[k] , fontsize=fontsizeval, fontweight='bold')
## if k==34:
##     plt.legend(bbox_to_anchor=(0.1, 0.99, 1., 0.802),ncol=5,fontsize=fontsizeval, frameon=True)
## if k==35:
##     plt.xlabel('Number of Clusters',fontsize = fontsizeval, fontweight ='bold')
## if k==34:    
##     plt.ylabel('Root Means Squared Error', fontsize = fontsizeval, fontweight ='bold')
## plt.subplots_adjust(hspace=0.5)
## plt.tick_params(axis='y',labelsize =fontsizeval)
## plt.tick_params(axis='x',labelsize =fontsizeval)
## plt.savefig('HydroError_'+'MaxGrpDays_' + str(PossibleNumGrpDays[-1]) + 'MaxNumClusters_' + str(PossibleNClusters[-1]) +'.png',bbox_inches="tight")
##

# Total error - this makes no numerical sense

# %%Average Normalized RMSE for across all the time series
# Eq. 2 of Poncelet et al. - Sum_s[RMSE_s/(max_s -min_s)]/len(s)
# IEEE TRANSACTIONS ON POWER SYSTEMS, VOL. 32, NO. 3, MAY 2017    

##sns.set(font_scale=1.1, context ='talk',style ='darkgrid')
##plt.figure(figsize=(6,6))
##for i in range(len(PossibleNumGrpDays)):
##
## df1 = OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]][ForcePeakday]['AnnualProfile']
## RMSEval =OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]][ForcePeakday]['RMSE'].values()
## df1collist = df1.columns.tolist()
## NRMSE_avg =[sum([list(OutputsbyGrpsbyNClusters[PossibleNumGrpDays[i]][PossibleNClusters[j]][ForcePeakday]['RMSE'].values())[k]/
##     (max(df1.loc[:,df1collist[k]])-min(df1.loc[:,df1collist[k]])) for k in range(len(RMSEval))])/len(RMSEval) for j in range(len(PossibleNClusters))]
##
## plt.plot(PossibleNClusters,NRMSE_avg,label=str(PossibleNumGrpDays[i]) +'d')
## plt.title('Average Normalized RMSE across all series' , fontsize=fontsizeval, fontweight='bold')
## plt.legend(loc=1,ncol=2,fontsize=fontsizeval, frameon=True)
## plt.xlabel('Number of Clusters',fontsize = fontsizeval, fontweight ='bold')
## plt.ylabel('Average Normalized RMSE', fontsize = fontsizeval, fontweight ='bold')
## plt.subplots_adjust(hspace=0.5)
## plt.tick_params(axis='y',labelsize =fontsizeval)
## plt.tick_params(axis='x',labelsize =fontsizeval)
##
##plt.savefig('AvgNRMSE_' + str(SysType) +'_MaxGrpDays_' + str(PossibleNumGrpDays[-1]) + 'MaxNumClusters_' + str(PossibleNClusters[-1]) +'Peakhr_'+ str(ForcePeakday) +'.png',bbox_inches="tight")

# %% Plotting original data vs. clustered load data in terms of duration curves

#NClusters =10
ForcePeakday ='yes'

fontsizeval =12

FullLengthOutputs = OutputsbyGrpsbyNClusters[NumGrpDays][NClusters][ForcePeakday]['AnnualProfile']
RMSE =OutputsbyGrpsbyNClusters[NumGrpDays][NClusters][ForcePeakday]['RMSE']
# Plotting load profiles
sns.set(font_scale=1.1, context ='talk',style ='white')
plt.figure(figsize=(6,8))
for i in range(3):
 plt.subplot(3,1,i+1)
 plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(AnnualTSeries.truncate(after=len(FullLengthOutputs)-1)[ColumnNames[i]].values),label='Data')
 plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(FullLengthOutputs[ColumnNames[i]].values),label=str(NClusters) + '_Output')
 if i==0:
     plt.legend(bbox_to_anchor=(0.1, 0.99, 1., .102), ncol=2,fontsize=fontsizeval, frameon=True)
    
 plt.title(str(ColumnNames[i]) +':RMSE=' + str(round(RMSE[ColumnNames[i]],2)), fontsize=fontsizeval, fontweight='bold')
 if i==2:
     plt.xlabel('Hours of year',fontsize = fontsizeval, fontweight ='bold')
     plt.xlim(0,8760)
     plt.xticks(range(0,8760,1000))
 if i ==1:
     plt.ylabel('Power (GW)', fontsize = fontsizeval, fontweight ='bold')
 plt.subplots_adjust(hspace=0.7)
 plt.tick_params(axis='y',labelsize =fontsizeval)
 plt.tick_params(axis='x',labelsize =fontsizeval)
plt.savefig('hier_Load_DurationCurve_'+'GrpDays_' + str(NumGrpDays) + 'NumClusters_' + str(NClusters) +'Peakhr_'+ str(ForcePeakday) +'.png',bbox_inches="tight")


# Plotting solar CF profiles
fontsizeval = 6
sns.set(font_scale=1.1, context ='talk',style ='white')
plt.figure(figsize=(6,8))
for i in range(2):
 #IP()
 plt.subplot(5,4,i+1)
 plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(AnnualTSeries.truncate(after=len(FullLengthOutputs)-1)[ColumnNames[i+3]].values),label='Data')
 plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(FullLengthOutputs[ColumnNames[i+3]].values),label=str(NClusters) + '_Output')
 if i==18:
     #plt.legend(bbox_to_anchor=(0.1, 0.99, 1., .102), ncol=2, loc='lower right',fontsize=fontsizeval, frameon=True)
     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=fontsizeval, borderaxespad=0.)
    
 plt.title("\n".join(wrap(str(ColumnNames[i+3])+ str(round(RMSE[ColumnNames[i+3]],2)),30)), fontsize=fontsizeval)
 if i==19:
     plt.xlabel('Hours of year',fontsize = fontsizeval, fontweight ='bold')
     plt.xlim(0,8760)
     #plt.xticks(range(0,8760,1000))
 if i ==8:
     plt.ylabel('Capacity factor', fontsize = fontsizeval, fontweight ='bold')
 plt.subplots_adjust(hspace=0.7)
 plt.tick_params(axis='y',labelsize =fontsizeval)
 plt.tick_params(axis='x',labelsize =fontsizeval)
plt.savefig('hier_'+'SolarCFCurve_'+'GrpDays_' + str(NumGrpDays) + 'NumClusters_' + str(NClusters) +'Peakhr_'+ str(ForcePeakday) +'.png',bbox_inches="tight")

### Plotting Wind CF profiles
##sns.set(font_scale=1.1, context ='talk',style ='white')
##plt.figure(figsize=(6,8))
##for i in range(12):
## #IP()
## plt.subplot(6,2,i+1)
## plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(AnnualTSeries.truncate(after=len(FullLengthOutputs)-1)[ColumnNames[i+22]].values),label='Data')
## plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(FullLengthOutputs[ColumnNames[i+22]].values),label=str(NClusters) + '_Output')
## if i==0:
##     plt.legend(bbox_to_anchor=(0.1, 0.99, 1., .102), ncol=2,fontsize=fontsizeval, frameon=True)
##    
## plt.title(str(ColumnNames[i+22]) +':RMSE=' + str(round(RMSE[ColumnNames[i+22]],2)), fontsize=fontsizeval)
## if i==11:
##     plt.xlabel('Hours of year',fontsize = fontsizeval, fontweight ='bold')
##     plt.xlim(0,8760)
##     plt.xticks(range(0,8760,1000))
## if i ==4:
##     plt.ylabel('Capacity factor', fontsize = fontsizeval, fontweight ='bold')
## plt.subplots_adjust(hspace=0.7)
## plt.tick_params(axis='y',labelsize =fontsizeval)
## plt.tick_params(axis='x',labelsize =fontsizeval)
##plt.savefig('WindCFCurve_'+'GrpDays_' + str(NumGrpDays) + 'NumClusters_' + str(NClusters) +'Peakhr_'+ str(ForcePeakday) +'.png',bbox_inches="tight")
##
##
### Plotting Hydro CF profiles
##sns.set(font_scale=1.1, context ='talk',style ='white')
##plt.figure(figsize=(6,8))
##for i in range(3):
## #IP()
## plt.subplot(3,1,i+1)
## plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(AnnualTSeries.truncate(after=len(FullLengthOutputs)-1)[ColumnNames[i+34]].values),label='Data')
## plt.plot(np.arange(1,len(FullLengthOutputs)+1,1),np.sort(FullLengthOutputs[ColumnNames[i+34]].values),label=str(NClusters) + '_Output')
## if i==0:
##     plt.legend(bbox_to_anchor=(0.1, 0.99, 1., .102), ncol=2,fontsize=fontsizeval, frameon=True)
##    
## plt.title(str(ColumnNames[i+34]) +':RMSE=' + str(round(RMSE[ColumnNames[i+34]],2)), fontsize=fontsizeval)
## if i==1:
##     plt.xlabel('Hours of year',fontsize = fontsizeval, fontweight ='bold')
##     plt.xlim(0,8760)
##     plt.xticks(range(0,8760,1000))
## if i ==1:
##     plt.ylabel('Capacity factor', fontsize = fontsizeval, fontweight ='bold')
## plt.subplots_adjust(hspace=0.7)
## plt.tick_params(axis='y',labelsize =fontsizeval)
## plt.tick_params(axis='x',labelsize =fontsizeval)
##plt.savefig('HydroCFCurve_'+'GrpDays_' + str(NumGrpDays) + 'NumClusters_' + str(NClusters) +'Peakhr_'+ str(ForcePeakday) +'.png',bbox_inches="tight")
##



#IP()
#Writing appropriate files
ClusteringOutput = pd.read_csv('hier_' + 'AZ_' +
                   'NumGrpDays_'+ str(NumGrpDays) +'_NClusters_' + str(NClusters) +
                   '_Peakhr_' + str(peakhr) + '_LoadWt_' + str(LoadWeightval) + '_'+ str(AnalysisType)+ '.csv')#('Data_AllPV_North_NumGrpDays_7_NClusters_10_Peakhr_yes_LoadWt_1_Bulk_May19.csv')

Original_Input = pd.read_csv('Generators_variability_original.csv')

ColumnNames_Load = EnergyVilleLoadbyZone.columns.tolist()

ColumnNames_Generator_cluster = RenewProfiles.columns.tolist()


ColumnNames_Generator = ['Resource','biomass_1','conventional_hydroelectric_2','geothermal_3','hydroelectric_pumped_storage_4','natural_gas_fired_combined_cycle_5','natural_gas_fired_combustion_turbine_6','onshore_wind_turbine_7','small_hydroelectric_8','solar_photovoltaic_9','biomass_10','conventional_hydroelectric_11','geothermal_12','hydroelectric_pumped_storage_13','natural_gas_fired_combined_cycle_14','natural_gas_fired_combustion_turbine_15','onshore_wind_turbine_16','small_hydroelectric_17','solar_photovoltaic_18','biomass_19','conventional_hydroelectric_20','conventional_steam_coal_21','conventional_steam_coal_22','conventional_steam_coal_23','conventional_steam_coal_24','hydroelectric_pumped_storage_25','natural_gas_fired_combined_cycle_26','natural_gas_fired_combined_cycle_27','natural_gas_fired_combined_cycle_28','natural_gas_fired_combustion_turbine_29','nuclear_30','onshore_wind_turbine_31','solar_photovoltaic_32','biomass_33','conventional_hydroelectric_34','conventional_steam_coal_35','hydroelectric_pumped_storage_36','natural_gas_fired_combined_cycle_37','natural_gas_fired_combustion_turbine_38','onshore_wind_turbine_39','solar_photovoltaic_40','biomass_41','conventional_hydroelectric_42','conventional_steam_coal_43','geothermal_44','natural_gas_fired_combined_cycle_45','natural_gas_fired_combustion_turbine_46','onshore_wind_turbine_47','solar_photovoltaic_48','biomass_49','conventional_hydroelectric_50','conventional_steam_coal_51','geothermal_52','hydroelectric_pumped_storage_53','natural_gas_fired_combined_cycle_54','natural_gas_fired_combustion_turbine_55','nuclear_56','onshore_wind_turbine_57','solar_photovoltaic_58','biomass_59','conventional_hydroelectric_60','natural_gas_fired_combined_cycle_61','natural_gas_fired_combustion_turbine_62','solar_photovoltaic_63','naturalgas_ccccsavgcf_mid_64','naturalgas_ccavgcf_mid_65','naturalgas_ctavgcf_mid_66','landbasedwind_ltrg1_mid_67','landbasedwind_ltrg1_mid_68','offshorewind_otrg3_mid_69','utilitypv_losangeles_mid_70','utilitypv_losangeles_mid_71','utilitypv_losangeles_mid_72','battery_mid_73','nuclear_mid_74','naturalgas_ccccsavgcf_mid_75','naturalgas_ccavgcf_mid_76','naturalgas_ctavgcf_mid_77','landbasedwind_ltrg1_mid_78','landbasedwind_ltrg1_mid_79','landbasedwind_ltrg1_mid_80','offshorewind_otrg3_mid_81','utilitypv_losangeles_mid_82','utilitypv_losangeles_mid_83','utilitypv_losangeles_mid_84','battery_mid_85','nuclear_mid_86','naturalgas_ccccsavgcf_mid_87','naturalgas_ccavgcf_mid_88','naturalgas_ctavgcf_mid_89','landbasedwind_ltrg1_mid_90','landbasedwind_ltrg1_mid_91','landbasedwind_ltrg1_mid_92','utilitypv_losangeles_mid_93','utilitypv_losangeles_mid_94','utilitypv_losangeles_mid_95','battery_mid_96','nuclear_mid_97','naturalgas_ccccsavgcf_mid_98','naturalgas_ccavgcf_mid_99','naturalgas_ctavgcf_mid_100','landbasedwind_ltrg1_mid_101','landbasedwind_ltrg1_mid_102','landbasedwind_ltrg1_mid_103','utilitypv_losangeles_mid_104','utilitypv_losangeles_mid_105','battery_mid_106','nuclear_mid_107','naturalgas_ccccsavgcf_mid_108','naturalgas_ccavgcf_mid_109','naturalgas_ctavgcf_mid_110','landbasedwind_ltrg1_mid_111','landbasedwind_ltrg1_mid_112','landbasedwind_ltrg1_mid_113','utilitypv_losangeles_mid_114','utilitypv_losangeles_mid_115','battery_mid_116','nuclear_mid_117','naturalgas_ccccsavgcf_mid_118','naturalgas_ccavgcf_mid_119','naturalgas_ctavgcf_mid_120','landbasedwind_ltrg1_mid_121','landbasedwind_ltrg1_mid_122','landbasedwind_ltrg1_mid_123','landbasedwind_ltrg1_mid_124','landbasedwind_ltrg1_mid_125','landbasedwind_ltrg1_mid_126','landbasedwind_ltrg1_mid_127','landbasedwind_ltrg1_mid_128','landbasedwind_ltrg1_mid_129','landbasedwind_ltrg1_mid_130','offshorewind_otrg3_mid_131','utilitypv_losangeles_mid_132','utilitypv_losangeles_mid_133','utilitypv_losangeles_mid_134','utilitypv_losangeles_mid_135','utilitypv_losangeles_mid_136','utilitypv_losangeles_mid_137','utilitypv_losangeles_mid_138','battery_mid_139','nuclear_mid_140','naturalgas_ccccsavgcf_mid_141','naturalgas_ccavgcf_mid_142','naturalgas_ctavgcf_mid_143','landbasedwind_ltrg1_mid_144','landbasedwind_ltrg1_mid_145','landbasedwind_ltrg1_mid_146','landbasedwind_ltrg1_mid_147','utilitypv_losangeles_mid_148','utilitypv_losangeles_mid_149','utilitypv_losangeles_mid_150','utilitypv_losangeles_mid_151','utilitypv_losangeles_mid_152','utilitypv_losangeles_mid_153','battery_mid_154','nuclear_mid_155','naturalgas_ccccs_100_157','naturalgas_ccccs_100_158','naturalgas_ccccs_100_159','naturalgas_ccccs_100_160','naturalgas_ccccs_100_161','naturalgas_ccccs_100_162','naturalgas_ccccs_100_163']

Representative_weeks = [int(x[1:]) for x in EachClusterRepPoint]#[24,2,8,12,31,39,17,4,45,34]

#Saving clustering data in a load_data file
#LoadSeries= pd.DataFrame(columns = ColumnNames_Load)
LoadSeries= pd.DataFrame()
Nhours = len(ClusteringOutput)

LoadSeries['Time_index'] = list(range(1,Nhours+1))
LoadSeries['Load_MW_z1'] = ClusteringOutput['Load_MW_z1'][0:Nhours] 
LoadSeries['Load_MW_z2'] = ClusteringOutput['Load_MW_z2'][0:Nhours]
LoadSeries['Load_MW_z3'] = ClusteringOutput['Load_MW_z3'][0:Nhours]



LoadSeries.to_csv('hier_Load_data.csv',index=False)

#Saving clustering data in a generators_variability file

GeneratorSeries=pd.DataFrame(columns = ColumnNames_Generator)
GeneratorSeries['Resource'] = Original_Input['Resource'][0:Nhours]
ColumnNames_Generator.remove('Resource')

for j in ColumnNames_Generator:
    GeneratorSeries[j] = 1

for k in ColumnNames_Generator_cluster:
    GeneratorSeries[k] = ClusteringOutput[k][0:Nhours] 

GeneratorSeries.columns = Original_Input.columns.tolist()

GeneratorSeries.to_csv('Generators_variability.csv',index=False)


