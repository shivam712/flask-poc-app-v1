import pandas as pds
import numpy as np
import math

global count
global count2
count=0
count2=0

pds.set_option("display.precision", 10)

class getRawData:
    
        stateExposureTable = pds.DataFrame()
        vehicleClassTable = pds.DataFrame()
        stateCountTable = pds.DataFrame()
        vehicleTypeWeightTable = pds.DataFrame() 
        rawDataProtection1Table = pds.DataFrame()
        constantValuesProtection1Table = pds.DataFrame()
        rawDataProtection2Table = pds.DataFrame()
        constantValuesProtection2Table = pds.DataFrame()

        logNormalizationProtection1Array = np.zeros(8)
        logNormalizationProtection2Array = np.zeros(12)
        def __init__(self):
                
                self.stateExposureTable = self.getStateExposureTable()
                self.vehicleClassTable = self.getVehicleClassTable()
                self.stateCountTable = self.getStateCountTable()
                self.vehicleTypeWeightTable = self.getVehicleTypeWeightTable()  
                self.rawDataProtection1Table = self.getRawDataProtection1()
                self.constantValuesProtection1Table = self.getConstantValuesProtection1()
                self.rawDataProtection2Table = self.getRawDataProtection2()
                self.constantValuesProtection2Table = self.getConstantValuesProtection2()

                self.logNormalizationProtection1Array = self.getLogNormalizationProtection1Array()        
                self.logNormalizationProtection2Array = self.getLogNormalizationProtection2Array()   

        def getStateExposureTable(self): 
                file=('Data Set/state_exposure_data.csv')
                self.stateExposureTable = pds.DataFrame(pds.read_csv(file))
                self.stateExposureTable['Exposure Percentage'] = self.stateExposureTable['Exposure Value']/(self.stateExposureTable['Exposure Value'].sum()) * 100
                return self.stateExposureTable

        def getVehicleClassTable(self): 
                file=('Data Set/vehicle_class_data.csv')
                self.vehicleClassTable = pds.DataFrame(pds.read_csv(file))
                self.vehicleClassTable.columns = ['Vehicle Class','Percentage']
                self.vehicleClassTable['Percentage'] = (self.vehicleClassTable['Percentage'].str.replace('%','')).astype(int)
                return self.vehicleClassTable

        def getStateCountTable(self):
                self.stateExposureTable = self.getStateExposureTable()
                self.stateCountTable = pds.DataFrame(self.stateExposureTable['Violation/No Fault'])
                self.stateCountTable = self.stateCountTable.groupby(['Violation/No Fault']).size().reset_index(name='Count')
                self.stateCountTable.columns = ['State','Count'] 
                self.stateCountTable['Percentage'] = self.stateCountTable['Count']/(self.stateCountTable['Count'].sum()) * 100
                return self.stateCountTable

        '''
        # No idea why this function is not working
        def funGetWeight(self,numVehicle,state):
                return (self.vehicleClassTable.loc[self.vehicleClassTable['Vehicle Class'] == self.numVehicle , 'Percentage'].item() * self.stateCountTable.loc[self.stateCountTable['State']==self.state,'Percentage'].item()/ 100)
        '''
        
        def getVehicleTypeWeightTable(self):
                temp = {'Vehicle Class':['Vehicle1','Vehicle2','Vehicle1','Vehicle2'],
                        'Type':['Violation','Violation','No Fault','No Fault',],
                        'Weight':[ (round(self.vehicleClassTable.loc[self.vehicleClassTable['Vehicle Class'] == "Vehicle1" , 'Percentage'].item()) * round(self.stateCountTable.loc[self.stateCountTable['State']=="Violation",'Percentage'].item())/ 100),
                                   (round(self.vehicleClassTable.loc[self.vehicleClassTable['Vehicle Class'] == "Vehicle2" , 'Percentage'].item()) * round(self.stateCountTable.loc[self.stateCountTable['State']=="Violation",'Percentage'].item())/ 100),
                                   (round(self.vehicleClassTable.loc[self.vehicleClassTable['Vehicle Class'] == "Vehicle1" , 'Percentage'].item()) * round(self.stateCountTable.loc[self.stateCountTable['State']=="No Fault",'Percentage'].item())/ 100),
                                   (round(self.vehicleClassTable.loc[self.vehicleClassTable['Vehicle Class'] == "Vehicle2" , 'Percentage'].item()) * round(self.stateCountTable.loc[self.stateCountTable['State']=="No Fault",'Percentage'].item())/ 100),
                                ]}
                                   #Can be made better
                self.vehicleTypeWeightTable = pds.DataFrame(temp)
                self.vehicleTypeWeightTable['Weight'] = self.vehicleTypeWeightTable['Weight'].round(0)
                return self.vehicleTypeWeightTable


        def getRawDataProtection1(self):
                file=('Data Set/raw_data_protection1.csv')
                return pds.DataFrame(pds.read_csv(file)) 

        def getConstantValuesProtection1(self):
                file=('Data Set/values_protection1_data.csv')
                return pds.DataFrame(pds.read_csv(file)) 

        def getRawDataProtection2(self):
                file=('Data Set/raw_data_protection2.csv')
                return pds.DataFrame(pds.read_csv(file)) 

        def getConstantValuesProtection2(self):
                file=('Data Set/values_protection2_data.csv')
                return pds.DataFrame(pds.read_csv(file)) 

        def getLogNormalizationProtection1Array(self):
                retention = (self.constantValuesProtection1Table.to_numpy())[2]
                retention = np.delete(retention,0)
                retention = retention.astype(np.float64)

                mu = (self.constantValuesProtection1Table.to_numpy())[3]
                mu = np.delete(mu,0)
                mu = mu.astype(np.float64)

                sigma = (self.constantValuesProtection1Table.to_numpy())[4]
                sigma = np.delete(sigma,0)
                sigma = sigma.astype(np.float64)

                maxClaim = (self.constantValuesProtection1Table.to_numpy())[5]
                maxClaim = np.delete(maxClaim,0)
                maxClaim = maxClaim.astype(np.float64)

                log1 = (sigma*maxClaim*0.5)*(1/(maxClaim*math.sqrt(2*3.14159265)*sigma)*np.exp(-(((np.log(maxClaim)-mu)**2)/(2*(sigma**2)))))
                log2 = (sigma*retention*0.5)*(1/(retention*math.sqrt(2*3.14159265)*sigma)*np.exp(-(((np.log(retention)-mu)**2)/(2*(sigma**2)))))
                return log2/log1
        
        def getLogNormalizationProtection2Array(self):
                retention = (self.constantValuesProtection2Table.to_numpy())[0]
                retention = np.delete(retention,0)
                retention = retention.astype(np.float64)

                mu = (self.constantValuesProtection2Table.to_numpy())[1]
                mu = np.delete(mu,0)
                mu = mu.astype(np.float64)

                sigma = (self.constantValuesProtection2Table.to_numpy())[2]
                sigma = np.delete(sigma,0)
                sigma = sigma.astype(np.float64)

                maxClaim = (self.constantValuesProtection2Table.to_numpy())[3]
                maxClaim = np.delete(maxClaim,0)
                maxClaim = maxClaim.astype(np.float64)

                log1 = (sigma*maxClaim*0.5)*(1/(maxClaim*math.sqrt(2*3.14159265)*sigma)*np.exp(-(((np.log(maxClaim)-mu)**2)/(2*(sigma**2)))))
                log2 = (sigma*retention*0.5)*(1/(retention*math.sqrt(2*3.14159265)*sigma)*np.exp(-(((np.log(retention)-mu)**2)/(2*(sigma**2)))))
                return log2/log1

def protection1():
        a = getRawData()

        intTable1 = a.rawDataProtection1Table.drop([0,1,2])
        intTable1 = intTable1.iloc[:,1:]
        intTable1 = intTable1.astype(np.float64)

        def funIntermediateTable1(x):
                return ((x-1)*a.logNormalizationProtection1Array)+1

        intTable1 = np.apply_along_axis(funIntermediateTable1,1,np.array(intTable1))
        intTable1 = pds.DataFrame(intTable1)
        intTable1.insert(loc=0, column='Months', value=np.arange(15,31,1))

        intTable2 = pds.DataFrame(np.zeros([14,9]))
        intTable2.iloc[:,0] = np.arange(1,15,1)

        firstRowIntTable1 = intTable1.iloc[0,1:]
        twelthRowIntTable1 = intTable1.iloc[12,1:]

        def funIntermediateTable2(x):
                global count
                count+=1
                return firstRowIntTable1**((np.log(firstRowIntTable1)/np.log(twelthRowIntTable1))**((15-count)/(27-15)))*(15/count)

        intTable2 = pds.DataFrame(np.apply_along_axis(funIntermediateTable2,1,np.array(intTable2.iloc[:,1:])))
        intTable2.insert(loc=0, column='Months', value=np.arange(1,15,1))

        intTable3 = pds.concat([intTable2,intTable1])

        def getWeightValue(vehicleClass,stateType):
                return (((a.vehicleTypeWeightTable.loc[(a.vehicleTypeWeightTable['Vehicle Class'] == vehicleClass) &
                (a.vehicleTypeWeightTable['Type']==stateType)])['Weight']).to_numpy())[0]

        exposureArray = [getWeightValue("Vehicle1","Violation"),getWeightValue("Vehicle1","Violation"),
                        getWeightValue("Vehicle2","No Fault"),getWeightValue("Vehicle2","No Fault"),
                        getWeightValue("Vehicle1","No Fault"),getWeightValue("Vehicle1","No Fault"),
                        getWeightValue("Vehicle2","Violation"),getWeightValue("Vehicle2","Violation")]


        intTable4 = intTable3.iloc[:,1:]

        def funIntermediateTable4(x):
                return (exposureArray/x)*0.01

        intTable4 = pds.DataFrame(np.apply_along_axis(funIntermediateTable4,1,intTable4))

        finalIncurred = intTable4.iloc[:,0]+intTable4.iloc[:,2]+intTable4.iloc[:,4]+intTable4.iloc[:,6]
        finalIncurred = 1/finalIncurred

        finalPaid = intTable4.iloc[:,1]+intTable4.iloc[:,3]+intTable4.iloc[:,5]+intTable4.iloc[:,7]
        finalPaid = 1/finalPaid

        finalTable = pds.DataFrame({'Months':np.arange(1,31,1),
                                        'Final Incurred':finalIncurred,
                                        'Final Paid':finalPaid})
        return finalTable

#Protection2 starts here
def protection2():
        a = getRawData()
        
        nunTable1 = a.rawDataProtection2Table.drop([0,1])
        nunTable1 = nunTable1.iloc[:,1:]
        nunTable1 = nunTable1.astype(np.float64)

        def nunIntermediateTable1(x):
                return ((x-1)*a.logNormalizationProtection2Array)+1

        nunTable1 = np.apply_along_axis(nunIntermediateTable1,1,np.array(nunTable1))
        nunTable1 = pds.DataFrame(nunTable1)
        nunTable1.insert(loc=0, column='Months', value=np.arange(12,31,1))

        nunTable2 = pds.DataFrame(np.zeros([11,13]))
        nunTable2.iloc[:,0] = np.arange(1,12,1)

        firstRowNunTable2= nunTable1.iloc[0,1:]
        twelthRowNunTable2 = nunTable1.iloc[12,1:]

        def nunIntermediateTable2(x):
                global count2
                count2+=1
                
                return firstRowNunTable2**((np.log(firstRowNunTable2)/np.log(twelthRowNunTable2))**((12-count2)/(24-12)))*(12/count2)

        nunTable2 = pds.DataFrame(np.apply_along_axis(nunIntermediateTable2,1,np.array(nunTable2.iloc[:,1:])))
        nunTable2.insert(loc=0, column='Months', value=np.arange(1,12,1))

        nunTable3 = pds.concat([nunTable2,nunTable1])

        stateExposureArray=[a.stateExposureTable.iloc[1,3],a.stateExposureTable.iloc[1,3],
                        a.stateExposureTable.iloc[0,3],a.stateExposureTable.iloc[0,3],
                        a.stateExposureTable.iloc[3,3],a.stateExposureTable.iloc[3,3],
                        a.stateExposureTable.iloc[2,3],a.stateExposureTable.iloc[2,3],
                        a.stateExposureTable.iloc[4,3],a.stateExposureTable.iloc[4,3],
                        a.stateExposureTable.iloc[5,3],a.stateExposureTable.iloc[5,3]]

        nunTable4 =nunTable3.iloc[:,1:]

        def nunIntermediateTable4(x):
                return (stateExposureArray/x)*0.01

        nunTable4 = pds.DataFrame(np.apply_along_axis(nunIntermediateTable4,1,nunTable4))

        finalIncurred2 = nunTable4.iloc[:,0]+nunTable4.iloc[:,2]+nunTable4.iloc[:,4]+nunTable4.iloc[:,6]+nunTable4.iloc[:,8]+nunTable4.iloc[:,10]
        finalIncurred2 = 1/finalIncurred2

        finalPaid2 = nunTable4.iloc[:,1]+nunTable4.iloc[:,3]+nunTable4.iloc[:,5]+nunTable4.iloc[:,7]+nunTable4.iloc[:,9]+nunTable4.iloc[:,11]
        finalPaid2 = 1/finalPaid2

        finalTable2 = pds.DataFrame({'Months':np.arange(1,31,1),'Final Incurred':finalIncurred2,
                                        'Final Paid':finalPaid2})
        return finalTable2
