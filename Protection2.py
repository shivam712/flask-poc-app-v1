import pandas as pds
import numpy as np
import math

global count
count=0

pds.set_option("display.precision", 10)

class getRawData:
    
        def __init__(self):         
            self.stateExposureTable = self.getStateExposureTable()
            self.vehicleClassTable = self.getVehicleClassTable()
            self.stateCountTable = self.getStateCountTable()
            self.vehicleTypeWeightTable = self.getVehicleTypeWeightTable()  
            self.rawDataProtection2Table = self.getRawDataProtection2()
            self.constantValuesProtection2Table = self.getConstantValuesProtection2()
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

        def getRawDataProtection2(self):
            file=('Data Set/raw_data_protection2.csv')
            return pds.DataFrame(pds.read_csv(file)) 

        def getConstantValuesProtection2(self):
            file=('Data Set/values_protection2_data.csv')
            return pds.DataFrame(pds.read_csv(file)) 

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

class tablesProtection2:

    def __init__(self):
        self.data = getRawData()
        self.intermediateTable1()
        self.intermediateTable2()
        self.intermediateTable3()
        self.intermediateTable4()
        self.finalTableProtection2()
   
    def intermediateTable1(self):
        self.intTable1 = self.data.rawDataProtection2Table.drop([0,1])
        self.intTable1 = self.intTable1.iloc[:,1:]
        self.intTable1 = self.intTable1.astype(np.float64)
        self.intTable1 = np.apply_along_axis(self.funIntermediateTable1,1,np.array(self.intTable1))
        self.intTable1 = pds.DataFrame(self.intTable1)
        self.intTable1.insert(loc=0, column='Months', value=np.arange(12,31,1))

    def funIntermediateTable1(self,x):
        return ((x-1)*self.data.logNormalizationProtection2Array)+1

    def intermediateTable2(self):
        self.intTable2 = pds.DataFrame(np.zeros([11,13]))
        self.intTable2.iloc[:,0] = np.arange(1,12,1)

        self.firstRowIntTable2= self.intTable1.iloc[0,1:]
        self.twelthRowIntTable2 = self.intTable1.iloc[12,1:]
        self.intTable2 = pds.DataFrame(np.apply_along_axis(self.funIntermediateTable2,1,np.array(self.intTable2.iloc[:,1:])))
        self.intTable2.insert(loc=0, column='Months', value=np.arange(1,12,1))

    def funIntermediateTable2(self,x):
        global count
        count+=1        
        return self.firstRowIntTable2**((np.log(self.firstRowIntTable2)/np.log(self.twelthRowIntTable2))**((12-count)/(24-12)))*(12/count)

    def intermediateTable3(self):
        self.intTable3 = pds.concat([self.intTable2,self.intTable1])

    def intermediateTable4(self):
        self.stateExposureArray=[self.data.stateExposureTable.iloc[1,3],self.data.stateExposureTable.iloc[1,3],
                            self.data.stateExposureTable.iloc[0,3],self.data.stateExposureTable.iloc[0,3],
                            self.data.stateExposureTable.iloc[3,3],self.data.stateExposureTable.iloc[3,3],
                            self.data.stateExposureTable.iloc[2,3],self.data.stateExposureTable.iloc[2,3],
                            self.data.stateExposureTable.iloc[4,3],self.data.stateExposureTable.iloc[4,3],
                            self.data.stateExposureTable.iloc[5,3],self.data.stateExposureTable.iloc[5,3]]
        self.intTable4 = self.intTable3.iloc[:,1:]
        self.intTable4 = pds.DataFrame(np.apply_along_axis(self.funIntermediateTable4,1,self.intTable4))

    def funIntermediateTable4(self,x):
        return (self.stateExposureArray/x)*0.01

    def finalTableProtection2(self):
        self.finalIncurred = self.intTable4.iloc[:,0]+self.intTable4.iloc[:,2]+self.intTable4.iloc[:,4]+self.intTable4.iloc[:,6]+self.intTable4.iloc[:,8]+self.intTable4.iloc[:,10]
        self.finalIncurred = 1/self.finalIncurred

        self.finalPaid = self.intTable4.iloc[:,1]+self.intTable4.iloc[:,3]+self.intTable4.iloc[:,5]+self.intTable4.iloc[:,7]+self.intTable4.iloc[:,9]+self.intTable4.iloc[:,11]
        self.finalPaid = 1/self.finalPaid

        self.finalTable = pds.DataFrame({'Months':np.arange(1,31,1),
                                        'Final Incurred':self.finalIncurred,
                                        'Final Paid':self.finalPaid})

        return self.finalTable
