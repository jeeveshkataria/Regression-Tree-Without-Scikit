#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:00:24 2020

@author: jeevesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:47:36 2020

@author: jeevesh
"""
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy import stats
import seaborn as sns



class Node:
  def __init__(self):
    self.left = None
    self.right = None
    self.leaf = 0
    self.splitting_value = None
    self.col  = None


class DecisionTree:
    def __init__(self):
        self.training_data_dataFrame = pd.DataFrame()
        self.test_data_dataFrame = pd.DataFrame()
        self.dataFrameForClean = pd.DataFrame()
        self.column_names = []
        self.drop_columns = []
        self.categorial_columns=[]
        self.test_data_categorial_columns=[]
        self.numerical_columns=[]
        self.test_data_numerical_columns=[]
        self.predictions=[]
        self.root1=None
        #def get_msr_for_categorical(col, dataFrame):
        
        
        
        
        
        
    def train(self,training_data):
        self.training_data_dataFrame = pd.read_csv(training_data)
        self.dataFrameForClean = self.training_data_dataFrame.copy()
        self.column_names = self.dataFrameForClean.columns
        #drop_columns=[]
        for i in self. column_names:
            if self.dataFrameForClean[i].isnull().sum() > 550:
                self.drop_columns.append(i)
                
        self.dataFrameForClean.drop(self.drop_columns,axis=1,inplace=True)
        
        self.categorial_columns=self.dataFrameForClean.dtypes[self.dataFrameForClean.dtypes == "object"].index
        
        self.numerical_columns=self.dataFrameForClean.dtypes[self.dataFrameForClean.dtypes != "object"].index
        
        self.dataFrameForClean.replace('None',np.nan,inplace=True)

        for col in self.categorial_columns:
            self.dataFrameForClean.fillna(self.dataFrameForClean.mode().iloc[0],inplace=True)
            
        for col in self.numerical_columns:
            self.dataFrameForClean.fillna(self.dataFrameForClean.mean(),inplace=True)
            
        self.root1 = self.build_tree(self.dataFrameForClean,6,0,0)  
        #self.printInorder(self.root1,0)
      
           
    def get_msr_for_categorical(self,col, dataFrame):
        uniqueValues = dataFrame[col].unique()
        col_specific_value=None
        col_value=0
        prev_msr=math.inf
        for feature in uniqueValues:
            same=self.dataFrameForClean[self.dataFrameForClean[col]==feature]
            large=self.dataFrameForClean[self.dataFrameForClean[col]!=feature]
            
            same = same.to_numpy()
            large = large.to_numpy()
            A=same[:,[-1]]
            B=large[:,[-1]]
    
            msr1=0
            msr2=0
            msr1=((A-np.mean(A))**2)
            msr2=((B-np.mean(B))**2)
    

    

            total = same.shape[0] + large.shape[0]
            wmsr = int((sum(msr1)*len(same) + sum(msr2)*len(large))/total)
            
    
            if((wmsr)<prev_msr):
                prev_msr = wmsr
                col_specific_value = feature

          
        return prev_msr , col_specific_value 
        
    
    
    def get_msr_for_int_mid_value(self,mid_value,col,data_frame):
        small_values_dataFrame = data_frame[data_frame[col]<mid_value]
        
        large_value_dataFrame = data_frame[data_frame[col]>=mid_value]
        
        large_value_dataFrame=large_value_dataFrame.to_numpy()
        
        small_values_dataFrame=small_values_dataFrame.to_numpy()
        
        B=large_value_dataFrame[:,[-1]]
        
        A=small_values_dataFrame[:,[-1]]
  
        msr1=0
        msr2=0


  
        msr1=((A-np.mean(A))**2)
        msr2=((B-np.mean(B))**2)
    

    
        total = small_values_dataFrame.shape[0] + large_value_dataFrame.shape[0]
        wmsr = int((sum(msr1)*len(small_values_dataFrame) + sum(msr2)*len(large_value_dataFrame))/total)
    
        return (wmsr);
    
    
    
    def get_msr_for_int(self,col,data_frame):
        uniqueValues = data_frame[col].unique()
  
        uniqueValues.sort()

        prev_msr = math.inf
  
        prev_mean =None
  
 
        for i in range(len(uniqueValues)-1):
            (mid_value) = float(uniqueValues[i] + uniqueValues[i+1] ) /2

            curr_msr = self.get_msr_for_int_mid_value(mid_value,col,data_frame)

    
            if curr_msr < prev_msr:
                prev_msr = curr_msr
                prev_mean = mid_value

  
        return prev_msr , prev_mean , col
    
    
    def get_col_to_split(self,dataFrame):
        prev_msr = math.inf
        value=None
        col_value=None
  
        for col in dataFrame.columns:
            if col == "SalePrice" or col == "Id":
                continue
    
            if(dataFrame[col].dtype == "object"):
                curr_msr , mean_value = self.get_msr_for_categorical(col,dataFrame)
            else:
                curr_msr , mean_value ,col = self.get_msr_for_int(col,dataFrame)
      
            if(curr_msr < prev_msr):
                prev_msr = curr_msr
                value = mean_value
                col_value = col
      
        return col_value , value
    
    def get_data_frame(self,training_data_build,splitting_value,col):
        if self.dataFrameForClean[col].dtype == "object":
            mask = training_data_build[col] == splitting_value
            training_data_small = training_data_build[mask]
            training_data_large = training_data_build[~mask]
            #print('training_data_small',training_data_small.shape[0],'training_data_large',training_data_large.shape[0])
            return training_data_small , training_data_large
        else:
            mask = training_data_build[col] < splitting_value
            training_data_small = training_data_build[mask]
            training_data_large = training_data_build[~mask]
            #print('training_data_small',training_data_small.shape[0],'training_data_large',training_data_large.shape[0])
            return training_data_small , training_data_large
        
        
        
        
    def build_tree(self,training_data_build, max_depth, min_size,depth):
        root = Node()
        
        if depth > max_depth:
            splitting_value=training_data_build['SalePrice'].mean();
            root.leaf = 1
            root.splitting_value = splitting_value;
            # print('depth',splitting_value)
            # print('@'*30)
            # print(training_data_build.shape[0])
            # print('@'*30)
            return root
        
        if training_data_build.shape[0] < 20:
            splitting_value=training_data_build['SalePrice'].mean();
            root.leaf = True
            root.splitting_value = splitting_value;
            # print('size',splitting_value)
            # print('@'*30)
            # print(training_data_build.shape[0])
            # print('@'*30)
            return root
        
        col , splitting_value = self.get_col_to_split(training_data_build)
        training_data_small , training_data_large  = self.get_data_frame(training_data_build,splitting_value,col)
          #print('splitting',splitting_value,' col:',col)
          #print('small',training_data_small.shape[0])
          #print('large',training_data_large.shape[0])
        del training_data_small[col]
        del training_data_large[col]

          #print(splitting_value)
  
  
  
        root.splitting_value = splitting_value
        root.col = col
        # print(splitting_value,'-',col,'-',depth)
  
        root.left = self.build_tree(training_data_small,max_depth,min_size,depth + 1)

        root.right =self.build_tree(training_data_large,max_depth,min_size,depth +1)
        #print('root.spli:',root.splitting_value,':root:',root.col,':depth:',depth,'leaf:',root.left)
        return root
    
    
    def printInorder(self,root2,depth):
        if root2:
            print(root2.splitting_value,root2.col,root2.leaf,depth)
            self.printInorder(root2.left,depth+1)
            self.printInorder(root2.right,depth+1)
    
   
    
    def find_ans(self,row,root1):
        
        if root1.left==None and root1.right==None:
            #print('leaf',root1.splitting_value)
            return root1.splitting_value
        #print('root1.col',root1.col,dataFrameForClean[root1.col].dtypes)
        # print ( 'compare between:', dataFrameForClean[root1.col] , '!!!!!!', row[root1.col],' ==', root1.splitting_value)
        if(self.dataFrameForClean[root1.col].dtypes == "object"):
            if(row[root1.col] == root1.splitting_value):
                #print('going left',root1.splitting_value)
                return self.find_ans(row,root1.left)
            else:
                #print('going right',root1.splitting_value)
                return self.find_ans(row,root1.right)
        else:
            if(row[root1.col] < root1.splitting_value):
                #print('g t left',root1.splitting_value)
                return self.find_ans(row,root1.left)
            else:
                #print('g to right',root1.splitting_value)
                return self.find_ans(row,root1.right)

   
    









    
        
    def predict(self,testing_data):
        self.test_data_dataFrame = pd.read_csv(testing_data)
        
        self.test_data_dataFrame.drop(self.drop_columns,axis=1,inplace=True)
        
        self.test_data_categorial_columns=self.test_data_dataFrame.dtypes[self.test_data_dataFrame.dtypes == "object"].index
        
        
        self.test_data_numerical_columns=self.test_data_dataFrame.dtypes[self.test_data_dataFrame.dtypes != "object"].index
        
        self.test_data_dataFrame.replace('None',np.nan,inplace = True)

        
        for col in self.test_data_categorial_columns:
            self.test_data_dataFrame.fillna(self.test_data_dataFrame.mode().iloc[0],inplace=True)

        
        for col in self.test_data_numerical_columns:
            self.test_data_dataFrame.fillna(self.test_data_dataFrame.mean(),inplace=True)
            
            
            
        
        

        for i in range(0,len(self.test_data_dataFrame)):
            self.predictions.append(self.find_ans(self.test_data_dataFrame.iloc[i],self.root1))
            
        return self.predictions
    
    

#dtree_regressor = DecisionTree()

#dtree_regressor.train('/home/jeevesh/Desktop/smai/assignment_1/q3/train.csv')

#print(dtree_regressor.p#rintInorder(0))
#predictions = dtree_regressor.predict('/home/jeevesh/Desktop/smai/assignment_1/q3/test.csv')
#test_labels = list()

#with open("/home/jeevesh/Desktop/smai/assignment_1/q3/test_labels.csv") as f:
#    for line in f:
#        test_labels.append(float(line.split(',')[1]))
        
#print (mean_squared_error(test_labels, predictions))              
          
    

