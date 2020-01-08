##Import de tous les modules potentiellement utilisés
import xgboost
from pathlib import Path
import urllib
import pandas as pd
import os
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from numpy import *
import math
import matplotlib.pyplot as plt
from lib import functions

'''Define YOUR path here'''
#os.chdir("C:/Users/loren/Documents/Travail/Fac/Master 2/Biostat/projet")
#os.chdir("C:/Users/leobl/Desktop/ProjetBiostats3_LQLB")

##File openning and vizualisation
import gzip

data = gzip.open('data/data_projet.gz', 'rb')
data_contenu = data.read()

data_decoded = data_contenu.decode()
'''print(data_decoded)'''
type(data_decoded) ##str (string)


data_df = pd.read_csv('data/data_projet.gz', compression='gzip', header=0, sep=',')
'''print(data_df)'''
type(data_df)#dataframe

##We reshape the data

#We remove the last row (empty)
data_df = data_df.drop([1603])
'''print(data_df)'''
#we split the data in two dataframes, one containing the genes and the other containing the tumors
data_genes = data_df.iloc[0:801,:]
data_tumors = data_df.iloc[802:1603,:]
'''print(data_genes)
print(data_tumors)'''
#we remove the column 'gene_0'
data_genes = data_genes.drop("gene_0", axis=1)
'''print(data_genes)'''
#we extract the column cointaining the tumors to concatenate it with the data_genes object
tumors = data_tumors.iloc[:,1]
'''print(tumors)'''
#we merge tumors and data_genes
data_genes.index = pd.DataFrame(tumors).index
data_fusion = pd.concat([data_genes, pd.DataFrame(tumors)], axis=1, sort=False)
'''print(data_fusion)'''
##rename the gene_0 column
data_fusion = data_fusion.rename(columns={"gene_0":"tumors"})
data_fusion = data_fusion.reset_index(drop=True)
'''print(data_fusion)'''

##we extract the y
y = data_fusion.tumors
y.shape
print(y) 

##we extract the X
X = data_fusion.iloc[:,1:20531]
X.head()

#Calculation of standard deviation for each gene
std = np.std(X, axis=0)
print(std)

##Reduction of the number of features

#####1) Using satndard devaition only
#Filtering of genes: only the one with the higher standard deviation are kept with a more and more stringent threshold
genes_variation = std[std > 1]
print(genes_variation)##we have 9204 genes left but it is still higher than the number of individuals
genes_variation = std[std > 2]
print(genes_variation)#2261 genes left but it is still higher than the number of individuals

genes_variation1 = std[std > 3]
print(genes_variation1)#499 genes so below the number of individuals

##Extraction of X (499 selected genes) : test 1
gene_list1 = pd.DataFrame(genes_variation1)
gene_list1 = gene_list1.index
gene_list1 = pd.DataFrame(gene_list1)
gene_list1.index = gene_list1.iloc[:,0]
gene_list1.columns = ['a']
X_t1 = pd.merge(X.T, gene_list1, left_index=True, right_index=True)
del X_t1['a']
X_t1 = X_t1.T
print(X_t1)


genes_variation2 = std[std > 4]
print(genes_variation2)#109 genes

##Extraction of X (109 selected genes): test 2
gene_list2 = pd.DataFrame(genes_variation2)
gene_list2 = gene_list2.index
gene_list2 = pd.DataFrame(gene_list2)
gene_list2.index = gene_list2.iloc[:,0]
gene_list2.columns = ['a']
X_t2 = pd.merge(X.T, gene_list2, left_index=True, right_index=True)
del X_t2['a']
X_t2 = X_t2.T
print(X_t2)

######2) Making a PC analysis (we don't use it here)
from numpy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

#we remove the mean to each column so that all the features are on the same scale and none with large values will dominate the results.(standardisation)
X_0mean = X - X.mean(0)

## PCA generation
pca = PCA(100)
pca.fit(X)
X_transformed = pca.transform(X)
X_pca = pd.DataFrame(X_transformed)
X_pca.head()

##Percentage of explained variance for each PC
pca.explained_variance_ratio_

##We plot individuals using the two first PC

X_pca['tumors'] = y
X_pca.rows = ['PC1’,’PC2','PC3','PC4','tumors']
X_pca.head()
#####PCA end



##Encoding of tumors
y_encoded = list()
for i in y:
    if i == 'BRCA': y_encoded.append(0)
    if i == 'KIRC': y_encoded.append(1)
    if i == 'COAD': y_encoded.append(2)
    if i == 'LUAD': y_encoded.append(3)
    if i == 'PRAD': y_encoded.append(4)

print(y_encoded)

##conversion of y_encoded into categorical values
from keras.utils import to_categorical
y_cat = to_categorical(y_encoded)

print(y_cat)

##we split the data into training and testing datasets
from sklearn.model_selection import train_test_split
##For test1
x_train1, x_test1, y_train1, y_test1 = train_test_split(X_t1, y_cat, test_size=0.33, random_state=42 )
##For test2
x_train2, x_test2, y_train2, y_test2 = train_test_split(X_t2, y_cat, test_size=0.33, random_state=42 )

from keras.models import Model
from keras.layers import Dense, Input

##Weights' generation
init = 'random_uniform'


###We create one model for each dataset (test1 and 2)

#################test1

##Model definition

input_layer1 = Input(shape=(499,))
mid_layer1 = Dense(10, activation = 'relu', kernel_initializer = init)(input_layer1)
output_layer1 = Dense(5, activation= 'softmax', kernel_initializer = init)(mid_layer1)

##Model compilation
model1 = Model(input = input_layer1, output = output_layer1)
model1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()
##we use categorical crossentropy as a loss function because targets are one-hot encoded


model1.fit(x_train1, y_train1, batch_size=32, epochs=100, verbose=1)

##Model prediction
z_t1 = model1.predict(x_test1)

###Contingency table

##We convert y_test1 et z_test1 into binary values to be able to compare them
bz_t1 = np.argmax(z_t1, axis=1)
by_test1 = np.argmax(y_test1, axis=1)
##Contingency table
crosstab_t1 = pd.crosstab(by_test1, bz_t1)
print(crosstab_t1)
##Accuracy calculation
functions.precision(crosstab_t1)

##Learning curve

functions.learning_curve(x_train1, y_train1, x_test1, y_test1, model1)



#################test2

##Model generation

input_layer2 = Input(shape=(109,))
mid_layer2 = Dense(10, activation = 'relu', kernel_initializer = init)(input_layer2)
output_layer2 = Dense(5, activation= 'softmax', kernel_initializer = init)(mid_layer2)

##Model compilation
model2 = Model(input = input_layer2, output = output_layer2)
model2.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()
##we use categorical crossentropy as a loss function because targets are one-hot encoded


model2.fit(x_train2, y_train2, batch_size=32, epochs=100, verbose=1)

##Model prediction
z_t2 = model2.predict(x_test2)

###Contingency table

##We convert y_test1 et z_test1 into binary values
bz_t2 = np.argmax(z_t2, axis=1)
by_test2 = np.argmax(y_test2, axis=1)
##Contingency table
crosstab_t2 = pd.crosstab(by_test2, bz_t1)
print(crosstab_t2)
##Accuracy calculation
functions.precision(crosstab_t2)

##Learning curve
functions.learning_curve(x_train2, y_train2, x_test2, y_test2, model2)
