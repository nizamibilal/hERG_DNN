import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit import DataStructs
import os
from os.path import exists
from tqdm.notebook import tqdm, trange ## for progress bar display


## Calculate morgan FP
def get_morgan(data, smiles, nBits=2048):

    '''
    Calculate Morgan FP of a given pandas dataframe. Strucutre as SMILES string should be present as a column in the dataframe. 

    Parameters: 
        data = input data (pands dataframe)
        smiles = Column name with SMILES
        nBits = number of bits (Morgan FP) default 2048

    Return:
        dataframe with morgan fingerprint of size (rows_in_data X nBits) 

    '''
    
    #nBits = 2048 # number of Bits for morgan fingerprint
    ##Generate columns name for dataframe to store morgan fingerprint
    col_name = []
    for bits in range(1, nBits+1):
        #name = 'ECFP'+str(bits)
        name = f'ECFP{str(bits)}'
        col_name.append(name)
    #print(col_name)
    
    ## Dataframe to store morgan fingerprint
    df_fp = pd.DataFrame(columns=col_name)
   
    ## loop over each molecule
    for ind in trange(len(data)):

        ## RDKit mol from smiles
        m = Chem.MolFromSmiles(data[smiles][ind])

        ## Morgan fignerprint of diameter = 2 which is quivalent to ECFP4
        fp = AllChem.GetMorganFingerprintAsBitVect(m,2, nBits=nBits)

        ## Concatenate the fp to dataframe
        if df_fp.empty:
            df_fp = pd.DataFrame(np.array(fp)).T
        else:
            df_fp = pd.concat([df_fp, pd.DataFrame(np.array(fp)).T])
        #print(df_evotec_new['smiles'][ind])

    return df_fp
    
## Calculate morgan FP
def get_morgan_ROMol(data, ROMol, nBits=2048):

    '''
    Calculate Morgan FP of a given pandas dataframe. Strucutre as RDKit MOL should be present as a column in the dataframe. 

    Parameters: 
        data = input data (pands dataframe)
        ROMol = Column name with RDKit MOL
        nBits = number of bits (Morgan FP) default 2048

    Return:
        dataframe with morgan fingerprint of size (rows_in_data X nBits) 

    '''
    
    #nBits = 2048 # number of Bits for morgan fingerprint
    ##Generate columns name for dataframe to store morgan fingerprint
    col_name = []
    for bits in range(1, nBits+1):
        #name = 'ECFP'+str(bits)
        name = f"ECFP{str(bits)}"
        col_name.append(name)
    #print(col_name)
    
    ## Dataframe to store morgan fingerprint
    df_fp = pd.DataFrame(columns=col_name)
   
    ## loop over each molecule
    for ind in trange(len(data)):

        ## RDKit mol
        m = data[ROMol][ind]

        ## Morgan fignerprint of diameter = 2 which is quivalent to ECFP4
        fp = AllChem.GetMorganFingerprintAsBitVect(m,2, nBits=nBits)

        ## Concatenate the fp to dataframe
        if df_fp.empty:
            df_fp = pd.DataFrame(np.array(fp)).T
        else:
            df_fp = pd.concat([df_fp, pd.DataFrame(np.array(fp)).T])
        #print(df_evotec_new['smiles'][ind])
    return df_fp


## get performance
def get_performance(cv_res):
    '''
    Function: Get the metrics from a Cross Valiation Search cv_results_

    Input:
        cv_res = cv_results_

    Returns: Series with metirics 'mean_test_AUC', 'mean_test_Accuracy', 'mean_test_BalancedAccuracy', 'mean_test_MCC', 'mean_test_Recall', 'mean_test_Precision'
    
    '''
    df = pd.DataFrame(cv_res)
    r = df.sort_values(by='rank_test_AUC').iloc[0]
    return r[['mean_test_AUC', 'mean_test_Accuracy', 'mean_test_BalancedAccuracy', 'mean_test_MCC', 'mean_test_Recall', 'mean_test_Precision']]