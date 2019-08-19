import pandas as pd
import numpy as np

def load_adni(folder='./pickled_adni', normed=True, task='AD_vs_Normal',
              return_matrix=False, return_spreadsheet=False):
    '''Load ADNI dataset
    
    Parameters
    ---
    
    folder: str,
        path to data folder
        
    normed: bool,
        whether to divide each matrix by its sum or not
        
    task: str,
        desirable task, possible values are:
        'AD_vs_EMCI',   'AD_vs_LMCI',     'AD_vs_Normal', 'AD_vs_SMC',
        'EMCI_vs_LMCI', 'EMCI_vs_Normal', 'EMCI_vs_SMC',  'LMCI_vs_Normal',
        'LMCI_vs_SMC',  'Normal_vs_SMC',  'All_vs_Normal','AD_vs_All'
        
    return_matrix: bool,
        if True returns array of size (n_subjects x 68 x 68), 
        if False return array of size (n_subjects x n_edges) (no self loops)
    
    return_spreadsheet: bool,
        whether to return meta data DataFrame
    '''
    
    # load data and meta data
    mats = np.load(f'{folder}/adni', allow_pickle=True)
    meta_data = pd.read_csv(f'{folder}/task_target.csv',
                            index_col=0, na_values=np.nan)

    # preprocess: fill diagonals, normalize
    for m in mats:
        np.fill_diagonal(m, 0)
            
    if normed:
        normed_mats = []
        for m in mats:
            normed_mats.append(m / m.sum())
        mats = np.array(normed_mats)
        
    # bag of edges
    if not return_matrix:
        rows, cols = np.triu_indices(68, k=1)
        boe = np.array([mat[rows, cols] for mat in mats])
        mats = boe
    
    # Select task target
    task_index = meta_data[['subj_id', 'target', task]].dropna().index.values
    mats = mats[task_index, :]
    target = meta_data[task].dropna().values
    subj_id = meta_data[['subj_id', task]].dropna()['subj_id'].values
    
    if return_spreadsheet:
        return mats, target, subj_id, meta_data[['subj_id', 'target', task]].dropna()
    
    return mats, target, subj_id

def load_thickness():
    # load data
    print('Load thickness data...\n')
    meta = pd.read_excel('../miccai2018-disease-progression/all/adniMERGE_final_for_GREG_10-20-2014_with_category_names.xlsx')

    adni1_csv_cs = pd.read_csv('../miccai2018-disease-progression/ADNI_Anat_measures/ADNI1/CorticalMeasuresENIGMA_ThickAvg_CROSS_ADNI1_sc.csv')
    adni1_csv_12 = pd.read_csv('../miccai2018-disease-progression/ADNI_Anat_measures/ADNI1/CorticalMeasuresENIGMA_ThickAvg_CROSS_ADNI1_12mo.csv')
    adni1_csv_24 = pd.read_csv('../miccai2018-disease-progression/ADNI_Anat_measures/ADNI1/CorticalMeasuresENIGMA_ThickAvg_CROSS_ADNI1_24mo.csv')


    adni2_csv_cs = pd.read_csv('../miccai2018-disease-progression/ADNI_Anat_measures/ADNI2/CorticalMeasuresENIGMA_ThickAvg_CROSS_ADNI2_sc.csv')
    adni2_csv_12 = pd.read_csv('../miccai2018-disease-progression/ADNI_Anat_measures/ADNI2/CorticalMeasuresENIGMA_ThickAvg_CROSS_ADNI2_12mo.csv')
    adni2_csv_24 = pd.read_csv('../miccai2018-disease-progression/ADNI_Anat_measures/ADNI2/CorticalMeasuresENIGMA_ThickAvg_CROSS_ADNI2_24mo.csv')

    adni1_csv = pd.concat([adni1_csv_cs,adni1_csv_12,adni1_csv_24], axis=0)
    adni1_csv['subj_id'] = adni1_csv['SubjID'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    adni2_csv = pd.concat([adni2_csv_cs,adni2_csv_12,adni2_csv_24], axis=0)
    adni2_csv['subj_id'] = adni2_csv['SubjID'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    # AGE, PTGENDER

    # concat ADNI1 and ADNI2, drop NA observations
    print('Combine ADNI1 and ADNI2 thickness...\n')
    adni_thick = pd.concat([adni1_csv, adni2_csv], axis=0)
    mmap = dict(zip(meta['SUBJECT_ID'].values, meta['DX_bl'].values))
    adni_thick['DX'] = adni_thick['subj_id'].apply(lambda x: mmap.get(x))
    mmap = dict(zip(meta['SUBJECT_ID'].values, meta['AGE'].values))
    adni_thick['age'] = adni_thick['subj_id'].apply(lambda x: mmap.get(x))
    mmap = dict(zip(meta['SUBJECT_ID'].values, meta['PTGENDER'].values))
    adni_thick['gender'] = adni_thick['subj_id'].apply(lambda x: mmap.get(x))
    adni_thick.dropna(inplace=True)
    adni_thick.replace(0, np.nan, inplace=True)
    adni_thick.dropna(inplace=True)

    print(f'Total #observations ADNI1 and ADNI2: {adni_thick.shape}')
    print(f'Including ADNI1 baseline visits: {adni1_csv_cs.shape[0]}, \
    1 year visits: {adni1_csv_12.shape[0]}, 2 year visits: {adni1_csv_24.shape[0]}')
    print(f'Including ADNI2 baseline visits: {adni2_csv_cs.shape[0]}, \
    1 year visits: {adni2_csv_12.shape[0]}, 2 year visits: {adni2_csv_24.shape[0]}')
    print(np.unique(adni_thick['DX'].values, return_counts=True), '\n')

    # Take only AD vs NC observations
    print('Take subset of AD and NC subjects...')
    ids = np.where((adni_thick['DX'] == 'AD') | (adni_thick['DX'] == 'CN'))[0]
    vals, counts = np.unique(adni_thick.iloc[ids]['subj_id'].values, return_counts=True)
    print('Unique SUBJECTS: ', vals.shape)
    print('#of observations from subjects: ', np.unique(counts, return_counts=True))
    data = adni_thick.iloc[ids]
    print(f'AD, NC table size: {data.shape}\n', )
    # print(data.drop_duplicates('subj_id')['DX'].value_counts())



    print('Take only baseline visit...')
    X = data.drop_duplicates('subj_id').drop(axis=1, columns=['SubjID', 'LThickness', 'RThickness', 'LSurfArea',
                                   'RSurfArea', 'ICV', 'subj_id', 'DX']).values

    y = np.where(data.drop_duplicates('subj_id')['DX'].values=='AD', 1, 0)
    colnames = data.drop_duplicates('subj_id').drop(axis=1, columns=['SubjID', 'LThickness', 'RThickness', 'LSurfArea',
                                          'RSurfArea', 'ICV', 'subj_id', 'DX']).columns

    print(np.unique(y, return_counts=True))
    print(X.shape,'\n')
    
    return X, y, colnames, data