import h5py
import numpy as np
import pandas as pd
from scipy import io
from typing import List
from pathlib import Path
from scipy.io import loadmat
import hdf5storage

def mat_to_df(p: Path, ix_deep: bool,
              extra_cols: List = [], ix: str = 'FILE_ID') -> pd.DataFrame:
    """ This is a function specifically written for Matlab files
        in our lab with a certain structure that are not able 
        to be read using scipy.io.loadmat

    Args:
        p (Path): The path at which the Matlab file is located
        ix_deep (bool): Whether the indexing is more levels than for
            most of the other structures
        extra_cols (List): The extra columns in the Matlab File
            we specifically want in the DataFrame. Defaults to [].
        ix (str): The index into the dictionary representing Matlab data
            to obtain the info as an array. Defaults to 'FILE_ID'.

    Returns:
        pd.DataFrame: The Matlab data as a pandas DataFrame with subjects as
            rows, and information about the subjects as columns
    """
    # Some versions of Matlab do not work with scipy.io.loadmat
    try:
        mat = io.loadmat(p)
    except NotImplementedError:
        mat = hdf5storage.loadmat(str(p))

    # We need to use deeper indexing for some of the structures
    # (one extra zero-indexing).
    # This code obtains all the info columns in the Matlab structure
    if ix_deep:
        columns_info = [mat[ix][0][i][0][0] for i in range(mat[ix][0].shape[0])]
    else:
        columns_info = [mat[ix][0][i][0] for i in range(mat[ix][0].shape[0])]

    # There may be some additional columns apart from 'FILE_ID
    # in the overall Matlab structure
    other_columns = list(mat.keys())

    # The 'sub_ID' column is removed in case it is there,
    # but its dtypes are added to important columns.
    # The dtypes of this array are names of columns with information
    # about the subjects
    if 'sub_ID' in mat.keys():
        other_columns.remove('sub_ID')
        file_columns = list(mat['sub_ID'].dtype.names)
    else:
        file_columns = []

    # All the columns we need added together into one big list
    columns = columns_info + other_columns + file_columns + extra_cols

    # Obtain the number of subjects in the Matlab data object
    num_subjects = mat['analysis_SCORE'].shape[0]

    # Allocate a DataFrame to store the data in
    arr = np.zeros((num_subjects, len(columns)))
    df = pd.DataFrame(arr, columns=columns)

    # The info columns correspond to the data under the
    # key: 'analysis_SCORE' in the Matlab file
    df[columns_info] = mat['analysis_SCORE']

    # Recover the data under the 'analysis_ID' key in the 
    # Matlab data object
    if ix_deep:
        df['analysis_ID'] = np.array([mat['analysis_ID'][i][0][0][0] for i in range(num_subjects)])        
    else:
        df['analysis_ID'] = np.array([mat['analysis_ID'][i][0][0] for i in range(num_subjects)])
    
    # Store the 'analysis_idx' information in the DataFrame
    if 'analysis_idx' in mat.keys():
        df['analysis_idx'] = np.array([mat['analysis_idx'][i][0] for i in range(num_subjects)])
    if 'exist_idx' in mat.keys():
        df['exist_idx'] = np.array([mat['exist_idx'][i][0] for i in range(num_subjects)])

    # Add all the information under the 'sub_ID' to the
    # DataFrame if 'sub_ID' exists
    for (i, file_column) in enumerate(file_columns):
        if ix_deep:
            df[file_column] = np.array([mat['sub_ID'][j][0][i][0][0] for j in range(num_subjects)])
        else:
            df[file_column] = np.array([mat['sub_ID'][j][0][i][0] for j in range(num_subjects)])
    
    # Add all the information in the extra columns to the DataFrame
    for extra_col in extra_cols:
        if ix_deep:
            df[extra_col] = np.array([mat[extra_col][i][0][0][0] for i in range(num_subjects)])
        else:
            df[extra_col] = np.array([mat[extra_col][i][0][0] for i in range(num_subjects)])

    # Remove duplicate columns
    df = df.loc[:,~df.columns.duplicated()]

    # Return final DataFrame
    return df

fbirn_data_path = Path('/data/users2/zfu/Matlab/GSU/Neuromark/Results/ICA/FBIRN')
fbirn_info_path = Path('/data/users2/zfu/Matlab/GSU/Neuromark/Results/Subject_selection/FBIRN/sub_info_FBIRN.mat')

info_df = mat_to_df(fbirn_info_path, ix_deep=True).reset_index()
info_df['sz'] = info_df['diagnosis(1:sz; 2:hc)']
info_df['sex'] = info_df['gender(1:male; 2:female)']

files = []
for index in info_df['index']:
        comp_path = fbirn_data_path / f'FBIRN_sub{str(index+1).zfill(3)}_timecourses_ica_s1_.nii'
        print(comp_path)
        if comp_path.is_file():
                files.append(comp_path)
        else:
                print('something is wrong')
                quit()

info_df['path'] = files
info_df.to_csv('data/ica_fbirn/info_df.csv')