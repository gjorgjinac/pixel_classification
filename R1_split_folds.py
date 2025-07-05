import pandas as pd

from  utils.utils_read_data import read_data_into_dataframe
from globals import *
from sklearn.model_selection import GroupKFold

df = read_data_into_dataframe(file_path, fields_of_interest)

groups = df['product_id']
lpgo = GroupKFold(n_splits=10)

project_test_ids_per_fold=[]
for fold, (train_index, test_index) in enumerate(lpgo.split(df, df, groups)):
    test_product_ids = list(set(groups.iloc[test_index].astype(str)))
    project_test_ids_per_fold+=[(fold,i) for i in test_product_ids]
pd.DataFrame(project_test_ids_per_fold, columns=['fold','product_id']).to_csv('data/project_test_ids_per_fold.csv', index=False)
