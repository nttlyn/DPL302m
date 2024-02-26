import os
import pandas as pd
from sklearn.model_selection import train_test_split

folder_path = 'D:\\SP24\\DPL302m\\PROJECT\\DPL302m_Group6_Helmets_Detection\\Label\\vehicle_2_labeled'

files = os.listdir(folder_path)

dfs = []

for file in files:
    if file.endswith('.txt'): 
        file_path = os.path.join(folder_path, file)  
        df = pd.read_csv(file_path, sep='\t')  
        dfs.append(df)  

full_dataset = pd.concat(dfs)

num_samples = len(full_dataset)
split_index = int(0.8 * num_samples)

train_data = full_dataset.iloc[:split_index]
test_data = full_dataset.iloc[split_index:]


train_file_path = 'D:\\SP24\\DPL302m\\PROJECT\\DPL302m_Group6_Helmets_Detection\\Dataset\\train_dt.csv'
test_file_path = 'D:\\SP24\\DPL302m\\PROJECT\\DPL302m_Group6_Helmets_Detection\\Dataset\\test_dt.csv'


train_data.to_csv(train_file_path, index=False)
test_data.to_csv(test_file_path, index=False)

print("Number of samples:", num_samples)
print("Split index:", split_index)

print("First few rows of training data:")
print(train_data.head())

print("First few rows of testing data:")
print(test_data.head())
