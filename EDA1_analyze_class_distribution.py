import h5py
import os
from utils.utils_visualization import *
from utils.utils_read_data import read_data_into_dataframe
from globals import *
binarize_labels=False
# Output folder for plots
output_dir = 'figures/distribution_analysis_test'
if binarize_labels:
    output_dir+='_binary'
os.makedirs(output_dir, exist_ok=True)


with h5py.File(file_path, 'r') as f:
    class_id_to_name={int(k): v.decode('utf-8') for k, v in zip(f['class_ids'][()], f['class_names'][()])}
    print(class_id_to_name)

if binarize_labels:
    class_id_to_name = {k: 'Other' if v != 'Cloud' else v for k, v in class_id_to_name.items()}

# Read data
df = read_data_into_dataframe(file_path, fields_of_interest)
df['classes'] = df['classes'].map(class_id_to_name)
categorical_features = df.select_dtypes(exclude=['number']).columns.tolist()
classes_names=df['classes'].unique()
feature_names=list(filter(lambda x: x.startswith('spectra_'), df.columns))

# 2D PCA visualization
generate_2d_pca_visualization(df, feature_names, output_dir)

# Pairplot of all features
generate_pairplot(df,feature_names,output_dir)

# Number of samples per class
generate_class_distribution(df,classes_names,output_dir)

# Histogram of band values per class
generate_feature_histogram(df,feature_names,classes_names,output_dir)

# Correlation heatmap
generate_correlation_heatmap(df, feature_names, output_dir)

