import os

import h5py
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils_read_data import read_data_into_dataframe
from globals import *
binarize_labels=False
# Output folder for plots
output_dir = 'figures/distribution_analysis'
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


# === 2D PCA VISUALIZATION ===
'''pca = PCA(n_components=2)
df_scaled=StandardScaler().fit_transform(df[feature_names])
X_pca = pca.fit_transform(df_scaled)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='classes', data=df.sample(5000), palette='Set2', alpha=0.6)
plt.title('2D PCA of Spectral Features')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_2d_visualization.png'))
plt.show()

plt.figure(figsize=(10, 10))
sns.pairplot(df[feature_names+['classes']].sample(5000), hue="classes")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pairplot.png'))
plt.show()

'''
sns.set_palette("Set2")
plt.figure(figsize=(10, 10))
sns.set(font_scale=2)
sns.pairplot(df[['spectra_0','spectra_1','spectra_8','spectra_9','spectra_10','spectra_11','spectra_12','classes']].sample(5000), hue="classes")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pairplot_selected_1.png'))
plt.show()
sns.set(font_scale=1)
exit()

# === NUMBER OF SAMPLES PER CLASS ===
plt.figure(figsize=(8, 5))
sns.countplot(x='classes', data=df, order=classes_names)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'class_distribution.png'))
plt.show()

# === HISTOGRAMS OF BAND VALUES PER CLASS ===
for band in feature_names:
    plt.figure(figsize=(10, 6))
    for label in classes_names:
        subset = df[df['classes'] == label][band]
        sns.histplot(subset, label=label)
    plt.title(f'Distribution of {band}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histogram_{band}.png'))
    plt.show()

# === CORRELATION HEATMAP ===
corr = df[feature_names].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Band Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'band_correlation_matrix.png'))
plt.show()

