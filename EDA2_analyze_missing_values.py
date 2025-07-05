import os

import matplotlib.pyplot as plt
import missingno as msno

from globals import *
from utils_read_data import read_data_into_dataframe

# Output folder for plots
output_dir = 'figures/missing_data_analysis'
os.makedirs(output_dir, exist_ok=True)


# Read data
df = read_data_into_dataframe(file_path, fields_of_interest)

print(df.describe())

plot_tasks = [
    (msno.matrix, 'missing_matrix'),
    (msno.bar, 'missing_bar_chart'),
    (msno.heatmap, 'missing_heatmap'),
    (msno.dendrogram, 'missing_dendrogram')
]

# Generate and save each plot
for plot_func, filename in plot_tasks:
    plt.figure()
    plot_func(df)
    plt.title(filename.replace('_', ' ').title())
    plt.savefig(os.path.join(output_dir, f'{filename}.png'))
    plt.close()

# Summary of missing values
missing_summary = df.isnull().sum().to_frame(name='Missing Count')
missing_summary['% Missing'] = (missing_summary['Missing Count'] / len(df)) * 100
print("\nMissing Value Summary:")
print(missing_summary)