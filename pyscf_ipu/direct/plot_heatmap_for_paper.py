import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-data_file',   type=str)
parser.add_argument('-output_name',   type=str, default="default_output.png")
parser.add_argument('-log',   type=bool, default=False)
opts = parser.parse_args()

# Load data from the pickle file
with open(opts.data_file, 'rb') as file:
    data_list = pickle.load(file)


# Extract phi, psi, and values from the loaded data
phi_values, psi_values, heatmap_values = zip(*data_list)

if opts.log:
    heatmap_values = np.log(np.abs(heatmap_values - np.mean(heatmap_values)))

print(heatmap_values)
# Create a meshgrid of phi and psi coordinates
phi_coordinates, psi_coordinates = np.meshgrid(np.linspace(min(phi_values), max(phi_values), 100),
                                               np.linspace(min(psi_values), max(psi_values), 100))

# Interpolate values on the grid
heatmap_interpolated = griddata((phi_values, psi_values), heatmap_values, (phi_coordinates, psi_coordinates), method='cubic', fill_value=0)

# Display the 2D matrix as an image
plt.imshow(heatmap_interpolated, cmap='viridis', origin='lower', extent=[min(psi_values), max(psi_values), min(phi_values), max(phi_values)])
plt.colorbar(label='Intensity')  # Add colorbar with label

# Set axis labels and title
plt.xlabel('Psi Angle')
plt.ylabel('Phi Angle')
plt.title('2D Heatmap from Pickle File')

# Save the plot to a PNG file
plt.savefig(opts.output_name)

# Show the plot
