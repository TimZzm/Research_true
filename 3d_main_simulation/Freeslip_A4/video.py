import matplotlib.animation as Animation
from PIL import Image
import os
from os import listdir
import matplotlib

import matplotlib.pyplot as plt
from dedalus.extras import plot_tools


folder_dir = 'liquid_water'

# Get a list of PNG file paths from the directory
file_paths = [os.path.join(folder_dir, f) for f in listdir(folder_dir) if os.path.isfile(os.path.join(folder_dir, f)) and f.endswith('.png')]

# Sort the list of file paths
file_paths.sort()

# Read the images using PIL
imgs = [Image.open(f) for f in file_paths]

fig = plt.figure(figsize=(16,4),dpi=200)
fig.patch.set_visible(False)
plt.axis('off')

# Wrap each image in a list to create a list of sequences of artists
imgs = [[plt.imshow(img, animated=True)] for img in imgs]

ani = Animation.ArtistAnimation(fig, imgs, interval=250, blit=True, repeat_delay=1000)

# Save the animation to a file
ani.save(f'liquidwater_A9.mp4',dpi=200)