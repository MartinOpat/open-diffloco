import numpy as np
from PIL import Image
import noise

# Grid dimensions
width, height = 100, 100

# Noise parameters
scale = 20.0
octaves = 4
persistence = 0.5
lacunarity = 2.0
seed = 42

img_array = np.zeros((height, width))

# Generate Perlin noise
for i in range(height):
    for j in range(width):
        x = i / scale
        y = j / scale
        v = noise.pnoise2(
            x,
            y,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            repeatx=1024,
            repeaty=1024,
            base=seed,
        )
        # Normalize to 0.0 - 1.0
        img_array[i][j] = (v + 1) / 2.0

img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)

# Save
img = Image.fromarray(img_array, mode="L")
img.save("terrain.png")

print("Successfully generated terrain.png")
