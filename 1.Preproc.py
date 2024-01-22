from PIL import Image
import os

original_image = Image.open("Data/input/Subss.jpg")

# tile_size = (64, 64)
tile_size = (128, 128)

tiles_x = original_image.size[0] // tile_size[0]
tiles_y = original_image.size[1] // tile_size[1]

images_dir = "Data/images/all"
os.makedirs(images_dir, exist_ok=True)

for i in range(tiles_x):
    for j in range(tiles_y):
        left = i * tile_size[0]
        upper = j * tile_size[1]
        right = left + tile_size[0]
        lower = upper + tile_size[1]

        tile = original_image.crop((left, upper, right, lower))
        tile_path = os.path.join(images_dir, f"tile_{i}_{j}.jpg")
        tile.save(tile_path)
