from PIL import Image
import os

image_folder = 'assets/imgs/linear/'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images.sort(key=lambda x: int(x.split('.')[0]))

first_image = Image.open(os.path.join(image_folder, images[0]))

other_images = [Image.open(os.path.join(image_folder, img)) for img in images[1:]]
other_images += [other_images[-1]] * 10

output_path = 'assets/gifs/result.gif'
first_image.save(output_path, save_all=True, append_images=other_images, duration=100, loop=100)
