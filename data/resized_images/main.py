from PIL import Image
import os

input_folder = "Stomach"
output_folder = "stomach_resized"
size = (512, 512)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        img_resized = img.resize(size, Image.LANCZOS)
        img_resized.save(os.path.join(output_folder, filename))
        print(f"{filename} 已轉換成 512x512")
