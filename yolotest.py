from ultralytics import YOLO
import os
import cv2


model_path = 'last.pt'

folder_path = 'images' # Folder containing images to be segmented example: images/1.JPEG

img_names = os.listdir(folder_path)
img_names.sort()


for image_path in img_names:
    print(image_path)
    img = cv2.imread("images/" + image_path)
    # resize image to 640x640
    img = cv2.resize(img, (640, 640))
    H, W, _ = img.shape

    model = YOLO(model_path)

    results = model(img)
    for result in results:
        for j, mask in enumerate(result.masks.data):
            mask = mask.cpu().numpy() * 255  # Move tensor to CPU before converting to NumPy array
            mask = cv2.resize(mask, (W, H))
            cv2.imwrite(f'results/{image_path[:-4]}{j}.png', mask)  # Save each mask with a unique filename

