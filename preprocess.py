import os
import cv2
import numpy as np

def preprocess_images(image_dir, img_size=(224, 224)):
    images = []
    labels = []
    for label in os.listdir(image_dir):
        path = os.path.join(image_dir, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, img_size)
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

if __name__ == "__main__":
    images, labels = preprocess_images('dataset/')
    np.save('images.npy', images)
    np.save('labels.npy', labels)
