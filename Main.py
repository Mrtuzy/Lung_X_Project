import os
import numpy as np
from Demo.SRCNN import SRCNN
from PIL import Image


class SuperResolution:
    def __init__(self, lr_folder, hr_folder):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.model = SRCNN()

    def load_images(self, folder):
        images = []
        for file_name in os.listdir(folder):
            img_path = os.path.join(folder, file_name)
            img = Image.open(img_path).convert('YCbCr')
            y, _, _ = img.split()
            img = np.array(y).astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            images.append(img)
        return np.array(images)

    def prepare_data(self):
        lr_images = self.load_images(self.lr_folder)
        hr_images = self.load_images(self.hr_folder)
        return lr_images, hr_images

    def train_model(self, epochs=100, batch_size=4):
        lr_images, hr_images = self.prepare_data()
        self.model.train(lr_images, hr_images, epochs=epochs, batch_size=batch_size)

    def save_model(self, file_path):
        self.model.save_model(file_path)

    def load_model(self, file_path):
        self.model.load_model(file_path)

    def evaluate_model(self):
        lr_images, hr_images = self.prepare_data()
        return self.model.evaluate(lr_images, hr_images)

    def predict_image(self, lr_image_path):
        img = Image.open(lr_image_path).convert('YCbCr')
        y, _, _ = img.split()
        lr_image = np.array(y).astype(np.float32) / 255.0
        lr_image = np.expand_dims(lr_image, axis=-1)  # Add channel dimension
        lr_image = np.expand_dims(lr_image, axis=0)  # Add batch dimension

        sr_image = self.model.predict(lr_image)[0]
        sr_image = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)

        return Image.fromarray(sr_image.squeeze(), mode='L')
def upscale_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, file_name)
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            new_width = image.width * 2
            new_height = image.height * 2
            upscaled_image = image.resize((new_width, new_height), Image.BICUBIC)
            output_path = os.path.join(output_folder, file_name)
            upscaled_image.save(output_path)


if __name__ == "__main__":
    pre_lr_folder = "LR_lung_images"
    hr_folder = "HR_lung_images"
    lr_folder = 'Input_Lr_lung_images'
    filtered_input_folder = 'Filtered_lung_images'
    filtered6x6_input_folder = 'Filtered_lung_images'

    # # print("Upscaling images...")
    # # upscale_images(pre_lr_folder, lr_folder)
    # # print("Upscaling complete.")
    # sr = SuperResolution(filtered6x6_input_folder, hr_folder)
    # print("Training model...")
    # sr.train_model(epochs=100, batch_size=4)
    # sr.save_model("6x6srcnn_model.h5")
    #
    # # Evaluate model
    # eval_results = sr.evaluate_model()
    # print(f"Evaluation results: {eval_results}")

    #Predict an image
    lr_image_path = "../../../Python/Lung_X_Project/Test/NewFiltered(6x6).png"
    sr = SuperResolution(lr_folder, hr_folder)
    sr.load_model("6x6srcnn_model.h5")
    sr_image = sr.predict_image(lr_image_path)
    sr_image.show()
    sr_image.save("Predicts/PreTestResultNew(6x6).png")
