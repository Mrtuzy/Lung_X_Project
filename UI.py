import subprocess
import sys
import tkinter.messagebox
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from SRCNN import *
from PIL import Image
import cv2
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
        return self.model.train(lr_images, hr_images, epochs=epochs, batch_size=batch_size)

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
class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Mchine Learning Image Upscaler")
        self.root.geometry("600x620")
        self.root.resizable(False, False)
        self.root.configure(bg="white")
        self.image_path = None

        self.image_label = Label(self.root, bg="white")
        self.image_label.place(x=20, y=20)

        self.select_button = Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.place(x=20, y=540)



        self.outputs_button = Button(self.root, text="Outputs", command=self.show_gallery)
        self.outputs_button.place(x=120, y=540)

        self.proccess_button = Button(self.root, text="Proccess", command=self.proccess_image)
        self.proccess_button.place(x=200, y=540)

        self.simulate_button = Button(self.root, text="Simulate", command=self.simulate_image)
        self.simulate_button.place(x=300, y=540)

        # List of strings for radio buttons
        self.radio_options = ["srcnn_model.h5", "new_srcnn_model.h5", "6x6srcnn_model.h5"]
        self.radio_var = StringVar(value=self.radio_options[0])

        # Create radio buttons dynamically
        for idx, option in enumerate(self.radio_options):
            radio_button = Radiobutton(self.root, text=option, variable=self.radio_var, value=option, bg="white")
            radio_button.place(x=400, y=540 + (idx * 30))

    def simulate_image(self):
        lr_train_folder = 'Filtered_lung_images'
        hr_train_folder = 'HR_lung_images'

        sr = SuperResolution(lr_folder=lr_train_folder, hr_folder=hr_train_folder)

        # Eğitim verilerini yükleyin
        lr_images, hr_images = sr.prepare_data()

        # Eğitim sürecini başlatın
        history = sr.train_model(epochs=5, batch_size=10)

        # Eğitim sürecinin grafiğini gösterin
        import matplotlib.pyplot as plt
        print(history.history.keys())
        print(history.history['loss'])
        print(history.history['val_loss'])
        plt.figure()
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Training Loss')
        plt.legend()
        plt.savefig('Predicts/training_loss.png')
        plt.close()

        # Eğitim grafiğini GUI'de gösterin
        result_path = 'Predicts/training_loss.png'
        tkinter.messagebox.showinfo("Training Simulation", "Training simulation complete. Check the result.")
        image = Image.open(result_path)
        image = image.resize((512, 512), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def proccess_image(self):
        pre_lr_folder = "LR_lung_images"
        hr_folder = "HR_lung_images"
        lr_folder = 'Input_Lr_lung_images'
        filtered_input_folder = 'Filtered_lung_images'
        filtered6x6_input_folder = 'Filtered_lung_images'
        result_path = 'Predicts/Result.png'
        if self.radio_var.get() == "new_srcnn_model.h5":
            lr_folder = filtered_input_folder
            lr_image_path = self.image_path
            sr = SuperResolution(lr_folder, hr_folder)
            sr.load_model(self.radio_var.get())
            sr_image = sr.predict_image(lr_image_path)
            sr_image.save(result_path)
            tkinter.messagebox.showinfo("Info", "Image has been processed successfully.")
            image = Image.open(result_path)
            image = image.resize((512, 512), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

        elif self.radio_var.get() == "srcnn_model.h5":
            lr_image_path = self.image_path
            image = cv2.imread(lr_image_path)
            height, weight, _ = image.shape
            sr = SuperResolution(lr_folder, hr_folder)
            sr.load_model(self.radio_var.get())
            new_image = np.zeros((height // 2, weight // 2, 3), dtype=np.uint8)
            a = 0
            b = 0
            for i in range(height):
                for j in range(weight):
                    if (i % 2 != 0) and (j % 2 != 0):
                        new_image[a, b] = image[i, j]
                        b += 1
                if b != 0:
                    a += 1
                b = 0
            cv2.imwrite("Demo.png", new_image)
            # Load the image
            image = Image.open("Demo.png")

            # Calculate the new dimensions
            new_width = image.width * 2
            new_height = image.height * 2

            # Resize the image using interpolation
            upscaled_image = image.resize((new_width, new_height), Image.BICUBIC)

            # Save the upscaled image
            upscaled_image.save("UpscaledDemo.png")

            sr_image = sr.predict_image("UpscaledDemo.png")
            sr_image.save(result_path)
            tkinter.messagebox.showinfo("Info", "Image has been processed successfully.")
            image = Image.open(result_path)
            image = image.resize((512, 512), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def select_image(self):
        file_path = filedialog.askopenfilename()
        self.image_path = file_path
        if file_path:
            image = Image.open(file_path)
            image = image.resize((512, 512), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

    def show_gallery(self):
        gallery_window = Toplevel(self.root)
        gallery_window.title("Image Gallery")
        gallery_window.geometry("800x600")
        gallery_window.configure(bg="white")

        canvas = Canvas(gallery_window, bg="white")
        scrollbar = Scrollbar(gallery_window, orient=VERTICAL, command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        def on_mouse_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mouse_wheel)

        def open_image_viewer(img_path):
            if os.name == 'nt':  # For Windows
                os.startfile(img_path)
            elif os.name == 'posix':  # For macOS and Linux
                subprocess.call(('open', img_path) if sys.platform == 'darwin' else ('xdg-open', img_path))

        output_folder = 'Predicts'
        if os.path.exists(output_folder):
            row = 0
            col = 0
            for img_name in os.listdir(output_folder):
                img_path = os.path.join(output_folder, img_name)
                image = Image.open(img_path)
                image = image.resize((200, 200), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)
                img_label = Label(scrollable_frame, image=photo, bg="white")
                img_label.image = photo
                img_label.grid(row=row, column=col, padx=10, pady=10)
                img_label.bind("<Button-1>", lambda e, img_path = img_path: open_image_viewer(img_path))
                col += 1
                if col == 3:
                    col = 0
                    row += 1
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    game = GUI()
    game.run()