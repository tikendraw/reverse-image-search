import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from model import ImageSimilarity
from pathlib import Path
import matplotlib.pyplot as plt


class ImageSimilarityGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Similarity")

        self.folder_entry = tk.Entry(self.root)
        self.folder_entry.pack()

        self.folder_button = tk.Button(
            self.root, text="Select Folder", command=self.select_folder
        )
        self.folder_button.pack()

        self.image_entry = tk.Entry(self.root)
        self.image_entry.pack()

        self.image_button = tk.Button(
            self.root, text="Select Image", command=self.select_image
        )
        self.image_button.pack()

        self.num_entry = tk.Entry(self.root)
        self.num_entry.pack()

        self.plot_button = tk.Button(
            self.root, text="Plot Images", command=self.plot_images
        )
        self.plot_button.pack()

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()

        self.image_similarity = None

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        self.folder_entry.delete(0, tk.END)
        self.folder_entry.insert(0, folder_path)

    def select_image(self):
        image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        self.image_entry.delete(0, tk.END)
        self.image_entry.insert(0, image_path)

    def plot_images(self):
        folder_path = Path(self.folder_entry.get())
        image_path = Path(self.image_entry.get())
        num_images = int(self.num_entry.get())

        self.image_similarity = ImageSimilarity(folder_path)

        similar_images, similarities = self.image_similarity.find_similar_images2(
            image_path, k=num_images
        )

        figure = Figure(figsize=(10, 5))
        for i, image_path in enumerate(similar_images):
            ax = figure.add_subplot(1, num_images + 1, i + 1)
            ax.imshow(plt.imread(image_path))
            ax.axis("off")
            ax.set_title(f"Similarity: {similarities[i]:.2f}")

        self.canvas = FigureCanvasTkAgg(figure, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = ImageSimilarityGUI()
    gui.run()
