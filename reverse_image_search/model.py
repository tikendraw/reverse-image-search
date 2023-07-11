import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import compress_pickle
import concurrent


class ImageSimilarity:
    def __init__(
        self,
        img_dir: Path,
        recursive: bool = False,
        BATCH_SIZE: int = 64,
        IMG_SIZE: int = 224,
        save_model: bool = True,
    ):
        self.batch_size = BATCH_SIZE
        self.img_size = IMG_SIZE
        self.img_dir = img_dir
        self.model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            alpha=1.0,
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classifier_activation="softmax",
        )

        self.model.trainable = False
        self.model.compile()

        self.save_model = save_model
        self.recursive = recursive
        self.ifeatures = None
        self.filename = "image_dict.lzma"
        self.image_dict = None
        self.images_found = None

    def get_image_paths(self, directory_path: Path, recursive: bool = False) -> list:
        image_extensions = [".jpg", ".jpeg", ".png"]  # Add more extensions if needed
        image_paths = []

        for file_path in directory_path.iterdir():
            if file_path.is_file() and (file_path.suffix.lower() in image_extensions):
                image_paths.append(str(file_path.absolute()))

            elif recursive and file_path.is_dir():
                image_paths.extend(self.get_image_paths(file_path, recursive))

        return image_paths

    def load_image(self, x):
        image_data = tf.io.read_file(x)
        image_features = tf.image.decode_jpeg(image_data, channels=3)
        image_features = tf.image.resize(image_features, (self.img_size, self.img_size))
        return image_features

    def load_image2(self, x):
        image_data = tf.keras.utils.img_to_array(x)
        return tf.image.resize(image_data, (self.img_size, self.img_size))

    def get_vectors(self, image_data: tf.data.Dataset) -> np.array:
        features = []
        for i in tqdm(image_data):
            y = self.model(i)
            pooled_features = tf.keras.layers.GlobalMaxPooling2D()(y)
            features.append(pooled_features)

        ifeatures = tf.concat(features, axis=0)
        ifeatures = tf.cast(ifeatures, tf.float16).numpy()
        return ifeatures

    def similar_image(self, x, k=5):
        x = (
            self.load_image(str(x.absolute()))
            if isinstance(x, Path)
            else self.load_image2(x)
        )

        x_logits = self.model(tf.expand_dims(x, 0))
        x_logits = (
            tf.keras.layers.GlobalAveragePooling2D()(x_logits)
            .numpy()
            .astype("float16")
            .reshape((1, -1))
            .tolist()
        )

        x_similarity = cosine_similarity(x_logits, self.ifeatures).tolist()[0]

        x_sim_idx = np.argsort(x_similarity)[::-1][:k]
        x_sim_values = sorted(x_similarity, reverse=True)[:k]
        keys_at_indices = [list(self.image_dict.keys())[index] for index in x_sim_idx]
        return keys_at_indices, x_sim_values

    def build_image_features(self):
        images = self.get_image_paths(self.img_dir, recursive=self.recursive)

        image_data = (
            tf.data.Dataset.from_tensor_slices(images)
            .map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
        )

        self.ifeatures = self.get_vectors(image_data)
        self.image_dict = OrderedDict(zip(images, self.ifeatures))

        # print('ifeatures.shape:', self.ifeatures.shape)
        # print('Features loaded!')

    def load_image_dict(self):
        if os.path.isfile(self.filename):
            image_dict = compress_pickle.load(self.filename, compression="lzma")
            images = self.get_image_paths(self.img_dir, recursive=self.recursive)
            if images == list(image_dict.keys()):
                self.image_dict = image_dict
                self.ifeatures = np.array(list(image_dict.values()))
            else:
                self.build_image_features()
        else:
            self.build_image_features()

    def save_image_dict(self):
        compress_pickle.dump(self.image_dict, self.filename, compression="lzma")

    def is_changed(self):
        images = self.get_image_paths(self.img_dir, recursive=self.recursive)
        previous_images = list(self.image_dict.keys())
        return images != previous_images

    def find_similar_images(self, x, k=5):
        # creating/loading vectors
        self.load_image_dict()
        if k == -1:
            k = self.ifeatures.shape[0]

        sim_img, x_sim = self.similar_image(x, k=k)
        # print('plotting')
        plt.figure(figsize=(5, 5))
        testimg = plt.imread(str(x.absolute()))
        plt.imshow(testimg)
        plt.title(f"{x.name}(main)")
        plt.show()
        self.show_images(sim_img, similar=x_sim)
        return x_sim

    def find_similar_images2(self, x, k=5):
        self.load_image_dict()
        if k == -1:
            k = self.ifeatures.shape[0]

        sim_img, x_sim = self.similar_image(x, k=k)
        return sim_img, x_sim

    def show_images(self, x: list, similar: list = None, figsize=None):
        n_plots = len(x)
        # print('n plots: ', n_plots)
        if figsize is None:
            # figsize = (20, int(n_plots // 5) * 4)
            figsize = (20, 5)

        # print('figsize: ',figsize)
        plt.figure(figsize=figsize)

        x = [Path(i) for i in x]
        for num, i in enumerate(x, 1):
            plt.subplot((n_plots // 5) + 1, 5, num)
            img = plt.imread(i)
            plt.imshow(img)
            title = (
                f"{i.name}\n({100 * similar[num - 1]:.2f}%)"
                if similar is not None
                else i.name
            )
            plt.title(title)
            plt.axis(False)
            plt.tight_layout()

        plt.show()

    def __call__(self, x: Path, k=5):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            finding = executor.submit(self.find_similar_images(x, k=5))

            if self.save_model and (
                self.is_changed() or (not Path(self.filename).exists())
            ):
                save_imagedict = executor.submit(self.save_image_dict)
