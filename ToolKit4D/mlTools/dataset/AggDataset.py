import os
import json
import numpy as np
import torch
from ...dataio import tif_read
from torch.utils.data import Dataset
from scipy.ndimage import zoom


# class AggDataset(Dataset):
#     def __init__(self, path: str, identifiers: list = None,
#                  shape: tuple = (256, 256, 256)):
#         self.path = path
#         self.shape = shape
#         self.identifiers = (identifiers if identifiers is not None
#                             else self._get_all_identifiers())
#         self.dataPath = self._get_file_paths()

#     def _get_file_paths(self):
#         dataPath = []
#         for identifier in self.identifiers:
#             identifier_path = os.path.join(self.path, identifier)
#             images = [image for image in os.listdir(identifier_path)
#                       if image.endswith('tif')]
#             for image in images:
#                 image_path = os.path.join(identifier_path, image)
#                 label_path = os.path.join(identifier_path, 'labels',
#                                           f"{image}_label.json")
#                 dataPath.append((image_path, label_path))
#         return dataPath

#     def _get_all_identifiers(self):
#         return [name for name in os.listdir(self.path)
#                 if os.path.isdir(os.path.join(self.path, name))]

#     def _resize_image(self, image, new_shape):
#         """Resizes a 3D image to a new shape using scipy's zoom."""
#         # Compute the zoom factors for each dimension
#         zoom_factors = [new_dim / old_dim for new_dim, old_dim in
#                         zip(new_shape, image.shape)]

#         # Apply zoom to the image
#         resized_image = zoom(image, zoom_factors, order=0)

#         return resized_image

#     def __len__(self):
#         return len(self.dataPath)

#     def __getitem__(self, idx):
#         image_path, label_path = self.dataPath[idx]

#         # load image and resize
#         image = tif_read(image_path)
#         image = np.where(image, 255, 0).astype(np.uint8)
#         image = self._resize_image(image, self.shape)

#         # convert to tensor
#         image = torch.tensor(image, dtype=torch.uint8)
#         image = image.float() / 255.0

#         # Add channel dimension
#         image = image.unsqueeze(0)

#         # load label
#         with open(label_path, 'r') as f:
#             label_data = json.load(f)
#             label = int(label_data["label"])

#         return image, label


class AggDataset(Dataset):
    def __init__(self, path: str, identifiers: list = None,
                 shape: tuple = (256, 256, 256), preload: bool = False):
        self.path = path
        self.shape = shape
        self.preload = preload
        self.identifiers = (identifiers if identifiers is not None
                            else self._get_all_identifiers())
        self.dataPath = self._get_file_paths()

        if self.preload:
            self.data = self._load_all_data_into_memory()

    def _get_file_paths(self):
        dataPath = []
        for identifier in self.identifiers:
            identifier_path = os.path.join(self.path, identifier)
            images = [image for image in os.listdir(identifier_path)
                      if image.endswith('tif')]
            for image in images:
                image_path = os.path.join(identifier_path, image)
                label_path = os.path.join(identifier_path, 'labels',
                                          f"{image}_label.json")
                dataPath.append((image_path, label_path))
        return dataPath

    def _get_all_identifiers(self):
        return [name for name in os.listdir(self.path)
                if os.path.isdir(os.path.join(self.path, name))]

    def _resize_image(self, image, new_shape):
        """Resizes a 3D image to a new shape using scipy's zoom."""
        # Compute the zoom factors for each dimension
        zoom_factors = [new_dim / old_dim for new_dim, old_dim in
                        zip(new_shape, image.shape)]

        # Apply zoom to the image
        resized_image = zoom(image, zoom_factors, order=0)

        return resized_image

    def _load_all_data_into_memory(self):
        all_data = []
        for image_path, label_path in self.dataPath:
            # load image and resize
            image = tif_read(image_path)
            image = np.where(image, 255, 0).astype(np.uint8)
            image = self._resize_image(image, self.shape)

            # convert to tensor
            image = torch.tensor(image, dtype=torch.uint8)
            image = image.float() / 255.0

            # Add channel dimension
            image = image.unsqueeze(0)

            # load label
            with open(label_path, 'r') as f:
                label_data = json.load(f)
                label = int(label_data["label"])

            all_data.append((image, label))

        return all_data

    def __len__(self):
        return len(self.dataPath)

    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        else:
            image_path, label_path = self.dataPath[idx]

            # load image and resize
            image = tif_read(image_path)
            image = np.where(image, 255, 0).astype(np.uint8)
            image = self._resize_image(image, self.shape)

            # convert to tensor
            image = torch.tensor(image, dtype=torch.uint8)
            image = image.float() / 255.0

            # Add channel dimension
            image = image.unsqueeze(0)

            # load label
            with open(label_path, 'r') as f:
                label_data = json.load(f)
                label = int(label_data["label"])

            return image, label
