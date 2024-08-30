# Peiyi Leng; edsml-pl1023
import numpy as np
import torch
from scipy.ndimage import zoom


def predict_NumAgglomerates(model, agglomerate):
    """
    Predict the number of agglomerates in a 3D binary image using a trained
    model.

    This function processes a 3D binary image of an agglomerate, resizes it,
    and prepares it for input to a trained PyTorch model. The model predicts
    the number of agglomerates present in the image.

    Args:
        model (torch.nn.Module): The trained PyTorch model used for prediction.
        agglomerate (numpy.ndarray): A 3D binary image representing the
                                     agglomerate to be analyzed.

    Returns:
        int: The predicted number of agglomerates in the image.
    """
    agglomerate = np.where(agglomerate, 255, 0).astype(np.uint8)
    agglomerate = resize_image(agglomerate, (64, 64, 64))
    agglomerate = torch.tensor(agglomerate, dtype=torch.uint8)
    agglomerate = agglomerate.float() / 255.0
    # Add channel dimension
    agglomerate = agglomerate.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(agglomerate.view(-1, 1, 64, 64, 64))
        y_pred = torch.round(output).long().squeeze()
        return int(y_pred.cpu().numpy())


def resize_image(image, new_shape):
    """
    Resize a 3D image to a new shape using the specified zoom factors.

    This function resizes a 3D image to the desired dimensions by computing
    the zoom factors for each axis and applying the zoom operation.

    Args:
        image (numpy.ndarray): The input 3D image to be resized.
        new_shape (tuple): The target shape for the resized image, specified
                           as a tuple of three integers (depth, height, width).

    Returns:
        numpy.ndarray: The resized 3D image.
    """
    # Compute the zoom factors for each dimension
    zoom_factors = [new_dim / old_dim for new_dim, old_dim in
                    zip(new_shape, image.shape)]

    # Apply zoom to the image
    resized_image = zoom(image, zoom_factors, order=0)

    return resized_image
