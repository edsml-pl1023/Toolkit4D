import numpy as np
import torch
from scipy.ndimage import zoom


def predict_NumAgglomerates(model, agglomerate):
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
    """Resizes a 3D image to a new shape using scipy's zoom."""
    # Compute the zoom factors for each dimension
    zoom_factors = [new_dim / old_dim for new_dim, old_dim in
                    zip(new_shape, image.shape)]

    # Apply zoom to the image
    resized_image = zoom(image, zoom_factors, order=0)

    return resized_image
