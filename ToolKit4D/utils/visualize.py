# Peiyi Leng; edsml-pl1023
import napari
import os
import tifffile


def visualize(image_path):
    """
    Visualize a 3D binary TIFF image using Napari with specific display settings.

    This function loads a 3D binary TIFF image from the specified file path, 
    creates a Napari viewer, and adds the image with custom display settings. 
    The image is displayed in 3D mode with an 'attenuated_mip' rendering 
    mode to enhance depth perception. The viewer's axes are shown by default.

    Args:
        image_path (str): The file path to the 3D binary TIFF image.
    """
    # Load the 3D binary TIFF file
    image_data = tifffile.imread(image_path)

    # Create a Napari viewer
    viewer = napari.Viewer()

    # Add the image with specific display settings
    layer = viewer.add_image(image_data,
                             name=os.path.basename(image_path),
                             colormap='gray',
                             opacity=0.5,
                             blending='translucent',
                             contrast_limits=(0, image_data.max()),
                             gamma=0.7)

    # Set display to 3D mode
    viewer.dims.ndisplay = 3

    # Set the rendering mode to 'attenuated_mip' for maximum attenuation effect
    layer.rendering = 'attenuated_mip'

    # Maximize the attenuation effect
    layer.opacity = 0.3  # Adjust opacity to enhance depth perception
    layer.contrast_limits = (0, image_data.max())  # Reinforce full contrast range

    # Show the axes by default
    viewer.axes.visible = True
    viewer.axes.colored = False  
    viewer.axes.labels = True    

    # Start the Napari viewer (this will open a window)
    napari.run()
