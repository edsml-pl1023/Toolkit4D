# Peiyi Leng; edsml-pl1023
import napari
import tifffile
import json
import os


def label_image(viewer, image_path, output_dir):
    """
    Loads a 3D binary TIFF image into a Napari viewer, allows
    the user to enter a label for the image, and saves the label
    as a JSON file in the specified output directory.

    Args:
        viewer (napari.Viewer): An instance of a Napari viewer
        for displaying the image.
        image_path (str): The file path to the 3D binary
        TIFF image to be labeled.
        output_dir (str): The directory where the JSON label file
        will be saved.

    Details:
        - The function adds the image to the Napari viewer with specific
          display settings:
            - Colormap: 'gray'
            - Opacity: 0.3
            - Blending: 'translucent'
            - Contrast limits: Set to the full range of the image data
            - Gamma: 0.7
        - The display mode is set to 3D, and the rendering mode is set to
          'attenuated_mip' to enhance depth perception.
        - The user is prompted to enter a label for the image, which is then
          saved as a JSON file with the image filename and the label.

    Returns:
        None
    """
    # Load the 3D binary TIFF file
    image_data = tifffile.imread(image_path)

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
    layer.opacity = 0.3
    layer.contrast_limits = (0, image_data.max())

    # Show the axes by default
    viewer.axes.visible = True
    viewer.axes.colored = False
    viewer.axes.labels = True

    # Allow the user to view the image and enter a label
    label = input(
        f"Enter the label for image '{os.path.basename(image_path)}': ")

    # Save the label to a JSON file with the same name as the image
    label_data = {
        "filename": os.path.basename(image_path),
        "label": label
    }
    json_filename = os.path.basename(image_path) + "_label.json"
    label_path = os.path.join(output_dir, json_filename)

    with open(label_path, 'w') as json_file:
        json.dump(label_data, json_file)

    print(f"Label for '{os.path.basename(image_path)}' saved to {label_path}")

    # Clear the viewer for the next image
    viewer.layers.clear()


def label_images_in_folder(folder_path):
    """
    Iterates over all TIFF files in a specified folder,
    loads each image into a Napari viewer for labeling,
    and saves the labels as JSON files in a subdirectory named 'labels'.

    Args:
        folder_path (str): The path to the folder containing the
        TIFF files to be labeled.

    Details:
        - The function identifies all '.tif' and '.tiff' files
          in the given folder.
        - It creates a 'labels' directory within the folder to
          store the JSON label files.
        - A Napari viewer is created to display each image, and
          the `label_image` function is called for each image to
          handle the labeling process.
        - The Napari viewer is closed after all images have been labeled.

    Returns:
        None
    """
    # Get a list of all TIFF files in the folder
    tiff_files = [f for f in os.listdir(folder_path) if
                  f.endswith('.tif') or f.endswith('.tiff')]

    # Create an output directory for the labels
    output_dir = os.path.join(folder_path, "labels")
    os.makedirs(output_dir, exist_ok=True)

    # Create the Napari viewer
    viewer = napari.Viewer()

    # Iterate over each TIFF file in the folder
    for tiff_file in tiff_files:
        image_path = os.path.join(folder_path, tiff_file)
        label_image(viewer, image_path, output_dir)

    # Close the viewer after all images are labeled
    napari.run()
