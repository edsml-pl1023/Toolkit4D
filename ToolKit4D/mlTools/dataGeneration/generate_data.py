# Peiyi Leng; edsml-pl1023
from ...pipeline import ToolKitPipeline
import os
import tifffile


def generate_data(folder_path, result_base_dir,
                  numAgg_range=[3, 4, 5, 6, 7, 8]):
    """
    This function aims to generate a set of training data by
    processing a collection of .raw files. It uses the ToolKitPipeline to
    segment and separate rocks, then saves the resulting agglomerate masks
    as TIFF files.

    Args:
        folder_path (str): The directory path where the .raw files are located.
        result_base_dir (str): The base directory where the processed
        results will be saved.
        numAgg_range (list, optional): A list of integers specifying the
        number of agglomerates to generate in the separation process.
        Defaults to [3, 4, 5, 6, 7, 8].

    Returns:
        None

    Example:
        generate_data('./raw', './results/ml')
    """
    # Get a list of all .raw files in the folder
    raw_files = [f for f in os.listdir(folder_path) if f.endswith('.raw')]

    file_paths = [os.path.join(folder_path, file_name) for
                  file_name in raw_files]
    print(file_paths)

    for file in file_paths:
        try:
            pipeline = ToolKitPipeline(file, load=True)
            # You can add more processing code here if needed
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue  # Skip to the next file if there is an error

        result_folder = os.path.join(result_base_dir, pipeline.identifier)
        os.makedirs(result_folder, exist_ok=True)

        pipeline.segment_rocks()
        for num_agglomerate in numAgg_range:
            if hasattr(pipeline, 'agglomerate_masks'):
                del pipeline.agglomerate_masks
            pipeline.separate_rocks(num_agglomerates=num_agglomerate)
            for i, agglomerate_mask in enumerate(pipeline.agglomerate_masks):
                output_path = os.path.join(
                    result_folder,
                    f'{pipeline.identifier}_NumAgg{num_agglomerate}_Agg{i}.tif'
                )
                tifffile.imwrite(output_path, agglomerate_mask)
                print(f"Saved: {output_path}")
