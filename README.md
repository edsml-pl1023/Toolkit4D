#  4D mineral characterisation toolkit

## Project Overview
This project involves the development of `ToolKit4D`, a Python package designed to process 3D X-ray microtomography (XMT) scans of agglomerate particles. The package was created by converting Matlab code to Python and integrating advanced machine learning techniques to enhance the segmentation process. Key features include automated watershed segmentation, efficient data processing routines, and a user-friendly interface that simplifies parameter selection and reduces manual effort. The project is structured into several subpackages, each responsible for specific functionalities, such as data I/O, thresholding, and segmentation, all of which work together within the `ToolKitpipeline` class to provide a comprehensive solution for analyzing XMT data.

## Code Structure
The repository is organised into the following structure:
```
.
├── ToolKit4D/
│   ├── __init__.py
│   ├── dataio
│   │   ├── __init__.py
│   │   ├── read_raw.py
│   │   ├── RGBtif_read.py
│   │   ├── RGBtif_write.py
│   │   └── tif_read.py
│   ├── mlTools
│   │   ├── __init__.py
│   │   ├── dataGeneration
│   │   │   ├── __init__.py
│   │   │   ├── generate_data.py
│   │   │   └── label_data.py
│   │   ├── dataset
│   │   │   ├── __init__.py
│   │   │   ├── AggDataset.py
│   │   │   └── mask_integrator.py
│   │   ├── model
│   │   │   ├── __init__.py
│   │   │   └── CompactUNet3D.py
│   │   ├── predicting
│   │   │   ├── __init__.py
│   │   │   ├── ML_separate_rocks.py
│   │   │   └── predict_NumAgglomerates.py
│   │   ├── training
│   │   │   ├── __init__.py
│   │   │   └── train.py
│   │   └── utils
│   │       ├── __init__.py
│   │       └── initialize.py
│   ├── stages
│   │   ├── __init__.py
│   │   ├── agglomerate_extraction.py
│   │   ├── segment_rocks.py
│   │   └── separate_rocks.py
│   ├── thresholding
│   │   ├── __init__.py
│   │   ├── threshold_grain.py
│   │   └── threshold_rock.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── remove_cylinder.py
│   │   └── visualize.py
│   └── pipeline.py
└── tests/
    ├── __init__.py
    ├── test_agglomerate_extraction.py
    ├── test_readraw.py
    ├── test_remove_cylinder.py
    ├── test_segment_rocks.py
    ├── test_separate_rocks.py
    ├── test_threshold_grain.py
    ├── test_threshold_rock.py
    └── test_data/
```

## Setup
To set up the project, ensure the following steps are followed:

1. Clone this repository:

    ```sh
    git clone https://github.com/yourusername/yourrepository.git
    ```

2. Navigate to the repository directory:

    ```sh
    cd /path/to/your/repository
    ```

3. Create a new conda environment:

    ```sh
    conda create -n ToolKit python=3.10 pip
    ```

4. Activate the environment:

    ```sh
    conda activate ToolKit
    ```

5. Install the project dependencies:

    ```sh
    pip install -e .
    ```

6. Test the installation:

    ```sh
    pytest
    ```

## Usage Guide
To help you get started with using this project, please follow these steps:

1. **Download Raw Images:**  
   Before proceeding, download the raw images required for the project and place them in the `./raw` directory. (e.g. LA3_d0_v1_uint16_unnormalised_2028_2028_2028.raw)

2. **Review the Usage Guide Notebook:**  
   After setting up the raw images, check the following Jupyter notebook for detailed instructions and examples:

**Notebook:** [notebooks/useguide.ipynb](./notebooks/useguide.ipynb)

This notebook covers the key features and functionalities of the project, including:

- Walk you through the `ToolKit4D` image processing routine, including:
    - Load raw images.
    - Process the images step by step through the pipeline.
    - Save results to the specified directory.

## Visualization Results
After saving the results of the processed images from [notebooks/useguide.ipynb](./notebooks/useguide.ipynb), or by placing processed images into the corresponding folders, you can visualize the results. For detailed instructions on how to visualize the processed images, refer to the following notebook:

**Notebook:** [notebooks/visualize.ipynb](./notebooks/visualize.ipynb)


## Hardware Requirements
Minimum of 150GB CPU RAM


## Author
Peiyi Leng

## License
MIT License