# Peiyi Leng; edsml-pl1023
import ToolKit4D.dataio as dio
import ToolKit4D.thresholding as thresh
import ToolKit4D.utils as ut
import ToolKit4D.stages as st
import ToolKit4D.mlTools as mlTools
import os
import gc
import warnings
import numpy as np
import tifffile
import torch
from typing import Literal
from ToolKit4D.mlTools.model import CompactUNet3D

# 1. call one functoin to execute all previous function
# 2. add option to load disk data stored at to each funciton
#    - so no need to run 'previous' function again
# 3. To avoid executing multiple times when execute previous methods;
#    use hasattr(self, 'property')
#    - if call at once; default parameter
#    - if call separatly: user parameter can set


class ToolKitPipeline:
    """
    A pipeline for processing 3D rock images, including thresholding, cylinder
    removal, rock segmentation, agglomerate separation, and grain thresholding.

    This class provides methods to process a raw 3D rock image through a series
    of steps, including thresholding, cylinder removal, segmentation, and grain
    extraction. The results of each step can be saved, and previously saved
    results can be loaded to continue processing without repeating earlier
    steps.

    Args:
        rawfile (str): The file path to the raw 3D image file.
        load (bool, optional): If True, loads previously saved processing
                               results from disk. Defaults to False.

    Attributes:
        rawfile (str): The file path to the raw 3D image file.
        identifier (str): A unique identifier for the dataset, derived from the
                          raw file name.
        im_size (list): The dimensions of the 3D image.
        im_type (str): The data type of the image (e.g., uint16).
        raw (numpy.ndarray): The raw 3D image data.
        image_folder (str): Directory path to save image-related outputs.
        threshold_folder (str): Directory path to save threshold-related
        outputs.

    Methods:
        initialize(): Resets all attributes except those defined in __init__.
        _load_saved_files(): Loads all saved files into corresponding
        attributes.
        _read_raw(): Reads the raw image data from the file.
        threshold_rock(save=False): Finds and applies a threshold to the rock
                                     image.
        remove_cylinder(ring_rad=792, ring_frac=1.2, del_attr=False,
                        save=False):
            Removes a cylindrical artifact from the image.
        segment_rocks(remove_cylinder=True, min_obj_size=1000, del_attr=False,
                      save=False): Segments the rocks in the image.
        separate_rocks(suppress_percentage=10, min_obj_size=1000, ML=False,
                       del_attr=False, save=False, num_agglomerates=None):
            Separates the rocks into agglomerates.
        agglomerate_extraction(min_obj_size=1000, del_attr=False, save=False):
            Extracts individual agglomerates from the segmented rocks.
        threshold_grain(method, del_attr=False, save=False): Thresholds the
                                                             grains within
                                                             each agglomerate
                                                             using a specified
                                                             method.
    """
    def __init__(self, rawfile, load: bool = False):
        """
        Initialize the ToolKitPipeline with a raw 3D image file.

        This constructor sets up the necessary attributes for processing a raw
        3D image file, including extracting metadata from the file name,
        reading the image data, and creating directories for saving processing
        results. If `load` is set to True, it will also load previously saved
        processing files into the instance.

        Args:
            rawfile (str): The file path to the raw 3D image file. The file
                           name should follow a specific format to extract
                           relevant metadata (e.g., image size and type).
            load (bool, optional): If True, loads previously saved processing
                                   results from disk into the instance.
                                   Defaults to False.
        """
        self.rawfile = rawfile
        clean_path = os.path.basename(rawfile)
        file_name = os.path.splitext(clean_path)[0]
        fnparts = file_name.split('_')
        self.identifier = fnparts[0] + fnparts[2] + fnparts[1]
        self.im_size = [int(fnparts[5]), int(fnparts[6]), int(fnparts[7])]
        self.im_type = fnparts[3]
        self.raw = self._read_raw()

        # create dir for 'save' image
        result_folder = os.path.join('./results', self.identifier + '/')
        os.makedirs(result_folder, exist_ok=True)
        # Create subfolders 'image' and 'threshold' under the result folder
        image_folder = os.path.join(result_folder, 'image/')
        threshold_folder = os.path.join(result_folder, 'threshold/')
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(threshold_folder, exist_ok=True)
        self.image_folder = image_folder
        self.threshold_folder = threshold_folder
        if load:
            self._load_saved_files()

    def initialize(self, delete_attrs=None):
        """
        Re-initialize the instance by deleting specified attributes or all
        attributes except those required for the initial setup.

        This method can be used to clear out attributes from the instance,
        either based on a provided list (`delete_attrs`) or by removing all
        attributes that are not essential for the initial state.

        Args:
            delete_attrs (list, optional): A list of attribute names to delete
                                           from the instance. If not provided,
                                           all attributes except the initial
                                           setup attributes will be deleted.
        """
        init_attrs = {'rawfile', 'identifier', 'im_size', 'im_type', 'raw',
                      'image_folder', 'threshold_folder'}
        attrs = set(self.__dict__.keys())
        # If delete_attrs is provided, delete those attributes
        if delete_attrs:
            for attr in delete_attrs:
                if attr in attrs:
                    delattr(self, attr)
        else:
            for attr in attrs - init_attrs:
                delattr(self, attr)
        gc.collect()

    def _load_saved_files(self):
        """
        Load all previously saved files into corresponding attributes.

        This method checks for the existence of saved files related to the
        current instance and loads them into the corresponding attributes.
        It includes rock threshold masks, column masks, optimized rock masks,
        agglomerate masks, fragments, and grain thresholds.

        Attributes Loaded:
            - rock_thresh_mask: The binary mask after rock thresholding.
            - rock_thresh: The threshold value used for rock segmentation.
            - column_mask: The binary mask representing the column after
                           cylinder removal.
            - optimized_rock_mask: The binary mask representing segmented
                                   rocks.
            - agglomerate_masks: A list of binary masks for separated
                                 agglomerates.
            - frags: A list of binary masks for individual fragments.
            - grain_threshs: A list of threshold values used for grain
                             segmentation.
            - grain_thresh_masks: A list of binary masks for segmented grains.
        """
        # Load rock_thresh_mask and rock_thresh
        rock_thresh_mask_path = os.path.join(self.image_folder,
                                             self.identifier +
                                             '_rock_thresh_mask.tif')
        rock_thresh_path = os.path.join(self.threshold_folder,
                                        self.identifier +
                                        '_rock_thresh.txt')
        if os.path.exists(rock_thresh_mask_path):
            self.rock_thresh_mask = dio.tif_read(rock_thresh_mask_path)
        if os.path.exists(rock_thresh_path):
            with open(rock_thresh_path, 'r') as file:
                self.rock_thresh = float(file.read().strip())

        # Load column_mask
        column_mask_path = os.path.join(self.image_folder,
                                        self.identifier +
                                        '_column_mask.tif')
        if os.path.exists(column_mask_path):
            self.column_mask = dio.tif_read(column_mask_path)

        # Load optimized_rock_mask
        optimized_rock_mask_path = os.path.join(self.image_folder,
                                                self.identifier +
                                                '_optimized_rock_mask.tif')
        if os.path.exists(optimized_rock_mask_path):
            self.optimized_rock_mask = dio.tif_read(optimized_rock_mask_path)

        # Load agglomerate_masks
        agglomerate_masks = []
        i = 0
        while True:
            agglomerate_mask_path = os.path.join(self.image_folder,
                                                 self.identifier +
                                                 f'_agglomerates_mask_{i}.tif')
            if os.path.exists(agglomerate_mask_path):
                agglomerate_masks.append(dio.tif_read(
                    agglomerate_mask_path))
                i += 1
            else:
                break
        if agglomerate_masks:
            self.agglomerate_masks = agglomerate_masks

        # Load frags
        frags = []
        i = 0
        while True:
            frag_path = os.path.join(self.image_folder,
                                     self.identifier +
                                     f'_frag_{i}.tif')
            if os.path.exists(frag_path):
                frags.append(dio.tif_read(frag_path))
                i += 1
            else:
                break
        if frags:
            self.frags = frags

        # Load grain_threshs and grain_thresh_masks
        grain_threshs = []
        grain_thresh_masks = []
        i = 0
        while True:
            grain_thresh_path = os.path.join(self.threshold_folder,
                                             self.identifier +
                                             f'_grain_thresh_{i}.txt')
            grain_thresh_mask_path = os.path.join(
                self.image_folder,
                self.identifier +
                f'_grain_thresh_mask_{i}.tif')
            if os.path.exists(grain_thresh_path) and (
               os.path.exists(grain_thresh_mask_path)):
                with open(grain_thresh_path, 'r') as file:
                    grain_threshs.append(float(file.read().strip()))
                grain_thresh_masks.append(dio.grain_mask_read(
                    grain_thresh_mask_path))
                i += 1
            else:
                break
        if grain_threshs:
            self.grain_threshs = grain_threshs
        if grain_thresh_masks:
            self.grain_thresh_masks = grain_thresh_masks

    def _read_raw(self):
        """
        Read the raw 3D image data from the file by calling
        dio.read_raw function.

        This method reads the raw image data using the specified image size and
        type, as determined from the file name.

        Returns:
            numpy.ndarray: The raw 3D image data.
        """
        raw = dio.read_raw(self.rawfile, self.im_size, self.im_type)
        return raw

    def threshold_rock(self, save: bool = False):
        """
        Find and apply a threshold to the rock image using the threshold_rock
        function.

        This method calculates the threshold for the rock image using the
        `thresh.threshold_rock` function and applies it to create a binary
        mask. If the threshold has not already been calculated, it will be
        computed and stored. The resulting threshold value and mask can be
        optionally saved to disk.

        Args:
            save (bool, optional): If True, saves the threshold value and the
                                   thresholded mask to disk. Defaults to False.
        """
        if not hasattr(self, 'rock_thresh'):
            print('-----Finding Rock Threshold-----')
            print('\t calling threshold_rock()')
            self.rock_thresh = thresh.threshold_rock(raw_image=self.raw)
            self.rock_thresh_mask = self.raw >= self.rock_thresh
            if save:
                tifffile.imwrite(
                    self.image_folder + self.identifier +
                    '_rock_thresh_mask.tif',
                    self.rock_thresh_mask
                )
                with open(self.threshold_folder + self.identifier +
                          '_rock_thresh.txt', 'w') as file:
                    file.write(str(self.rock_thresh))

    def remove_cylinder(self, ring_rad: int = 792, ring_frac: float = 1.2,
                        del_attr: bool = False, save: bool = False):
        """
        Remove cylindrical artifacts from the rock image using the
        remove_cylinder function.

        This method removes cylindrical artifacts from the binary rock mask
        created during thresholding. It first ensures the rock mask is
        generated using `threshold_rock()` and then applies the
        `ut.remove_cylinder` function to remove the cylinder. The result is
        stored as a column mask, which can be optionally saved. If specified,
        the rock threshold attribute can be deleted after processing.

        Args:
            ring_rad (int, optional): The inner radius of the cylinder to
                                      remove. Defaults to 792.
            ring_frac (float, optional): The ratio of the outer radius to the
                                         inner radius. Defaults to 1.2.
            del_attr (bool, optional): If True, deletes the rock threshold
                                       attribute after processing. Defaults to
                                       False.
            save (bool, optional): If True, saves the column mask to disk.
                                   Defaults to False.
        """
        if not hasattr(self, 'column_mask'):
            self.threshold_rock()
            print('-----Removing Cylinder-----')
            print('\t calling remove_cylinder()')
            if del_attr:
                del self.rock_thresh
                gc.collect()
            self.column_mask = ut.remove_cylinder(self.rock_thresh_mask,
                                                  ring_rad, ring_frac)
            if save:
                tifffile.imwrite(self.image_folder + self.identifier +
                                 '_column_mask.tif',
                                 self.column_mask
                                 )

    def segment_rocks(self, remove_cylinder: bool = True,
                      min_obj_size: int = 1000, del_attr: bool = False,
                      save: bool = False):
        """
        Segment rocks in the 3D image by optionally removing the cylinder and
        then applying segmentation.

        This method segments the rocks in the 3D image. It first optionally
        removes the cylindrical artifact using `remove_cylinder()`, and then
        applies the `st.segment_rocks` function to segment the rocks. The
        resulting segmented rock mask is stored and can be optionally saved.
        Attributes used during processing can be deleted if specified.

        Args:
            remove_cylinder (bool, optional): If True, removes the cylinder
                                              before segmentation. Defaults to
                                              True.
            min_obj_size (int, optional): The minimum object size for rocks
                                          to be considered during segmentation.
                                          Defaults to 1000.
            del_attr (bool, optional): If True, deletes intermediate attributes
                                       after processing. Defaults to False.
            save (bool, optional): If True, saves the segmented rock mask to
                                   disk. Defaults to False.
        """
        if not hasattr(self, 'optimized_rock_mask'):
            if remove_cylinder:
                self.remove_cylinder(del_attr=del_attr, save=save)
                initial_mask = self.column_mask
                if del_attr:
                    del self.rock_thresh_mask
                    del self.column_mask
                    gc.collect()
            else:
                self.threshold_rock(del_attr=del_attr, save=save)
                initial_mask = self.rock_thresh_mask
                if del_attr:
                    del self.rock_thresh_mask
                    gc.collect()
            print('-----Segment Rocks-----')
            print('\t calling segment_rocks()')
            self.optimized_rock_mask = st.segment_rocks(
                initial_mask,
                min_obj_size=min_obj_size)
            if save:
                tifffile.imwrite(self.image_folder + self.identifier +
                                 '_optimized_rock_mask.tif',
                                 self.optimized_rock_mask)

    def separate_rocks(self, suppress_percentage: int = 10,
                       min_obj_size: int = 1000, ML: bool = False,
                       del_attr: bool = False, save: bool = False,
                       num_agglomerates=None):
        """
        Separate rocks into individual agglomerates using different methods
        based on specified parameters.

        This method separates rocks into individual agglomerates after
        segmentation. It can use different approaches depending on the
        parameters:
        - A binary search for a specified number of agglomerates.
        - A machine learning-based method using a pre-trained model.
        - A standard suppression-based separation method.

        The resulting agglomerate masks can be optionally saved, and
        intermediate attributes can be deleted if specified.

        Args:
            suppress_percentage (int, optional): The percentage used to
                                                 suppress shallow minima
                                                 during agglomerate separation.
                                                 Defaults to 10.
            min_obj_size (int, optional): The minimum object size for
                                          agglomerates to be considered.
                                          Defaults to 1000.
            ML (bool, optional): If True, uses a machine learning-based method
                                 for agglomerate separation. Defaults to False.
            del_attr (bool, optional): If True, deletes intermediate attributes
                                       after processing. Defaults to False.
            save (bool, optional): If True, saves the agglomerate masks to
                                   disk. Defaults to False.
            num_agglomerates (int, optional): The desired number of
                                              agglomerates to find. If
                                              provided, a binary search
                                              method will be used.
        """
        self.segment_rocks()
        if not hasattr(self, 'agglomerate_masks'):
            print('-----Separate Agglomerates-----')
            print('\t calling separate_rocks()')
            if num_agglomerates:
                self.agglomerate_masks = st.binary_search_agglomerates(
                    num_agglomerates, min_obj_size, self.optimized_rock_mask
                    )
            elif ML:
                # start from 5 to overcome under-segmentation
                # To avoid over-segmentation: set the max guess (15)
                model = CompactUNet3D(n_channels=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.load_state_dict(
                        torch.load('./model/compact_bs32_epoch25_p.pth',
                                   map_location=torch.device('cpu')))
                print('\t -- model loaded successfully')
                guess = 5
                self.agglomerate_masks = (
                    mlTools.predicting.recursive_agglomerate_search(
                        guess, min_obj_size, self.optimized_rock_mask, model))

            else:
                self.agglomerate_masks = st.separate_rocks(
                    self.optimized_rock_mask,
                    suppress_percentage=suppress_percentage,
                    min_obj_size=min_obj_size)
            if save:
                for i, agglomerate_mask in enumerate(self.agglomerate_masks):
                    tifffile.imwrite(
                        self.image_folder + self.identifier +
                        f'_agglomerates_mask_{i}.tif',
                        agglomerate_mask)

    def agglomerate_extraction(self, min_obj_size: int = 1000,
                               del_attr: bool = False, save: bool = False):
        """
        Extract individual agglomerates from the segmented rocks.

        This method extracts individual agglomerates from the segmented rocks
        by applying the `st.agglomerate_extraction` function. It iterates over
        the agglomerate masks generated by `separate_rocks()` and extracts the
        fragments corresponding to each agglomerate. The extracted fragments
        can be optionally saved, and intermediate attributes can be deleted if
        specified.

        Args:
            min_obj_size (int, optional): The minimum object size for fragments
                                          to be considered during extraction.
                                          Defaults to 1000.
            del_attr (bool, optional): If True, deletes intermediate attributes
                                       after processing. Defaults to False.
            save (bool, optional): If True, saves the extracted fragments to
                                   disk. Defaults to False.
        """
        self.separate_rocks()
        if not hasattr(self, 'frags'):
            print('-----Extract Agglomerates-----')
            print('\t calling agglomerate_extraction()')
            self.frags = []
            for i, agglomerate_mask in enumerate(self.agglomerate_masks):
                print(f'\t - agglomerate {i}:')
                self.frags.append(st.agglomerate_extraction(agglomerate_mask,
                                  self.raw, min_obj_size=min_obj_size))
            if save:
                for i, frag in enumerate(self.frags):
                    tifffile.imwrite(
                        self.image_folder + self.identifier +
                        f'_frag_{i}.tif', frag.astype('uint16'))

    def threshold_grain(self, method: Literal['entropy', 'moments'],
                        del_attr: bool = False, save: bool = False):
        """
        Apply a grain thresholding method to the extracted agglomerates.

        This method thresholds the grains within each extracted agglomerate
        using the specified method (`entropy` or `moments`). It computes the
        threshold for each fragment and applies it to create binary masks
        representing different grain phases. The results can be optionally
        saved, and intermediate attributes can be deleted if specified.

        Args:
            method (str): The method to use for grain thresholding, either
                          'entropy' or 'moments'.
            del_attr (bool, optional): If True, deletes attributes not used in
                                       this and subsequent functions.
                                       Defaults to False.
            save (bool, optional): If True, saves the computed grain thresholds
                                   and masks to disk. Defaults to False.
        """
        self.agglomerate_extraction()
        if not hasattr(self, 'grain_threshs'):
            print('-----Finding Grain Threshold')
            if del_attr:
                del self.optimized_rock_mask
                gc.collect()

            self.grain_threshs = []
            self.grain_thresh_masks = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if method == 'entropy':
                    print('\t calling th_entropy_lesf()')
                    for i, frag in enumerate(self.frags):
                        print(f'\t - agglomerate {i}:')
                        self.grain_threshs.append(thresh.th_entropy_lesf(frag))
                elif method == 'moments':
                    print('\t calling th_moments()')
                    for i, frag in enumerate(self.frags):
                        print(f'\t - agglomerate {i}:')
                        self.grain_threshs.append(thresh.th_moments(frag))

                for i, grain_thresh in enumerate(self.grain_threshs):
                    grain_thresh_mask = np.zeros_like(self.frags[i], dtype=int)
                    grain_thresh_mask[self.frags[i] > grain_thresh] = 2
                    grain_thresh_mask[(self.frags[i] > 0) &
                                      (self.frags[i] < grain_thresh)] = 1
                    grain_thresh_mask[self.frags[i] == 0] = 0
                    self.grain_thresh_masks.append(grain_thresh_mask)

            if save:
                for i, grain_thresh in enumerate(self.grain_threshs):
                    with open(self.threshold_folder + self.identifier +
                              f'_grain_thresh_{i}.txt', 'w') as file:
                        file.write(str(grain_thresh))
                    dio.grain_mask_write(
                        self.grain_thresh_masks[i],
                        self.image_folder + self.identifier +
                        f'_grain_thresh_mask_{i}.tif'
                        )
