import ToolKit4D.dataio as dio
import ToolKit4D.thresholding as thresh
import ToolKit4D.utils as ut
import ToolKit4D.stages as st
import os
import gc
import warnings
import numpy as np
import tifffile
from typing import Literal

# 1. write the basic structure
# 2. add clean ram inside class and compare the saving of ram
# 3. add lines to write intermediate result into disk
# 4. call one functoin to execute all previous function
# 5. pay attension: some function will change the variable inside
#    - so for those i want to keep; pass copy to function
# 6. add option to load disk data stored at (3) to each funciton
#    - so no need to run 'previous' function again
#    - but increase time for loading data
# 7. To avoid executing multiple times when execute previous methods;
#    use hasattr(self, 'property')
#    - if call at once; default parameter
#    - if call separatly: user parameter can set


class ToolKitPipeline:
    """_summary_: processing per image per instance
    """
    def __init__(self, rawfile, load: bool = False):
        """_summary_

        Args:
            rawfile_path (_type_): complete path to raw file
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

    def initialize(self):
        """
        Removes all attributes of the instance except those defined
        in __init__. Used when you want to try other methods parameters
        but no need to reload raw image.
        """
        init_attrs = {'rawfile', 'identifier', 'im_size', 'im_type', 'raw',
                      'image_folder', 'threshold_folder'}
        attrs = set(self.__dict__.keys())
        for attr in attrs - init_attrs:
            delattr(self, attr)
        gc.collect()

    def _load_saved_files(self):
        """Load all saved files into corresponding attributes"""
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
        self.agglomerate_masks = []
        i = 0
        while True:
            agglomerate_mask_path = os.path.join(self.image_folder,
                                                 self.identifier +
                                                 f'_agglomerates_mask_{i}.tif')
            if os.path.exists(agglomerate_mask_path):
                self.agglomerate_masks.append(dio.tif_read(
                    agglomerate_mask_path))
                i += 1
            else:
                break

        # Load frags
        self.frags = []
        i = 0
        while True:
            frag_path = os.path.join(self.image_folder,
                                     self.identifier +
                                     f'_frag_{i}.tif')
            if os.path.exists(frag_path):
                self.frags.append(dio.tif_read(frag_path))
                i += 1
            else:
                break

        # Load grain_threshs and grain_thresh_masks
        self.grain_threshs = []
        self.grain_thresh_masks = []
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
                    self.grain_threshs.append(float(file.read().strip()))
                self.grain_thresh_masks.append(dio.grain_mask_read(
                    grain_thresh_mask_path))
                i += 1
            else:
                break

    def _read_raw(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        raw = dio.read_raw(self.rawfile, self.im_size, self.im_type)
        return raw

    def threshold_rock(self, del_attr: bool = False, save: bool = False):
        """_summary_

        Args:
            del_attr (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
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

    def remove_cylinder(self, ring_rad: int = 99, ring_frac: float = 1.5,
                        del_attr: bool = False, save: bool = False):
        """_summary_

        Args:
            ring_rad (int, optional): _description_. Defaults to 99.
            ring_frac (float, optional): _description_. Defaults to 1.5.
            del_attr (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
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
                      min_obj_size: int = 2, del_attr: bool = False,
                      save: bool = False):
        """
        different from Matlab code; Matlab: downsample from raw then
        thershold and remove; Here: threshold and remove then downsample

        Args:
            remove_cylinder (bool, optional): _description_. Defaults to True.
            min_obj_size (int, optional): _description_. Defaults to 2.
            del_attr (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
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
                       del_attr: bool = False, save: bool = False):
        """_summary_

        Args:
            suppress_percentage (int, optional): _description_. Defaults to 10.
            del_attr (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
        """
        self.segment_rocks()
        if not hasattr(self, 'agglomerate_masks'):
            print('-----Separate Agglomerates-----')
            print('\t calling separate_rocks()')
            self.agglomerate_masks = st.separate_rocks(
                self.optimized_rock_mask,
                suppress_percentage=suppress_percentage)
            if save:
                for i, agglomerate_mask in enumerate(self.agglomerate_masks):
                    tifffile.imwrite(
                        self.image_folder + self.identifier +
                        f'_agglomerates_mask_{i}.tif',
                        agglomerate_mask)

    def agglomerate_extraction(self, min_obj_size: int = 2,
                               del_attr: bool = False, save: bool = False):
        """_summary_

        Args:
            min_obj_size (int, optional): _description_. Defaults to 2.
            del_attr (bool, optional): _description_. Defaults to False.
            save (bool, optional): _description_. Defaults to False.
        """
        self.separate_rocks()
        if not hasattr(self, 'frags'):
            print('-----Extract Agglomerates-----')
            print('\t calling agglomerate_extraction()')
            self.frags = []
            for i, agglomerate_mask in enumerate(self.agglomerate_masks):
                print('-\t agglomerate {i}:')
                self.frags.append(st.agglomerate_extraction(agglomerate_mask,
                                  self.raw, min_obj_size=min_obj_size))
            if save:
                for i, frag in enumerate(self.frags):
                    tifffile.imwrite(
                        self.image_folder + self.identifier +
                        f'_frag_{i}.tif', frag.astype('uint16'))

    def threshold_grain(self, method: Literal['entropy', 'moments'],
                        del_attr: bool = False, save: bool = False):
        """_summary_

        Args:
            method (str): either 'entropy' or 'moments'
            del_attr (bool, optional): _description_. delete attributes not used in this and the following functions
            Defaults to False.
            save (bool, optional): _description_. Defaults to False.
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
                        print('-\t agglomerate {i}:')
                        self.grain_threshs.append(thresh.th_entropy_lesf(frag))
                elif method == 'moments':
                    print('\t calling th_moments()')
                    for i, frag in enumerate(self.frags):
                        print('-\t agglomerate {i}:')
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
