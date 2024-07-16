import ToolKit4D.dataio as dio
import ToolKit4D.thresholding as thresh
import ToolKit4D.utils as ut
import ToolKit4D.stages as st
import os
import gc
import warnings
import scipy.io

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
    def __init__(self, rawfile):
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

    def _read_raw(self):
        raw = dio.read_raw(self.rawfile, self.im_size, self.im_type)
        return raw

    def threshold_rock(self, del_attr: bool = False, save: bool = False):
        if not hasattr(self, 'rock_thresh'):
            print('-----Finding Rock Threshold-----')
            print('\t calling threshold_rock()')
            self.rock_thresh = thresh.threshold_rock(raw_image=self.raw)
            self.rock_thresh_mask = self.raw >= self.rock_thresh
            if save:
                scipy.io.savemat(
                    './results/' + self.identifier + '_rock_thresh_mask.mat',
                    {'rock_thresh_mask': self.rock_thresh_mask})
                with open('./results/' + self.identifier +
                          '_rock_thresh.txt', 'w') as file:
                    file.write(str(self.rock_thresh))

    def remove_cylinder(self, ring_rad: int = 99, ring_frac: float = 1.5,
                        del_attr: bool = False, save: bool = False):
        if not hasattr(self, 'column_mask'):
            self.threshold_rock()
            print('-----Removing Cylinder-----')
            print('\t calling remove_cylinder()')
            if del_attr:
                # delattr(self, 'rock_thresh')
                del self.rock_thresh
                gc.collect()
            self.column_mask = ut.remove_cylinder(self.rock_thresh_mask,
                                                  ring_rad, ring_frac)
            if save:
                scipy.io.savemat(
                    './results/' + self.identifier + '_column_mask.mat',
                    {'_column_mask': self.column_mask})

    def segment_rocks(self, remove_cylinder: bool = True,
                      min_obj_size: int = 2, del_attr: bool = False,
                      save: bool = False):
        """
        different from Matlab code; Matlab: downsample from raw then
        thershold and remove; Here: threshold and remove then downsample
        """
        if not hasattr(self, 'optimized_rock_mask'):
            if remove_cylinder:
                self.remove_cylinder(del_attr=del_attr)
                initial_mask = self.column_mask
                if del_attr:
                    # delattr(self, 'rock_thresh_mask')
                    # delattr(self, 'column_mask')
                    del self.rock_thresh_mask
                    del self.column_mask
                    gc.collect()
            else:
                self.threshold_rock()
                initial_mask = self.rock_thresh_mask
                if del_attr:
                    # delattr(self, 'rock_thresh_mask')
                    del self.rock_thresh_mask
                    gc.collect()
            print('-----Segment Rocks-----')
            print('\t calling segment_rocks()')
            self.optimized_rock_mask = st.segment_rocks(
                initial_mask,
                min_obj_size=min_obj_size)
            if save:
                scipy.io.savemat(
                    './results/' + self.identifier +
                    '_optimized_rock_mask.mat',
                    {'optimized_rock_mask': self.optimized_rock_mask})

    def agglomerate_extraction(self, min_obj_size: int = 2,
                               del_attr: bool = False, save: bool = False):
        self.segment_rocks()
        if not hasattr(self, 'frag'):
            print('-----Extract Agglomerates-----')
            print('\t calling agglomerate_extraction()')
            self.frag = st.agglomerate_extraction(self.optimized_rock_mask,
                                                  self.raw,
                                                  min_obj_size=min_obj_size)
            if save:
                scipy.io.savemat('./results/' + self.identifier + '_frag.mat',
                                 {'frag': self.frag})

    def th_entropy_lesf(self, del_attr: bool = False, save: bool = False):
        self.agglomerate_extraction()
        if not hasattr(self, 'grain_thresh'):
            print('-----Finding Grain Threshold')
            print('\t calling th_entropy_lesf()')
            if del_attr:
                # delattr(self, 'optimized_rock_mask')
                # delattr(self, 'raw')
                del self.optimized_rock_mask
                del self.raw
                gc.collect()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.grain_thresh = thresh.th_entropy_lesf(self.frag)

            if save:
                with open('./results/' + self.identifier +
                          '_grain_thresh.txt', 'w') as file:
                    file.write(str(self.grain_thresh))
