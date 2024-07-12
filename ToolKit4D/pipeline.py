import ToolKit4D.dataio as dio
import ToolKit4D.thresholding as thresh
import ToolKit4D.utils as ut
import ToolKit4D.stages as st
import os
import warnings

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

    def threshold_rock(self):
        if not hasattr(self, 'rock_thresh'):
            print('-----Finding Rock Threshold-----')
            print('\t calling threshold_rock()')
            self.rock_thresh = thresh.threshold_rock(raw_image=self.raw)
            self.rock_thresh_mask = self.raw >= self.rock_thresh

    def remove_cylinder(self, ring_rad: int = 99, ring_frac: float = 1.5):
        if not hasattr(self, 'column_mask'):
            self.threshold_rock()
            print('-----Removing Cylinder-----')
            print('\t calling remove_cylinder()')
            # delattr(self, 'rock_thresh')
            self.column_mask = ut.remove_cylinder(self.rock_thresh_mask,
                                                  ring_rad, ring_frac)

    def segment_rocks(self, remove_cylinder: bool = True):
        """
        different from Matlab code; Matlab: downsample from raw then
        thershold and remove; Here: threshold and remove then downsample
        """
        if not hasattr(self, 'optimized_rock_mask'):
            if remove_cylinder:
                self.remove_cylinder()
                initial_mask = self.column_mask
                # delattr(self, 'rock_thresh_mask')
                # delattr(self, 'column_mask')
            else:
                self.threshold_rock()
                initial_mask = self.rock_thresh_mask
                # delattr(self, 'rock_thresh_mask')
            print('-----Segment Rocks-----')
            print('\t calling segment_rocks()')
            self.optimized_rock_mask = st.segment_rocks(initial_mask)

    def agglomerate_extraction(self):
        self.segment_rocks()
        if not hasattr(self, 'frag'):
            print('-----Extract Agglomerates-----')
            print('\t calling agglomerate_extraction()')
            self.frag = st.agglomerate_extraction(self.optimized_rock_mask,
                                                  self.raw)

    def th_entropy_lesf(self):
        self.agglomerate_extraction()
        if not hasattr(self, 'grain_thresh'):
            print('-----Finding Grain Threshold')
            print('\t calling th_entropy_lesf()')
            # delattr(self, 'optimized_rock_mask')
            # delattr(self, 'raw')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.grain_thresh = thresh.th_entropy_lesf(self.frag)
