import ToolKit4D.dataio as dio
import ToolKit4D.thresholding as thresh
import ToolKit4D.utils as ut
import os


# 1. write the basic structure
# 2. add clean ram inside class and compare the saving of ram
# 3. add lines to write intermediate result into disk
# 4. call one functoin to execute all previous function
# 5. pay attension: some function will change the variable inside
#    - so for those i want to keep; pass copy to function
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
        self.rock_thresh = thresh.threshold_rock(raw_image=self.raw)
        self.mask = self.raw >= self.rock_thresh

    def remove_cylinder(self, ring_rad: int = 99, ring_frac: float = 1.5):
        self.threshold_rock()
        self.mask = ut.remove_cylinder(self.mask, ring_rad, ring_frac)

    def segment_rocks(self, remove_cylinder: bool = True):
        # both set value for attribute: self.mask
        if remove_cylinder:
            self.remove_cylinder()
        else:
            self.threshold_rock()
        # then downample self.mask
        # this is different from Matlab code; in matlab; it downsample from raw
        # then do the threshold rock (for mask) and remove cylinder
        self.optimized_mask = ut.segment_rocks(self.mask)

    def plot(self, slice):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(self.optimized_mask[:, :, slice])
        axes[1].imshow(self.mask[:, :, 4*slice])
