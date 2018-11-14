import numpy as np
import h5py
# from magellan_data import MagellanJavaWrapper
import ast


def hdfdatapath(channel_name, time_index=0, position_index=0):
    """
    Method for getting the string path to image data in the HDF file
    :return:
    """
    return '/'.join(['pixel_data', 't_' + str(time_index), channel_name, 'position_' + str(position_index)]) + '/'


class MagellanHDFContainer:
    """
    This class manages an HDF5 file containing image data and metadata derived from a magellan dataset
    Also supports a set of method for manual marking of stuff (cells) using imageUI
    """
    def __init__(self, path):
        self.path = path
        #open in readwrite mode, but dont overwrite
        self.file = h5py.File(path,'r+')
        #read summary metadata
        self.summary_metadata = ast.literal_eval(self.file.attrs['summary_metadata'])
        self.rgb = self.file.attrs['RGB']
        self.bytedepth = self.file.attrs['ByteDepth']
        self.numrows = self.file.attrs['NumRows']
        self.numcols = self.file.attrs['NumCols']
        self.overlapx = self.summary_metadata['GridPixelOverlapX']
        self.overlapy = self.summary_metadata['GridPixelOverlapY']
        self.tilewidth = self.file.attrs['TileWidth']
        self.tileheight = self.file.attrs['TileHeight']
        self.pixelsizeXY_um = self.file.attrs['PixelSizeXY_um']
        self.pixelsizeZ_um = self.file.attrs['PixelSizeZ_um']
        self.channelnames = list(self.file.attrs['ChannelNames'].astype(str))
        #params for interacting with imageUI
        self.display_position_index = 0
        self.display_channel_index = 0
        self.display_slice_index = 0
        self.display_time_index = 0
        #map out position metadata
        keys = np.array(list(self.file['image_keys']))
        self.num_positions = np.max(keys[:,3]) + 1
        self.num_time_points = np.max(keys[:,2]) + 1
        self.num_channels = np.max(keys[:, 0]) + 1

    def getdisplayimage(self):
        """
        Called by ImageUI to interactively view data
        :return: tuple with pixels and string describing current tile
        """
        channelname = self.channelnames[self.display_channel_index]
        img = self.read_image(channelname, position_index=self.display_position_index,
                              relative_z_index=self.display_slice_index, time_index=self.display_time_index)
        (row, col) = self.get_row_col(self.display_position_index)
        return (img, 'Frame: {} Position: {} of {} Row: {} Col: {} Channel: {} Slice: {} of {}'.format(self.display_time_index,
                                        self.display_position_index+1, self.num_positions, row, col, channelname, self.display_slice_index + 1,
                                self.get_num_slices_at(position_index=self.display_position_index)))

    def update_display_indices(self, delta_channel=0, delta_slice=0, delta_time=0, delta_position=0):
        self.display_position_index = max(0, min(self.display_position_index + delta_position, self.num_positions - 1))
        self.display_channel_index = max(0, min(self.display_channel_index + delta_channel, len(self.channelnames) - 1))
        self.display_time_index = max(0, min(self.display_time_index + delta_time, self.num_time_points - 1))
        self.display_slice_index = max(0, min(self.display_slice_index + delta_slice,
                                              self.get_num_slices_at(self.display_position_index) - 1))

    #global String : scalar annotation
    def write_annotation(self, name, value):
        """
        store string:scalar annotation in top level
        """
        self.file.attrs[name] = value
        self.file.flush()

    def read_annotation(self, name):
        """
        read a scalar annotation from top level
        :return:
        """
        if name not in self.file.attrs:
            return None
        return self.file.attrs[name]

    #global String:array mappings
    def store_array(self, name, array):
        """
        Store a numpy array. if array of the same name already exists, overwrite it
        :param name:
        :param array:
        :return:
        """
        if name in self.file:
            #delete and remake
            del(self.file[name])
        self.file.create_dataset(name, data=array)
        self.file.flush()

    def read_array(self, name):
        """
        Return a RAM copy of previously stored numoy array
        """
        if name in self.file:
            return np.copy(self.file[name])
        return None

    #Store array data specific to a given tile
    def _tile_annotation_path(self, channel_index, slice_index, time_index, position_index):
        """
        Path in HDF file specific to currently displayed tile
        """
        if channel_index is None:
            channel_index = self.display_channel_index
        if slice_index is None:
            slice_index = self.display_slice_index
        if time_index is None:
            time_index = self.display_time_index
        if position_index is None:
            position_index = self.display_position_index

        return hdfdatapath(channel_name=self.channelnames[channel_index], time_index=time_index,
                           position_index=position_index) + 'Slice{}_annoation'.format(slice_index)

    def read_all_tile_annotations(self):
        annotations = []
        for c in range(self.num_channels):
            for t in range(self.num_time_points):
                for p in range(self.num_positions):
                    for z in range(self.get_num_slices_at(position_index=p)):
                        path = self._tile_annotation_path(channel_index=c, slice_index=z, time_index=t, position_index=p)
                        entry = self.read_array(path)
                        if entry is not None:
                            id = {'channel': c, 'z': z, 'time': t, 'position': p}
                            annotations.append((entry, id))
        return annotations

    def clear_all_tile_annotations(self):
        for c in range(self.num_channels):
            for t in range(self.num_time_points):
                for p in range(self.num_positions):
                    for z in range(self.get_num_slices_at(position_index=p)):
                        path = self._tile_annotation_path(channel_index=c, slice_index=z, time_index=t, position_index=p)
                        if path in self.file:
                            # delete and remake
                            del (self.file[path])
        self.file.flush()

    def read_tile_annotations(self, channel_index=None, slice_index=None, time_index=None, position_index=None):
        """
        Read numpy array, if channel/slice/frame/position arent supplied, defautl to currently displayed one
        :return:
        """
        hdf_path = self._tile_annotation_path(channel_index=channel_index, slice_index=slice_index,
                                              time_index=time_index, position_index=position_index)
        return self.read_array(hdf_path)

    def store_tile_annotations(self, array, channel_index=None, slice_index=None, time_index=None, position_index=None):
        """
        Store numpy array of annotations specifc to a single tile. Default to currently displayed if channel/slice/frame
        aren't supplied
        """
        hdf_path = self._tile_annotation_path(channel_index=channel_index, slice_index=slice_index,
                                                      time_index=time_index, position_index=position_index)
        self.store_array(hdf_path, array)

    def get_num_slices_at(self, position_index):
        pathinhdf = hdfdatapath(self.channelnames[0], position_index=position_index) + 'voxels'
        if pathinhdf not in self.file:
            return 0
        return self.file[pathinhdf].shape[0]

    def get_z_offset(self, position_index, channel_index=0, time_index=0):
        pathinhdf = hdfdatapath(self.channelnames[channel_index], time_index, position_index) + 'voxels'
        return self.file[pathinhdf].attrs['z_index_offset'] * self.pixelsizeZ_um

    def get_row_col(self, position_index):
        pos_md = self.summary_metadata['InitialPositionList'][position_index]
        return (pos_md['GridRowIndex'], pos_md['GridColumnIndex'])


    # def converttoglobalcoordinates(self,row,col,x,y):
    #     """
    #     Take row and column indices and x and y pixel coordinates in tile and return global coordinates
    #     """
    #     visibletilewidth = self.tilewidth - self.overlapx
    #     visibletileheight = self.tileheight - self.overlapy
    #     globalx = x + col * visibletilewidth
    #     globaly = y + row * visibletileheight
    #     return (globalx, globaly)

    def read_image(self, channel_name=None, relative_z_index=0, time_index=0, position_index=None, row_col_indices=None,
                  channel_index=None, xy_slice=None, return_metadata=False):
        """
        read a numpy array of pixels corresponding to the requested image
        :param channel_name:
        :param row:
        :param col:
        :param frameindex:
        :param relativesliceindex:
        :param position_index: ignores row and col if supplied
        :param xy_slice: index sub xy patch of image (y1,y2), (x1,x2)
        :return: numpy array of flaots of pixels
        """
        #determine channel index
        if channel_index is None and channel_name is None:
            channel_name = self.channelnames[0] #caller doesnt care about channels
        elif channel_index is not None and channel_name is None:
            channel_name = self.channelnames[channel_index]
        #determine positon index
        if position_index is None and row_col_indices is None:
            position_index = 0 #caller doesnt care about positions
        elif position_index is None and row_col_indices is not None:
            position_index = self.position_index_from_grid_coords[row_col_indices]

        pathinhdf = hdfdatapath(channel_name=channel_name, position_index=position_index, time_index=time_index) + 'voxels'
        if pathinhdf not in self.file:
            return None
        data_hypercube = self.file[pathinhdf]
        if relative_z_index < 0 or relative_z_index >= data_hypercube.shape[0]:
            raise Exception("slice index out of range")
        if xy_slice is None:
            image = data_hypercube[relative_z_index,  ...].astype(np.float32)
        else:
            image = data_hypercube[relative_z_index, max(0, xy_slice[0][0]):xy_slice[0][1], max(0, xy_slice[1][0]):xy_slice[1][1],...].astype(np.float32)
        if return_metadata:
            z_index_offset = self.get_z_offset(position_index=position_index, channel_index=self.channelnames.index(channel_name),time_index=time_index)
            return image, ast.literal_eval(str(data_hypercube.attrs['z{}_metadata'.format(z_index_offset)]))
        else:
            return image

    def close(self):
        self.file.close()