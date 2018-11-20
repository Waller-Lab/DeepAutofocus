import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from joblib import Parallel, delayed

from defocusnetwork import DefocusNetwork
from imageprocessing import radialaverage
from pygellan import MagellanDataset
import h5py
import os
from magellanhdf import MagellanHDFContainer
import json


def get_patch_metadata(image_size, split_k):
    """
    Split up raw image from sensor into sub patches for training network.
    :param image_size: tuple with (image width, image height)
    :param split_k: number of sub images to split into along each dimension (i.e. split_k=2 gives 4 sub images)
    :return: pixel dimension of patches (they are square and a power of two), number of patches from each raw image
    """
    shape = min(image_size)
    patch_size = 2**int(np.log2(shape/split_k))
    # patch_size = int(shape / split_k)
    patches_per_image = (shape // patch_size) **2
    return patch_size, patches_per_image

def calc_focal_plane(data, position_index, split_k, parallel=None, show_output=False):
    """
    Calculate radially averaged power spectrum of images at different focal postitions, and take the mean of high
    frequencies to measure focus qaulity. Then use these measurements to compute the optimal focal plane
    :param data: implementation of DataWrapper class
    :param position_index:
    :param split_k:
    :param parallel: if supplied use multiple threads to speed up power spectrum computations
    :param show_output if supplied, create a plot showing the calculation of th ground truth focal plane
    :return:
    """
   # print("\rCalculating focal plane, position {} of {}   ".format(position_index, data.get_num_xy_positions()),end='')

    def crop(image, index, split_k):
        """
        Crop raw image to appropriate patch size
        :return: One sub crop
        """
        y_tile_index = index // split_k
        x_tile_index = index % split_k
        return image[y_tile_index * patch_size:(y_tile_index + 1) * patch_size, x_tile_index * patch_size:(x_tile_index + 1) * patch_size]

    def calc_power_spectrum(image):
        """
        :return: Raidally averaged log power spectrum
        """
        pixelsft = np.fft.fftshift(np.fft.fft2(image))
        powerspectrum = pixelsft * pixelsft.conj()
        logpowerspectrummag = np.log(np.abs(powerspectrum))
        return radialaverage(logpowerspectrummag)

    def compute_focal_plane(powerspectralist):
        """
        Compute focal plane from a list of radially averaged power spectra, interpolating to get sub-z spacing percision
        :param powerspectralist: list of radially averaged power spectra
        :return:
        """
        powerspectra_arr = np.array(powerspectralist)
        # take sum of log power spectra (lower half
        pssum = np.sum(powerspectra_arr[:, powerspectra_arr.shape[1] // 4:], axis=1)
        # interpolate to find non integer best focal plane
        interpolant = interpolate.interp1d(np.arange(pssum.shape[0]), pssum, kind='cubic')
        xx = np.linspace(0, pssum.shape[0] - 1, 10000)
        yy = interpolant(xx)
        if show_output:
            plt.figure(1)
            plt.plot(xx * data.get_pixel_size_z_um(), yy)
            plt.plot(np.arange(pssum.shape[0]) * data.get_pixel_size_z_um(), pssum, 'o')
            plt.xlabel('Focal position (um)')
            plt.ylabel('High frequency content')
        return xx[np.argmax(yy)] * data.get_pixel_size_z_um()

    patch_size, patches_per_image = get_patch_metadata((data.get_image_width(), data.get_image_height()), split_k)
    num_crops = split_k**2

    radial_avg_power_spectrum = lambda image: calc_power_spectrum(crop(image, 0, 1))

    num_slices = data.get_num_z_slices_at(position_index)
    #load images
    images = [data.read_ground_truth_image(z_index=slice, position_index=position_index)
                for slice in range(num_slices)]
    if parallel is None:
        powerspectra = [radial_avg_power_spectrum(image) for image in images]
    else:
        powerspectra = parallel(delayed(radial_avg_power_spectrum)(image) for image in images)

    #Use same focal plane for all crops
    focal_plane = compute_focal_plane(powerspectra)
    best_focal_planes = {crop_index: focal_plane for crop_index in range(num_crops)}
    print("\rCalculated focal plane, position {} of {}: {:.3f}".format(position_index, 
                                                             data.get_num_xy_positions(),focal_plane),end='')
    return best_focal_planes

def generator_fn(data_wrapper_list, focal_planes, tile_split_k, position_indices_list, ignore_first_slice=False):
    """
    Function that generates pairs of training images and defocus distances used for training defocus prediction network
    :param data_wrapper_list list of DataWrappers
    :param focal_planes nested dict with DataWrapper, position index, and crop index as keys
    :param tile_split_k number of crops to divide each image into for training
    :param position_indices_list list same length as data_wrapper_list that has list of position indices to use for each
    dataset
    :param ignore_first_slice discard the top z slice (which was often not in the focal positon it was supposed to
    be on the system we used for testing)
    and true focal plane position as values
    :yield: dictionary with LED name key and image value for a random slice/position among
    valid slices and in the set of positions we specified
    """
    for data_wrapper, position_indices in zip(data_wrapper_list, position_indices_list):
        dataset_slice_pos_tuples = []
        #get all slice index position index combinations
        for pos_index in position_indices:
            slice_indices = np.arange(data_wrapper.get_num_z_slices_at(position_index=pos_index))
            for z_index in slice_indices:
                if z_index == 0 and ignore_first_slice:
                    continue
                dataset_slice_pos_tuples.append((data_wrapper, z_index, pos_index))
    print('{} sliceposition tuples'.format(len(dataset_slice_pos_tuples)),end='')
    indices = np.arange(len(dataset_slice_pos_tuples))

    def inner_generator(indices, focal_planes):
        patch_size, patches_per_image = get_patch_metadata((dataset_slice_pos_tuples[0][0].get_image_width(),
                                    dataset_slice_pos_tuples[0][0].get_image_height()), tile_split_k)
        for index in indices:
            data_wrapper, z_index, pos_index = dataset_slice_pos_tuples[index]
            for patch_index in range(patches_per_image):
                single_led_images = read_patch(data_wrapper, pos_index=pos_index, z_index=z_index,
                                               split_k=tile_split_k, patch_index=patch_index)

                defocus_dist = focal_planes[data_wrapper][pos_index][patch_index] - \
                               data_wrapper.get_pixel_size_z_um()*z_index
                yield single_led_images, defocus_dist
    return lambda: inner_generator(indices, focal_planes)

def feature_vector_generator_fn(feature_vectors, defocus_dists, mode, split_k, training_fraction=0.8):
    """
    Generator function feature vectors (i.e the part of the Fourier transform that feeds into trainable layers of network)
    :param feature_vectors: 2d numpy array (n x feature vector length)
    :param defocus_dists: numpy array of defocus distances
    :param mode: 'training', 'validation', or 'all'
    :param split_k: number of crops to split data into
    :param training_fraction: fraction of data to use in training set
    :return: generator function that gives one feture vector-defocus distance pair at a time
    """
    n = feature_vectors.shape[0]
    #Split every XY position crop completely into training or validation so they represent different image content
    n_full = n / (split_k**2)
    full_indices = np.arange(n_full)
    np.random.shuffle(full_indices)
    num_train = int(len(full_indices) * training_fraction)
    if mode == 'trianing':
        full_indices = full_indices[:num_train]
    elif (mode == 'validation'):
        full_indices = full_indices[num_train:]
    elif (mode == 'all'):
        pass
    #get actual data indices
    splits_per_tile = split_k**2
    data_indices = np.concatenate([np.arange(splits_per_tile*index, splits_per_tile*(index+1)) for index in full_indices]).astype(np.int32)
    if mode == 'training':
        np.random.seed(123)
        np.random.shuffle(data_indices)
    #not sure if this is absolutely needed but just in case...
    # feature_vectors = np.copy(feature_vectors)
    # defocus_dists = np.copy(defocus_dists)
    def inner_generator(linescans, defocus_dists, indices):
        #yield data in a shuffled order
        for index in indices:
            yield linescans[index, :], defocus_dists[index]

    return lambda: inner_generator(feature_vectors, defocus_dists, data_indices)

def read_patch(data_wrapper, pos_index, z_index, split_k, patch_index):
    """
    Crop a square region out of larger image for netwrok training
    :param data_wrapper:
    :param pos_index: index of XY position
    :param z_index: z slice index
    :param split_k: number of crops along each dimension
    :param patch_index: index of the crop
    :return: 2d numpy array of floats corresponding to image patch
    """

    return data_wrapper.read_prediction_image(position_index=pos_index, z_index=z_index,
                                              patch_index=patch_index, split_k=split_k)

def read_or_calc_focal_planes(data_wrapper, split_k, n_cores=1, show_output=False):
    """
    Compute or load pre computed focal planes for each XY position
    :param data_wrapper:
    :param split_k: splits per image
    :param n_cores: number of threads to use for parallelization using joblib. If set to 1
    parallelization not used
    :return:
    """
    def get_name(pos_index):
        return 'pos{}_focal_plane'.format(pos_index)

    def read_or_compute(pos_index, parallel=None):
        if data_wrapper.read_focal_plane(get_name(pos_index)) is None:
            #calculate and save it
            focal_plane = calc_focal_plane(data_wrapper, pos_index, split_k=split_k, parallel=parallel, show_output=show_output)
            for crop_index in focal_plane.keys():
                data_wrapper.store_focal_plane(get_name(pos_index), focal_plane[crop_index])
        else:
            print('Reading precomputed focal plane pos index {} of {}\r'.format(pos_index + 1,
                                                                     data_wrapper.get_num_xy_positions()), end='')
            #read saved value from previous computation
            focal_plane = {}
            for crop_index in range(split_k**2):
                focal_plane[crop_index] = data_wrapper.read_focal_plane(get_name(pos_index))
        return focal_plane

    if n_cores == 1:
        #single threaded execution
        focal_planes = {pos_index: read_or_compute(pos_index=pos_index) for pos_index in range(data_wrapper.get_num_xy_positions())}
    else:
        #parallelized
        with Parallel(n_jobs=n_cores) as parallel:
            focal_planes = {pos_index: read_or_compute(pos_index=pos_index, parallel=parallel) for pos_index
                            in range(data_wrapper.get_num_xy_positions())}

    return focal_planes

def read_or_calc_design_mat(data_wrapper, position_indices, focal_planes, deterministic_params):
    """
    Load a precomputed design matrix, or use the DefoucusNetwork class to compute it and then store for future use. The
    design matrix corresponds to the 'determninstic' beginning part of the graph (i.e. the Fourier transform)
    :param data_wrapper:
    :param position_indices
    :param focal_planes:
    :param deterministic_params: dictionary of parameters describing the structure of the network
    :return:
    """
    param_id_string = 'new' + str(deterministic_params) + str(position_indices[0]) + '_' + str(len(position_indices))
    # compute or read from storage deterministic outputs
    feature_name = 'features_' + param_id_string
    defocus_name = 'defocus_dists_' + param_id_string
    features = data_wrapper.read_array(feature_name)
    defocus_dists = data_wrapper.read_array(defocus_name)
    if features is None:
        patch_size, patches_per_image = get_patch_metadata((data_wrapper.get_image_width(),
                                                data_wrapper.get_image_height()), deterministic_params['tile_split_k'])
        generator = generator_fn([data_wrapper], focal_planes, tile_split_k=deterministic_params['tile_split_k'],
                             position_indices_list=[position_indices], ignore_first_slice=True)
        #Use the deterministic part of the network only to compute design matrix
        with DefocusNetwork(input_shape=(patch_size, patch_size), train_generator=generator,
                                     deterministic_params=deterministic_params) as network:
            features, defocus_dists = network.evaluate_deterministic_graph()
        data_wrapper.store_array(feature_name, features)
        data_wrapper.store_array(defocus_name, defocus_dists)
    return features, defocus_dists

def compile_deterministic_data(data_wrapper_list, postion_indices_list, focal_planes, deterministic_params):
    """
    For all hdf wrappers in data, load design matrix and targets and concatenate them
    Puts the data that has already been fourier transformed and flattened into design matrix
    Computes this using a deterministic neural network if needed, otherwise loads it from the file
    to save time
    :param data_wrapper_list list of DataWrapper objects to compute on
    :param postion_indices_list corresponding list of position indices to use from each one
    """
    deterministic_train_data = [read_or_calc_design_mat(dataset, position_indices, focal_planes,
                deterministic_params) for dataset, position_indices in zip(data_wrapper_list, postion_indices_list)]
    # collect training data from all experiments
    features = []
    targets = []
    for f, t in deterministic_train_data:
        features.append(f)
        targets.append(t)

    #pool all data together
    targets = np.concatenate(targets)
    features = np.concatenate(features)
    if np.any(np.isnan(features)):
        raise Exception('NAN detected in deterministic data')
    return features, targets

def plot_results(pred, target, color, draw_rect=False):
    #don't plot too many points
    indices = np.arange(pred.shape[0])
    np.random.shuffle(indices)
    plt.scatter(target[indices[:500]], pred[indices[:500]], marker='o', c=color, linewidths=0, edgecolors=None)
    plt.xlabel('True defocus (µm)')
    plt.ylabel('Predicted defocus (µm)')
    if draw_rect:
        min_target = np.min(target)
        max_target = np.max(target)
        height = (max_target - min_target)*np.sqrt(2)
        width = 2
        plt.gca().add_patch(mpatches.Rectangle([min_target, min_target+width/np.sqrt(2)], width, height,
                                               angle=-45, color=[0, 1, 0, 0.2]))
#         plt.plot([min_target, max_target], [min_target, max_target], 'g-')

def cartToNa(point_list_cart, z_offset=8):
    """functions for calcuating the NA of an LED on the quasi-dome based on it's index for the quasi-dome illuminator
    converts a list of cartesian points to numerical aperture (NA)

    Args:
        point_list_cart: List of (x,y,z) positions relative to the sample (origin)
        z_offset : Optional, offset of LED array in z, mm

    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (Na_x, NA_y)
    """
    yz = np.sqrt(point_list_cart[:, 1] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)
    xz = np.sqrt(point_list_cart[:, 0] ** 2 + (point_list_cart[:, 2] + z_offset) ** 2)

    result = np.zeros((np.size(point_list_cart, 0), 2))
    result[:, 0] = np.sin(np.arctan(point_list_cart[:, 0] / yz))
    result[:, 1] = np.sin(np.arctan(point_list_cart[:, 1] / xz))

    return(result)

def loadLedPositonsFromJson(file_name, z_offset=8):
    """Function which loads LED positions from a json file
    Args:
        fileName: Location of file to load
        zOffset : Optional, offset of LED array in z, mm
        micro : 'TE300B' or 'TE300A'
    Returns:
        A 2D numpy array where the first dimension is the number of LEDs loaded and the second is (x, y, z) in mm
    """
    json_data = open(file_name).read()
    data = json.loads(json_data)

    source_list_cart = np.zeros((len(data['led_list']), 3))
    x = [d['x'] for d in data['led_list']]
    y = [d['y'] for d in data['led_list']]
    z = [d['z'] for d in data['led_list']]

    source_list_cart[:, 0] = x
    source_list_cart[:, 1] = y
    source_list_cart[:, 2] = z

    source_list_na = cartToNa(source_list_cart, z_offset=z_offset)

    return source_list_na, source_list_cart

def get_led_na(led_index):
    source_list_na, source_list_cart = loadLedPositonsFromJson('quasi_dome_design.json')
    angles_xy = np.arcsin(np.abs(source_list_na))
    angle = np.arctan(np.sqrt(np.tan(angles_xy[:, 0])**2 + np.tan(angles_xy[:, 1])**2 ))
    return np.sin(angle[led_index - 1])

class MagellanWithAnnotation(MagellanDataset):
    """
    This class takes the python wrapper for a Micro-Magellan dataset, and adds in the ability to store annoations in an
    hdf5 file
    """

    def __init__(self, dataset_path):
        super().__init__(dataset_path=dataset_path)
        self.file = h5py.File(os.path.join(dataset_path, 'annotations'))

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

    def store_array(self, name, array):
        """
        Store a numpy array. if array of the same name already exists, overwrite it
        :param name:
        :param array:
        :return:
        """
        if name in self.file:
            # delete and remake
            del (self.file[name])
        self.file.create_dataset(name, data=array)
        self.file.flush()

    def read_array(self, name):
        """
        Return previously stored numoy array
        """
        if name in self.file:
            return self.file[name]
        return None

class HDFDataWrapper:
    """
    Version that reads the deprecated magellan hdf files
    """

    def __init__(self, path):
        self.hdf = MagellanHDFContainer(path)


    def read_ground_truth_image(self, position_index, z_index):
        """
        Read image in which focus quality can be measured form quality of image
        :param pos_index: index of xy position
        :param z_index: index of z slice (starting at 0)
        :param xy_slice: (cropped region of image)
        :return:
        """
        return self.hdf.read_image(channel_name='DPC_Bottom', position_index=position_index, 
                                   relative_z_index=z_index)

    def read_prediction_image(self, position_index, z_index, patch_index, split_k):
        """
        Read image used for single shot prediction (i.e. single LED image)
        :param pos_index: index of xy position
        :param z_index: index of z slice (starting at 0)
        :param split_k: number of crops along each dimension
        :param patch_index: index of the crop
        :return:
        """
        patch_size, patches_per_image = get_patch_metadata((self.get_image_width(),
                                                            self.get_image_height()), split_k)
        y_tile_index = patch_index // split_k
        x_tile_index = patch_index % split_k
        xy_slice = [[y_tile_index * patch_size, (y_tile_index + 1) * patch_size],
                    [x_tile_index * patch_size, (x_tile_index + 1) * patch_size]]
        return self.hdf.read_image(channel_name='autofocus', position_index=position_index,
                                   relative_z_index=z_index, xy_slice=xy_slice)

    def get_image_width(self):
        """
        :return: image width in pixels
        """
        return self.hdf.tilewidth

    def get_image_height(self):
        """
        :return: image height in pixels
        """
        return self.hdf.tileheight

    def get_num_z_slices_at(self, position_index):
        """
        return number of z slices (i.e. focal planes) at the given XY position
        :param position_index:
        :return:
        """
        return self.hdf.get_num_slices_at(position_index)

    def get_pixel_size_z_um(self):
        """
        :return: distance in um between consecutive z slices
        """
        return self.hdf.pixelsizeZ_um

    def get_num_xy_positions(self):
        """
        :return: total number of xy positons in data set
        """
        return self.hdf.num_positions

    def store_focal_plane(self, name, focal_position):
        """
        Store the computed focal plane as a string, float pair
        """
        self.hdf.write_annotation(name, focal_position)

    def read_focal_plane(self, name):
        """
        read a previously computed focal plane
        :param name: key corresponding to an xy position for whch focal plane has already been computed
        :return:
        """
        return self.hdf.read_annotation(name)

    def store_array(self, name, array):
        """
        Store a numpy array containing the design matrix for training the non-deterministic part of the network (i.e.
        after the Fourier transform) so that it can be retrained quickly without having to recompute
        :param name:
        :param array: (n examples) x (d feature length) numpy array
        """
        self.hdf.store_array(name, array)

    def read_array(self, name):
        """
        Read and return a previously computed array
        :param name:
        :return:
        """
        return self.hdf.read_array(name)

