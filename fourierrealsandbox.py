from afutil import get_patch_metadata, read_or_calc_focal_planes, compile_deterministic_data,\
         feature_vector_generator_fn, MagellanWithAnnotation, plot_results, get_led_na, HDFDataWrapper
from defocusnetwork import DefocusNetwork
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


class DataWrapper:

    def __init__(self, magellan):
        self.magellan = magellan

    def read_ground_truth_image(self, position_index, z_index):
        """
        Read image in which focus quality can be measured form quality of image
        :param pos_index: index of xy position
        :param z_index: index of z slice (starting at 0)
        :param xy_slice: (cropped region of image)
        :return:
        """
        return self.magellan.read_image(channel_name='DPC_Bottom', pos_index=position_index,
                                        z_index=z_index + min(self.magellan.get_z_slices_at(position_index))).astype(
            np.float)

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
        image = self.magellan.read_image(channel_name='autofocus', pos_index=position_index, z_index=z_index +
                                                                                                     min(
                                                                                                         self.magellan.get_z_slices_at(
                                                                                                             position_index))).astype(
            np.float)
        # crop
        return image[xy_slice[0][0]:xy_slice[0][1], xy_slice[1][0]:xy_slice[1][1]]

    def get_image_width(self):
        """
        :return: image width in pixels
        """
        return self.magellan.image_width

    def get_image_height(self):
        """
        :return: image height in pixels
        """
        return self.magellan.image_height

    def get_num_z_slices_at(self, position_index):
        """
        return number of z slices (i.e. focal planes) at the given XY position
        :param position_index:
        :return:
        """
        return len(self.magellan.get_z_slices_at(position_index))

    def get_pixel_size_z_um(self):
        """
        :return: distance in um between consecutive z slices
        """
        return self.magellan.pixel_size_z_um

    def get_num_xy_positions(self):
        """
        :return: total number of xy positons in data set
        """
        return self.magellan.get_num_xy_positions()

    def store_focal_plane(self, name, focal_position):
        """
        Store the computed focal plane as a string, float pair
        """
        self.magellan.write_annotation(name, focal_position)

    def read_focal_plane(self, name):
        """
        read a previously computed focal plane
        :param name: key corresponding to an xy position for whch focal plane has already been computed
        :return:
        """
        return self.magellan.read_annotation(name)

    def store_array(self, name, array):
        """
        Store a numpy array containing the design matrix for training the non-deterministic part of the network (i.e.
        after the Fourier transform) so that it can be retrained quickly without having to recompute
        :param name:
        :param array: (n examples) x (d feature length) numpy array
        """
        self.magellan.store_array(name, array)

    def read_array(self, name):
        """
        Read and return a previously computed array
        :param name:
        :return:
        """
        return self.magellan.read_array(name)

# parameters for the deterministic part of the network
deterministic_params = {'non_led_width': 0.1, 'led_width': 0.6, 'tile_split_k': 2}

# load data
data = DataWrapper(MagellanWithAnnotation(
    '/media/hugespace/henry/data/deepaf2/2018-9-27 Cells and histology af data/Neomounted cells 12x12 30um range 1um step_1'))

# load or compute target focal planes using 22 CPU cores to speed computation
focal_planes = {dataset: read_or_calc_focal_planes(dataset, split_k=deterministic_params['tile_split_k'],
                                                   n_cores=22, show_output=True) for dataset in [data]}

# split cell data into training and validation sets
num_pos = data.get_num_xy_positions()
train_positions = list(range(int(num_pos * 0.9)))
validation_positions = list(range(max(train_positions) + 1, num_pos))

# Compute or load already computed design matrices
train_features, train_targets = compile_deterministic_data([data], [train_positions], focal_planes,
                                                           deterministic_params=deterministic_params)
validation_features, validation_targets = compile_deterministic_data([data], [validation_positions],
                                                                     focal_planes,
                                                                     deterministic_params=deterministic_params)

#make genenrator function for providing training examples and seperate validation generator for assessing its progress
#stop training once error on validation set stops decreasing
train_generator = feature_vector_generator_fn(train_features, train_targets, mode='all',
                                        split_k=deterministic_params['tile_split_k'])
val_generator = feature_vector_generator_fn(validation_features, validation_targets, mode='all',
                                        split_k=deterministic_params['tile_split_k'])


#feed in the dimensions of the cropped input so the inference network knows what to expect
#although the inference network is not explicitly used in this notebook, it is created so that the model tensforflow
#creates could later be used on real data
patch_size, patches_per_image = get_patch_metadata((data.get_image_width(),
                                        data.get_image_height()), deterministic_params['tile_split_k'])

#Create network and train it
with DefocusNetwork(input_shape=train_features.shape[1], train_generator=train_generator,
                             val_generator=val_generator, predict_input_shape=[patch_size, patch_size],
                             deterministic_params=deterministic_params, train_mode='train') as network:
    #run training set and both valdation sets through network to generate predictions
    train_prediction_defocus, train_target_defocus = network.predict(train_generator)
    val_prediction_defocus, val_target_defocus = network.predict(val_generator)

# plt.figure(figsize=(14,11))
# plot_results(train_prediction_defocus, train_target_defocus)
# plot_results(val_prediction_defocus, val_target_defocus, draw_rect=True)
# plt.legend(['Training set',  'Validation set','Ground truth', 'Objective depth of focus'])
# print('Training data RMSE: {}'.format( np.sqrt(np.mean((train_prediction_defocus - train_target_defocus) ** 2))))
# print('Validation data RMSE: {}'.format(np.sqrt(np.mean((val_prediction_defocus - val_target_defocus) ** 2))))