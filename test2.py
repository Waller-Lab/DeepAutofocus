from afutil import get_patch_metadata, read_or_calc_focal_planes, compile_deterministic_data,\
    feature_vector_generator_fn, MagellanWithAnnotation, plot_results, get_led_na, HDFDataWrapper
from defocusnetwork import DefocusNetwork
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

deterministic_params = {'non_led_width': 0.1, 'led_width': 0.6, 'tile_split_k': 2}

# load data
train_datasets = [
    HDFDataWrapper(
        '/media/hugespace/henry/data/deepaf2/2018-8-21 Slide 1-10/slide 1-10 seconds mid acquisition autofocus_1.hdf'),
    # HDFDataWrapper('/media/hugespace/henry/data/deepaf2/2018-8-26 Slide set 2 af training/Slide 2-0a 6x6 30um_1.hdf'),
    # HDFDataWrapper(
    #     '/media/hugespace/henry/data/deepaf2/2018-8-26 Slide set 2 af training/Slide 2-0a 6x6 30um again_1.hdf'),
    # HDFDataWrapper('/media/hugespace/henry/data/deepaf2/2018-8-26 Slide set 2 af training/Slide 2-0c 6x6 30um_1.hdf'),
    # HDFDataWrapper(
    #     '/media/hugespace/henry/data/deepaf2/2018-8-26 Slide set 2 af training/Slide 2-0c 6x6 30um again_1.hdf'),
    # HDFDataWrapper('/media/hugespace/henry/data/deepaf2/2018-8-26 Slide set 2 af training/Slide 2-0d 6x6 30um_1.hdf'),
    # HDFDataWrapper(
    #     '/media/hugespace/henry/data/deepaf2/2018-8-26 Slide set 2 af training/Slide 2-0d 6x6 30um again_1.hdf'),
    # HDFDataWrapper(
    #     '/media/hugespace/henry/data/deepaf2/2018-8-1 A bit more pre collection af/300k 7x7 30um range 50ms_1.hdf'),
    # HDFDataWrapper(
    #     '/media/hugespace/henry/data/deepaf2/2018-8-1 A bit more pre collection af/75k 7x7 30um range 50ms_1.hdf')
]

test_dataset = HDFDataWrapper(
    '/media/hugespace/henry/data/deepaf2/2018-8-21 Slide 1-10/slide 1-10 third mid acquisition autofocus_1.hdf')

# load or compute target focal planes using 22 CPU cores to speed computation
focal_planes = {dataset: read_or_calc_focal_planes(dataset, split_k=deterministic_params['tile_split_k'],
                n_cores=22, show_output=True) for dataset in [*train_datasets, test_dataset]}

train_positions = [list(range(int(dataset.get_num_xy_positions() * 0.9))) for dataset in train_datasets]
validation_positions = [list(range(int(dataset.get_num_xy_positions() * 0.9),
                                  dataset.get_num_xy_positions())) for dataset in train_datasets]
# Compute or load already computed design matrices
train_features, train_targets = compile_deterministic_data(train_datasets, train_positions,
                           focal_planes, deterministic_params=deterministic_params)
validation_features, validation_targets = compile_deterministic_data(train_datasets,
                           validation_positions, focal_planes, deterministic_params=deterministic_params)
# test dataset is entirely different acquisition
test_features, test_targets = compile_deterministic_data([test_dataset],
          list(range(test_dataset.get_num_xy_positions())), focal_planes, deterministic_params=deterministic_params)

train_generator = feature_vector_generator_fn(train_features, train_targets, mode='all',
                                              split_k=deterministic_params['tile_split_k'])
val_generator = feature_vector_generator_fn(validation_features, validation_targets, mode='all',
                                            split_k=deterministic_params['tile_split_k'])
test_generator = feature_vector_generator_fn(test_features, test_targets, mode='all',
                                             split_k=deterministic_params['tile_split_k'])

# feed in the dimensions of the cropped input so the inference network knows what to expect
# although the inference network is not explicitly used in this notebook, it is created so that the model tensforflow
# creates could later be used on real data
patch_size, patches_per_image = get_patch_metadata((test_dataset.get_image_width(),
                                test_dataset.get_image_height()), deterministic_params['tile_split_k'])

# Create network and train it
defocus_prediction_network = DefocusNetwork(input_shape=train_features.shape[1], train_generator=train_generator,
                                            val_generator=val_generator, predict_input_shape=[patch_size, patch_size],
                                            deterministic_params=deterministic_params, train_mode='train')

# run training set and both valdation sets through network to generate predictions
train_prediction_defocus, train_target_defocus = defocus_prediction_network.predict(train_generator)
test_prediction_defocus, test_target_defocus = defocus_prediction_network.predict(test_generator)

plt.figure(figsize=(14, 11))
plot_results(train_prediction_defocus, train_target_defocus)
plot_results(test_prediction_defocus, test_target_defocus)
plt.legend(['Training set', 'Test set', 'Ground truth', '20x 0.5NA objective depth of focus'])
print('Training data (cells) RMSE: {}'.format(
    np.sqrt(np.mean((train_prediction_defocus - train_target_defocus) ** 2))))
print('Validation data (cells) RMSE: {}'.format(
    np.sqrt(np.mean((test_prediction_defocus - test_target_defocus) ** 2))))

plt.show()
os.mkdir('figures')
plt.savefig('figures/good_test_performance.eps')