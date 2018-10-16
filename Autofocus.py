import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage
from joblib import Parallel, delayed

from regressor import RegressorNetwork
from imageprocessing import radialaverage
from magellanhdf import MagellanHDFContainer


def get_patch_metadata(data, split_k):
    shape = min([data.tileheight, data.tilewidth])
    patch_size = 2**int(np.log2(shape/split_k))
    # patch_size = int(shape / split_k)
    patches_per_image = (shape // patch_size) **2
    return patch_size, patches_per_image

def calc_focal_plane(data, position_index, split_k, parallel=None):
    '''
    Calc power spectrum of phase images at different slices to determine optimal focal plane
    '''
    print("Calculating focal plane, position {} of {}".format(position_index, data.num_positions))

    def crop(image, index, split_k):
        y_tile_index = index // split_k
        x_tile_index = index % split_k
        return image[y_tile_index * patch_size:(y_tile_index + 1) * patch_size, x_tile_index * patch_size:(x_tile_index + 1) * patch_size]

    def calc_power_spectrum(image):
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
        # plt.plot(xx * data.pixelsizeZ_um, yy)
        # plt.plot(np.arange(pssum.shape[0]) * data.pixelsizeZ_um, pssum, 'o')
        # plt.xlabel('Focal position (um)')
        # plt.ylabel('High frequency content')
        return xx[np.argmax(yy)] * data.pixelsizeZ_um

    patch_size, patches_per_image = get_patch_metadata(data, split_k)
    num_crops = split_k**2

    radial_avg_power_spectrum = lambda image: calc_power_spectrum(crop(image, 0, 1))

    num_slices = data.get_num_slices_at(position_index)
    #load images
    images = [data.read_image(channel_name='DPC_Bottom', relative_z_index=slice, position_index=position_index)
                for slice in range(num_slices)]
    if parallel is None:
        powerspectra = [radial_avg_power_spectrum(image) for image in images]
    else:
        powerspectra = parallel(delayed(radial_avg_power_spectrum)(image) for image in images)

    #Use same focal plane for all crops
    focal_plane = compute_focal_plane(powerspectra)
    best_focal_planes = {crop_index: focal_plane for crop_index in range(num_crops)}
    print('focal plane: {}'.format(focal_plane))
    return best_focal_planes

def generator_fn(hdf_datas, focal_planes, mode, deterministic_params, ignore_first_slice=True, train_fraction=0.8, shuffle=True):
    """
    Function that generates data as it is needed for neural network training
    :param hdf_datas list of hdf datasets used for training
    :param led_indices: single LED images to load
    :param position_indices: which positions to draw from
    :param focal_planes nested dict with hdf datset position index and crop index as keys
    and true focal plane position as values
    :yield: dictionary with LED name key and image value for a random slice/position among
    valid slices and in the set of positions we specified
    """
    for hdf_data in hdf_datas:
        num_positions = hdf_data.num_positions
        dataset_slice_pos_tuples = []
        #get all slice index position index combinations
        for pos_index in range(num_positions):
            slice_indices = np.arange(hdf_data.get_num_slices_at(position_index=pos_index))
            for slice_index in slice_indices:
                if slice_index == 0 and ignore_first_slice:
                    continue
                dataset_slice_pos_tuples.append((hdf_data, slice_index, pos_index))
    print('{} sliceposition tuples'.format(len(dataset_slice_pos_tuples)))
    #split into training and validation
    indices = np.arange(len(dataset_slice_pos_tuples))
    if shuffle:
        np.random.shuffle(indices)
    num_train = int(len(indices) * train_fraction)
    if mode == 'all':
        pass #use them all
    elif mode == 'trianing':
        indices = indices[:num_train]
    elif (mode == 'validation'):
        indices = indices[num_train:]

    def inner_generator(indices, focal_planes, deterministic_params):
        patch_size, patches_per_image = get_patch_metadata(dataset_slice_pos_tuples[0][0], deterministic_params['tile_split_k'])
        for index in indices:
            hdf_data, slice_index, pos_index = dataset_slice_pos_tuples[index]
            for patch_index in range(patches_per_image):
                single_led_images = {led_index: read_patch(hdf_data, led_index=led_index, pos_index=pos_index,
                                            slice_index=slice_index, split_k=deterministic_params['tile_split_k'],
                                            patch_index=patch_index) for led_index in deterministic_params['led_indices']}
                #dont use ones where full coverage failed
                if focal_planes[hdf_data][pos_index][patch_index] < 3 or hdf_data.get_num_slices_at(position_index=pos_index)\
                        *hdf_data.pixelsizeZ_um - focal_planes[hdf_data][pos_index][patch_index] < 3:
                    print('skipping due to incomplete z coverage {}  position {}'.format(hdf_data.file,pos_index))
                    continue

                defocus_dist = focal_planes[hdf_data][pos_index][patch_index] - hdf_data.pixelsizeZ_um*slice_index
                yield single_led_images, defocus_dist
    return lambda: inner_generator(indices, focal_planes, deterministic_params)

def linescan_generator_fn(linescans, defocus_dists, mode, split_k, training_fraction=0.8):
    """
    Generator function for linescans
    trainig mode splits data and shuffles, validation splits but doesnt shuffle, all does nothing
    """
    #use_examples = np.abs(defocus_dists) < 20
    #linescans = linescans[use_examples]
    #defocus_dists = defocus_dists[use_examples]

    n = linescans.shape[0]
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
    linescans = np.copy(linescans)
    defocus_dists = np.copy(defocus_dists)
    def inner_generator(linescans, defocus_dists, indices):
        #yield data in a shuffled order
        for index in indices:
            yield linescans[index, :], defocus_dists[index]

    return lambda: inner_generator(linescans, defocus_dists, data_indices)

def read_patch(hdf_data, led_index, pos_index, slice_index, split_k, patch_index):
    """
    read a patch from a 2048x2048 image
    dotn waste time reading other stuff and cropping
    """
    channel_name = 'led_{}'.format(led_index)
    if channel_name not in hdf_data.channelnames:
        channel_name = 'LED{}'.format(led_index)
    #override LED index if just using autofocus
    if 'autofocus' in hdf_data.channelnames:
        channel_name = 'autofocus'
    patch_size, patches_per_image = get_patch_metadata(hdf_data, split_k)
    y_tile_index = patch_index // split_k
    x_tile_index = patch_index % split_k
    xy_slice = [[y_tile_index*patch_size, (y_tile_index+1)*patch_size],[x_tile_index*patch_size, (x_tile_index+1)*patch_size]]
    return hdf_data.read_image(channel_name=channel_name, position_index=pos_index, relative_z_index=slice_index, xy_slice=xy_slice)

def read_or_calc_focal_planes(hdf, split_k, n_cores=1, recalculate=False):
    """
    compute or load the pre-computed focal planes for each crop in each position
    :return:
    """
    print('Getting focal plane for {}'.format(hdf.path))
    def get_name(pos_index):
        return 'pos{}_focal_plane'.format(pos_index)

    def read_or_compute(pos_index, parallel):
        if hdf.read_annotation(get_name(pos_index)) is None or recalculate:
            #calculate and save it
            focal_plane = calc_focal_plane(hdf, pos_index, split_k=split_k, parallel=parallel)
            for crop_index in focal_plane.keys():
                hdf.write_annotation(get_name(pos_index), focal_plane[crop_index])
        else:
            #read saved value from previous computation
            focal_plane = {}
            for crop_index in range(split_k**2):
                focal_plane[crop_index] = hdf.read_annotation(get_name(pos_index))
        return focal_plane

    if n_cores == 1:
        #single threaded execution
        focal_planes = {pos_index: read_or_compute(pos_index=pos_index) for pos_index in range(hdf.num_positions)}
    else:
        #parallelized
        with Parallel(n_jobs=n_cores) as parallel:
            focal_planes = {pos_index: read_or_compute(pos_index=pos_index, parallel=parallel) for pos_index in range(hdf.num_positions)}

    return focal_planes

def read_or_calc_design_mat(hdf, focal_planes, deterministic_params, recalculate=False):
    """
    Use the regressor class to precompute the early part of the network
    :return:
    """

    param_id_string = str(deterministic_params)

    generator = generator_fn([hdf], focal_planes, mode='all', deterministic_params=deterministic_params, shuffle=False)
    # compute or read from storage deterministic outputs
    feature_name = 'features_' + param_id_string
    defocus_name = 'defocus_dists_' + param_id_string
    features = hdf.read_array(feature_name)
    defocus_dists = hdf.read_array(defocus_name)
    if features is None or recalculate:
        patch_size, patches_per_image = get_patch_metadata(hdf, deterministic_params['tile_split_k'])
        regressor = RegressorNetwork(input_shape=(patch_size, patch_size),
                                     train_generator=generator, deterministic_params=deterministic_params, regressor_only=False)
        features, defocus_dists = regressor.evaluate_deterministic_graph()
        hdf.store_array(feature_name, features)
        hdf.store_array(defocus_name, defocus_dists)
    return features, defocus_dists

def compile_deterministic_data(data, focal_planes, deterministic_params, recalculate=False):
    """
    For all hdf wrappers in data, load design matrix and targets and concatenate them
    Puts the data that has already been fourier transformed and flattened into design matrix
    Computes this using a deterministic neural network if needed, otherwise loads it from the file
    to save time
    """
    deterministic_train_data = [read_or_calc_design_mat(dataset, focal_planes, deterministic_params,
                                                        recalculate=recalculate) for dataset in data]
    # collect training data from all experiments
    features = []
    targets = []
    for f, t in deterministic_train_data:
        features.append(f)
        targets.append(t)

    #pool all data together
    targets = np.concatenate(targets)
    features = np.concatenate(features)
    return features, targets

def open_datasets():
    """
    :return: a dict with names as keys and hdf wrapper objects as values, plus names of test and train datasets
    """
    train_data = [MagellanHDFContainer('/Users/henrypinkard/Desktop/Leukosight data/2018-8-27 Histology autofocus calibration test/Histology af 5x5 20um range 0.5 step_1.hdf')]
    train_data[0].num_positions = train_data[0].num_positions - 1
    test_data = [MagellanHDFContainer('/Users/henrypinkard/Desktop/Leukosight data/2018-8-27 Histology autofocus calibration test/Histology af 4x4 20um range 0.5 step_1.hdf')]


    # train_data=[]
    #train_data.append(MagellanHDFContainer('F:/af/2018-7-19 with RBCs.hdf'))
    #train_data.append(MagellanHDFContainer('F:/af/5x5 80um 300k with RBCs_1.hdf'))
    # train_data.append(MagellanHDFContainer('F:/af/75k 7x7 30um range 50ms_1.hdf'))
    # train_data.append(MagellanHDFContainer('F:/af/300k 7x7 30um range 50ms_1.hdf'))
    #
    # for cells in [75, 300, 1200]:
    #     for exposure in [10, 50, 150]:
    #         if cells == 300 and exposure == 50:
    #             continue #test data
    #         train_data.append(MagellanHDFContainer(
    #             'F:/af/{}k cells {} ms exposure_1.hdf'.format(cells, exposure)))
    #
    # test_data = []
    # for cells in [300]:
    #     for exposure in [50]:
    #         test_data.append(MagellanHDFContainer(
    #             'F:/af/{}k cells {} ms exposure_1.hdf'.format(cells, exposure)))


    return train_data, test_data

def average_predictions(pred, target, block_size):
    pred_avg = np.median(np.reshape(pred, [-1, block_size]), axis=1)
    target_avg = np.median(np.reshape(target, [-1, block_size]), axis=1)
    return pred_avg, target_avg

def plot_results(pred, target, name, draw_rect=False):
    plt.plot(target, pred, '.')
    plt.xlabel('Target defocus (um)')
    plt.ylabel('Predicted defocus (um)')
    if draw_rect:
        min_target = np.min(target)
        max_target = np.max(target)
        height = (max_target - min_target)*np.sqrt(2)
        width = 2
        plt.gca().add_patch(mpatches.Rectangle([min_target, min_target+width/np.sqrt(2)], width, height, angle=-45, color=[1, 0, 0, 0.2]))
        plt.plot([min_target, max_target], [min_target, max_target], 'r-')
    print('{} RMSE: {}'.format(name, np.sqrt(np.mean((pred - target) ** 2))))

def main():
    #LEDs in vertical axis of array:
    # 4 12 28 48 83 119 187
    # 3 11 27 47 84 120 188
    #Max LED is 213 (1 indexed)
    # 3 has defect in it, 4 doesnt

    ############################################
    ##### Parmeters
    ############################################
    # architecture = 'fourier_magnitude'
    deterministic_params = {'non_led_width': 0.1, 'led_width': 0.6, 'autofocus_angle': None, 'led_indices': [119],
                                            'tile_split_k': 2, 'architecture': 'fourier_magnitude'}
    #train_mode = 'train'
    train_mode = 'finetune'
    #train_mode = 'load'

    #did 28 83 119 187 fourier 119 seems to be best
    #also did 120, 4 120, 119 120 which are crappy in terms of generalization
    #tried 4, 84, 120 as an example of a 3 LED pattern

    train_data, test_data = open_datasets()

    patch_size, patches_per_image = get_patch_metadata(train_data[0], deterministic_params['tile_split_k'])
    print('{0} architecture with patch size of {1}x{1}'.format(deterministic_params['architecture'], patch_size))

    # ImageUI(hdf_datasets['50k cells lots of training data_1.hdf'])

    #load or compute focal planes
    train_focal_planes = {hdf: read_or_calc_focal_planes(hdf, split_k=deterministic_params['tile_split_k']) for hdf in train_data}
    test_focal_planes = {hdf: read_or_calc_focal_planes(hdf, split_k=deterministic_params['tile_split_k']) for hdf in test_data}

    #load or compute design matrices
    train_features, train_targets = compile_deterministic_data(train_data, train_focal_planes, deterministic_params=deterministic_params)
    test_features, test_targets = compile_deterministic_data(test_data, test_focal_planes, deterministic_params=deterministic_params)

    #load or train model
    if train_mode == 'load':
        regressor = RegressorNetwork(input_shape=train_features.shape[1], train_generator=None,
                                     val_generator=None, predict_input_shape=[patch_size, patch_size],
                                deterministic_params=deterministic_params, regressor_only=True, train_mode=train_mode)
    else: #finetune or train
        #make genenrator function for training the learnable part of the network
        train_generator = linescan_generator_fn(train_features, train_targets, mode='training', split_k=deterministic_params['tile_split_k'], training_fraction=1.0)
        val_generator = linescan_generator_fn(test_features, test_targets, mode='all', split_k=deterministic_params['tile_split_k'])
        #this call creates network and trains it
        regressor = RegressorNetwork(input_shape=train_features.shape[1], train_generator=train_generator,
                                     val_generator=val_generator, predict_input_shape=[patch_size, patch_size],
                                     deterministic_params=deterministic_params, regressor_only=True, train_mode=train_mode)

    #visualize results
    #validation data drawn from same set used in training, test data is a seperate file
    train_prediction_defocus, train_target_defocus = regressor.analyze_performance(
        linescan_generator_fn(train_features, train_targets, mode='all', split_k=deterministic_params['tile_split_k']))

    test_prediction_defocus, test_target_defocus = regressor.analyze_performance(
            linescan_generator_fn(test_features, test_targets, mode='all', split_k=deterministic_params['tile_split_k']))

    #average predictions
    train_pred_avg, train_target_avg = average_predictions(train_prediction_defocus, train_target_defocus, deterministic_params['tile_split_k'] ** 2)
    test_pred_avg, test_target_avg = average_predictions(test_prediction_defocus, test_target_defocus, deterministic_params['tile_split_k']**2)

    plt.figure(1)
    plot_results(train_prediction_defocus, train_target_defocus, 'Training')
    plot_results(test_prediction_defocus, test_target_defocus, 'Test', draw_rect=True)
    plt.legend(['Training data average', 'Test data averaged', 'Ground truth', 'Objective depth of focus'])

    #show data before tile averaging
    plt.figure(2)
    plot_results(train_pred_avg, train_target_avg, 'Training')
    plot_results(test_pred_avg, test_target_avg, 'Test', draw_rect=True)
    plt.legend(['Training data average', 'Test data averaged', 'Ground truth', 'Objective depth of focus'])

    plt.show(block=True)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()