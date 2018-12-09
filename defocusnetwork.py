import numpy as np
import tensorflow as tf
import shutil
import sys
import os

class DefocusNetwork:    

    def __init__(self, input_shape, train_generator, deterministic_params=None,
                 val_generator=None, predict_input_shape=None, train_mode=None, **kwargs):
        """
        Constructor for defocus prediction network. If in 'train mode', go ahead and train the network.
        The network will stop training when error on the validation set has not decreased for (val_overshoot_steps)
        :param input_shape: Dimension of input used for training
        :param train_generator: generator function to provide training examples
        :param deterministic_params: arguments for specifiying architecture of deterministic part
        :param val_generator: generator function for validation example
        :param predict_input_shape: the shape of a raw image put into the network for inference mode
        :param train_mode: 'train', 'load', or 'finetune'
        """

        # hyperparameters for the trainable part of the network
        self.hyper_params = {'batch_size': 25, 'learning_rate': 2e-5, 'steps_per_validation': 25,
                        'val_overshoot_steps': 2000,
                        'num_hidden_units': [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                        'regularization_strength': 0.0, 'dropout_rate': 0.0, 'input_dropout_rate': 0.6,
                        'total_steps': 10000000000000}
        for key in kwargs.keys():
            if key in self.hyper_params.keys():
                self.hyper_params[key] = kwargs[key]
        # directory paths for logging training, exporting trained model, checkpoints, and loading model
        self.params = {'log_dir': './log', 'export_path': "./exported_model", 'checkpoint_path': './checkpoints',
                       'load_model_path': './exported_model'}
        for key in kwargs.keys():
            if key in self.params.keys():
                self.params[key] = kwargs[key]
        self.deterministic_params = deterministic_params

        self.input_shape = input_shape
        self.predict_input_shape = predict_input_shape
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.train_mode = train_mode
        
        #clear everyhting so other instances of this class wont interfere with this one
        tf.reset_default_graph()

        self.sess = tf.Session()
        if train_mode is not None:
            if train_mode == 'train':
                #not using deterministic front end, but rather trainable backend
                if 'normalizations' in kwargs.keys():
                    print('loading normalizations')
                    self.mean = kwargs['normalizations']['mean']
                    self.stddev = kwargs['normalizations']['std']
                else:  
                    print('computing normalizations')
                    self._compute_normalizations(self.train_generator)
                predict_input_op, predict_output_op = self._train()
                self.predict_input_op = predict_input_op
                self.predict_output_op = predict_output_op
            elif train_mode == 'load':
                #load full saved model instead of training
                tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.params['load_model_path'])
                self.predict_input_op = tf.get_default_graph().get_tensor_by_name('deterministic/truediv:0')
                self.predict_output_op = tf.get_default_graph().get_tensor_by_name('predict_network/output:0')
            elif train_mode == 'finetune':
                #load the normalization values form the old graph
                #Might be able to replace this and not save and reopen session...
                tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.params['load_model_path'])
                self.mean = self.sess.run(tf.get_default_graph().get_tensor_by_name('predict_network/Const:0'))
                self.stddev = self.sess.run(tf.get_default_graph().get_tensor_by_name('predict_network/Const_1:0'))
                self.sess.close()
                tf.reset_default_graph()
                self.sess = tf.Session()
                predict_input_op, predict_output_op = self._train(load_variables=True)
                self.predict_input_op = predict_input_op
                self.predict_output_op = predict_output_op

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def compute_gradients(self, generator_fn):
        gradient_op = tf.get_default_graph().get_tensor_by_name(
            'predict_network/prediction_feature_gradient/predict_network/sub_grad/Reshape:0')
        gradients = []
        targets = []
        dim = self.predict_input_shape[0]
        #divide this by two because it was used twice
        led_width_pix = int(self.deterministic_params['led_width'] * dim) // 2
        # divide this by two to exclude duplicatte information
        non_led_width_pix = int(self.deterministic_params['non_led_width'] * dim) //2
        for input in generator_fn():
            raw_gradient = np.ravel(self.sess.run(gradient_op, feed_dict={self.predict_input_op: np.reshape(input[0], [1, -1])}))
            first_half = raw_gradient[:raw_gradient.size // 2]
            second_half = raw_gradient[-raw_gradient.size // 2:]
            img1 = np.reshape(first_half, [led_width_pix, non_led_width_pix])
            img2 = np.reshape(second_half, [led_width_pix, non_led_width_pix])
            #stack em together
            gradients.append(np.concatenate((img1, img2), axis=0))
            targets.append(input[1])
        return gradients, targets

    def evaluate_deterministic_graph(self):
        """
        evaluate deterministic graph over training data
        Used either for precomputing deterministic part of graph or for calculating training normalizations
        :return:
        """
        # iterate over entire dataset to comput normalizations
        normalization_dataset = self._make_dataset(repeat=False, generator_fn=self.train_generator)
        input_data_op, target_op, _ = self._build_input_pipeline('training', dataset=normalization_dataset)
        linescan_op = self._build_deterministic_graph(input_data_op)
        linescans = None
        targets = None
        print("Evaluating deterministic graph over training set...")
        i = 0
        while True:
            print('batch {}\r'.format(i),end='')
            i += 1
            try:
                [new_linescans, new_targets] = self.sess.run([linescan_op, target_op])
                if linescans is None:
                    linescans = new_linescans
                    targets = new_targets
                else:
                    linescans = np.vstack((linescans, new_linescans))
                    targets = np.hstack((targets, new_targets))
            except tf.errors.OutOfRangeError:
                break

        return linescans, targets.T  # return n x p and n x 1

    def predict(self, generator_fn, consensus=True):
        """
        Compute predicted and target defocus for all data pairs provided by generator function
        :param generator_fn:
        :param recompute_normalizations:
        :return:
        """
        prediction = np.array([])
        target = np.array([])
        for input in generator_fn():
            pred_new = self.sess.run(self.predict_output_op,
                                     feed_dict={self.predict_input_op: np.reshape(input[0], [1, -1])})
            prediction = np.concatenate((prediction, pred_new))
            target = np.concatenate((target, np.array([input[1]])))
        if consensus:
            pred_consensus, target_consensus = self._combine_predictions(prediction, target)
            return pred_consensus, target_consensus
        else:
            return prediction, target

    def _train(self, load_variables=False):
        """
        Automatically train model
        :param sess:
        :param prediction_generator_fn:
        :return: predict_op
        """
        train_dataset = self._make_dataset(repeat=True, generator_fn=self.train_generator)
        val_dataset = self._make_dataset(repeat=False, generator_fn=self.val_generator)
        # build seperate graphs because they each have different input Datasets
        train_op = self._build_graph(graph_mode='training', dataset=train_dataset)
        validation_error_op, validation_error_init_op = self._build_graph(graph_mode='evaluate', dataset=val_dataset)
        predict_input_tensor, predict_output_tensor = self._build_graph(graph_mode='predict')

        # May want to switch to MonitoredTRainingSession when moving to a distributed environment
        def remove_if_present(path):
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass

        remove_if_present(self.params['checkpoint_path'])
        remove_if_present(self.params['export_path'])
        remove_if_present(self.params['log_dir'])

        # to combine them into a single op that generates all the summary data.
        summary_op = tf.summary.merge_all()
        train_log_writer = tf.summary.FileWriter(self.params['log_dir'], graph=self.sess.graph)

        # initialize all variables in the graph
        if load_variables:
            saver = tf.train.Saver()
            saver.restore(self.sess, self.params['load_model_path'] + '/variables/variables')
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

        saver = tf.train.Saver()

        # train until told to stop
        min_validation_error = sys.float_info.max
        step = 0
        min_error_step = 0
        print("Training model...")
        while True:
            if 'total_steps' in self.hyper_params.keys() and step >= self.hyper_params['total_steps']:
                break
#             print('Step: {}'.format(step))
            # make one training step
            self.sess.run(train_op)
            # train_log_writer.add_summary(summary, global_step=step)
            # occasionally compute loss over whole validation set
            if step % self.hyper_params['steps_per_validation'] == 0:
                # initialize validation iterator to go through entire validation set once
                self.sess.run(validation_error_init_op)
                # initialize or reinitialize RMSE so we can calculate it fresh over validation set
                self.sess.run(tf.local_variables_initializer())
                # go through all validation examples and calculate metric
                while True:
                    try:
                        # run both the update op and the running avg and store the latter
                        error = self.sess.run(validation_error_op)[1]
                    except tf.errors.OutOfRangeError:
                        break
                train_log_writer.add_summary(self.sess.run(summary_op), global_step=step)
                print("Step {}, val loss: {}".format(step, error))
#                 print("Step {}, val loss: {}\r".format(step, error),end='')
                if error < min_validation_error:
                    min_error_path = saver.save(self.sess, self.params['checkpoint_path'] + '/checkpoint',
                                                global_step=step)
                    min_validation_error = error
                    min_error_step = step
                elif step - min_error_step > self.hyper_params['val_overshoot_steps']:
                    break
            step += 1

        # saver.restore(self.sess, min_error_path)
        # export saved graph
        print("Exporting model...")
        #TODO: this is exporting the input as the input to the deterministic graph rather than the trainable input,
        #TODO: might want to fix this in the future. For now it is fixed on tha JAva side
        tf.saved_model.simple_save(self.sess, self.params['export_path'],
                                   inputs={'input': predict_input_tensor},
                                   outputs={'output': predict_output_tensor})
        print("Export complete")
        return predict_input_tensor, predict_output_tensor

    def _combine_predictions(self, pred, target):
        """
        Take the median of all predictions coming from a single raw image--that is, all crops that came from the same
        original image
        """
        block_size = self.deterministic_params['tile_split_k'] ** 2
        pred_consensus = np.median(np.reshape(pred, [-1, block_size]), axis=1)
        target_consensus = np.median(np.reshape(target, [-1, block_size]), axis=1)
        return pred_consensus, target_consensus

    def _compute_normalizations(self, generator_fn):
        # going to train, rather than compute something deterministic, so compute normalization
        #linescan architecture
        linescans = self._read_design_mat(generator_fn=generator_fn)
        # compute mean and SD for training
        self.mean = np.mean(linescans, axis=0)
        self.stddev = np.std(linescans, axis=0)

    def _read_design_mat(self, generator_fn):
        # design_mat precomputed, just need to read in
        iterator = self._make_dataset(repeat=False, generator_fn=generator_fn).make_one_shot_iterator()
        design_mat = []
        input_op = iterator.get_next()
        while True:
            try:
                design_mat.append(self.sess.run(input_op[0]))
            except tf.errors.OutOfRangeError:
                break
        design_mat = np.vstack(design_mat)
        return design_mat

    def _make_dataset(self, repeat, generator_fn):
        """
        Data contains generator functions, image size, and led indeices
        from TF docs: A Dataset can be used to represent an input pipeline as a collection of elements (nested structures of tensors)
        # and a "logical plan" of transformations that act on those elements.
        """
        if  self.train_mode is not None:
            types = (tf.float32, tf.float32)
            shapes = (tf.TensorShape(self.input_shape), tf.TensorShape([]))
        else: #design mat has already been computed
            types = (tf.float32, tf.float32)
            shapes = (tf.TensorShape([*self.input_shape]), tf.TensorShape([]))

        dataset = tf.data.Dataset.from_generator(generator_fn, types, shapes)

        if repeat:
            return dataset.batch(self.hyper_params['batch_size']).repeat()
        else:
            return dataset.batch(self.hyper_params['batch_size'])

    def _build_input_pipeline(self, graph_mode, dataset=None):
        with tf.name_scope('{}_input'.format(graph_mode)):
            if graph_mode == 'evaluate':
                iterator = dataset.make_initializable_iterator()
                input, target = iterator.get_next()
                return input, target, iterator.initializer
            elif graph_mode == 'training' or graph_mode == 'analyze':
                iterator = dataset.make_one_shot_iterator()
                input, target = iterator.get_next()
                return input, target, None
            else: #predict mode
                return tf.placeholder(tf.float32, shape=[None, *self.predict_input_shape], name='input'), None, None

    def _build_deterministic_graph(self, input_tensor):
        print('Building deterministic graph...')

        with tf.name_scope("deterministic"):
            # later might want: tf.image.central_crop
            # for now, add together all images

            if type(input_tensor) is dict:
                incoherent_sum = None
                for key in input_tensor.keys():
                    if incoherent_sum is None:
                        incoherent_sum = input_tensor[key]
                    else:
                        incoherent_sum = incoherent_sum + input_tensor[key]
            else:
                incoherent_sum = tf.cast(input_tensor, tf.float32)
            #subtract mean
            incoherent_sum = incoherent_sum - tf.reduce_mean(incoherent_sum)

            #take magnitude of top quadrant of FT as feature vec
            ft = tf.fft2d(tf.cast(incoherent_sum, tf.complex64))
            # take low frequency part of fourier spectrum that encompasses the direction of th LED axis
            dim = ft.get_shape()[1].value
            ##divide this by two because its used twice
            led_width_pix = int(self.deterministic_params['led_width'] * dim) // 2
            # divide this by two to exclude duplicatte information
            non_led_width_pix = int(self.deterministic_params['non_led_width'] * dim) // 2
            if 'architecture' in self.deterministic_params.keys() and self.deterministic_params['architecture'] == 'keep_it_real':
                ft_real = tf.real(ft)
                features = tf.concat([tf.layers.flatten(ft_real[:, :led_width_pix, :non_led_width_pix]),
                                      tf.layers.flatten(ft_real[:, -led_width_pix:, :non_led_width_pix])], axis=1)
                # l2 normalize
                normalized = features / tf.expand_dims(tf.norm(features, axis=1), axis=1)
                # Do equivalent of log transform
#                 normalized_abs = tf.abs(normalized)
#                 rescale = tf.log(normalized_abs + np.finfo(np.float32).eps) / normalized_abs
#                 normalized * rescale
                return normalized
            else:
                ft_mag = tf.abs(ft)
                features = tf.concat([tf.layers.flatten(ft_mag[:, :led_width_pix, :non_led_width_pix]),
                                      tf.layers.flatten(ft_mag[:, -led_width_pix:, :non_led_width_pix])], axis=1)
                #l2 normalize
                normalized = features / tf.expand_dims(tf.norm(features, axis=1), axis=1)
                #log transform--make sure its always a positive number
#                 normalized = tf.log(normalized + np.finfo(np.float32).eps)
                return normalized

    def _build_graph(self, graph_mode, dataset=None):
        input_data, target, validation_iterator_init_op = self._build_input_pipeline(graph_mode=graph_mode, dataset=dataset)
        if graph_mode == 'predict':
            input_data = self._build_deterministic_graph(input_data)
        print("Building trainable graph...")
        with tf.name_scope("{}_network".format(graph_mode)):
            #Add all LED images together
            if type(input_data) is dict:
                input_tensor = None
                for key in input_data.keys():
                    if input_tensor is None:
                        input_tensor = input_data[key]
                    else:
                        input_tensor = input_tensor + input_data[key]
            else:
                input_tensor = input_data

            normalized_input = (input_tensor - tf.constant(self.mean, dtype=tf.float32)) / tf.constant(self.stddev, dtype=tf.float32)
            #a feature vector as input being fed into one or more hidden layers
            normalized_input = tf.layers.dropout(normalized_input, training=graph_mode == 'training', rate=self.hyper_params['input_dropout_rate'])
            # other layers (containing weights, biases as tf.Variables)
            # regularizer = tf.contrib.layers.l1_regularizer(scale=self.hyper_params['regularization_strength'])
            regularizer = tf.contrib.layers.l2_regularizer(scale=self.hyper_params['regularization_strength'])
            current_layer = normalized_input
            index = 0
            for num_hidden in self.hyper_params['num_hidden_units']:
                current_layer = tf.layers.dense(inputs=current_layer, units=num_hidden, activation=tf.nn.relu,
                                name='hidden{}'.format(index), reuse=tf.AUTO_REUSE, kernel_regularizer=regularizer)
#                 current_layer = tf.layers.batch_normalization(inputs=current_layer, name='hidden{}_batchnorm'.format(index),
#                                               training=graph_mode == 'training', reuse=tf.AUTO_REUSE)
                current_layer = tf.layers.dropout(current_layer, training=graph_mode == 'training', rate=self.hyper_params['dropout_rate'])
                index += 1
            output = tf.layers.dense(inputs=current_layer, units=1, activation=None, name='output_weights', reuse=tf.AUTO_REUSE, kernel_regularizer=regularizer)
            output = tf.layers.dropout(output, training=graph_mode == 'training', rate=self.hyper_params['dropout_rate'])

            predictions = tf.squeeze(output, axis=1, name="output")
            #for calculating saliency map
            gradient = tf.gradients(predictions, input_data, name="prediction_feature_gradient")[0]
            if graph_mode == 'predict':
                return input_tensor, predictions
            if graph_mode == 'evaluate':
                #use metric so it can be computed over bigger set than fits in memory
                with tf.name_scope("Validation_metrics"):
                    rmse, update_op = tf.metrics.root_mean_squared_error(target, predictions)
                    tf.summary.scalar("Validation_RMSE", rmse)
                return (rmse, update_op), validation_iterator_init_op
            elif graph_mode == 'training':
                with tf.name_scope("loss_function"):
                    loss = tf.losses.mean_squared_error(target, predictions)
#                     loss = tf.sqrt(loss)
                    if self.hyper_params['regularization_strength'] == 0:
                        total_loss = loss
                    else:
                        with tf.name_scope("regularization"):
                            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                            total_reg_loss = tf.add_n(reg_losses)
                        total_loss = total_reg_loss + loss
                with tf.name_scope("optimizer"):
#                     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#                     with tf.control_dependencies(update_ops):
#                         # Ensures that we execute the update_ops before performing the train_step
                    train_step = tf.train.AdamOptimizer(self.hyper_params['learning_rate']).minimize(total_loss, global_step=tf.train.get_global_step())
                        # train_step = tf.train.GradientDescentOptimizer(self.hyper_params['learning_rate']).minimize(loss, global_step=tf.train.get_global_step())
                return train_step
            elif graph_mode == 'analyze': #for plotting error vs defocus distance
                with tf.name_scope("analysis_metrics"):
                    return predictions, target
            else:
                raise Exception('Unknown graph mode')
