# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow as tf
import numpy as np
import time
import pandas as pd


class Classifier_FCN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if(verbose==True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory+'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = tf.keras.layers.Input(input_shape)

        conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Activation(activation='relu')(conv1)

        conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation('relu')(conv2)

        conv3 = tf.keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Activation('relu')(conv3)

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                         min_lr=0.0001)

        file_path = self.output_directory+'best_model.hdf5'

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                              save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val,y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 16
        nb_epochs = 400

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = tf.keras.models.load_model(self.output_directory+'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred , axis=1)
        # save_logs(self.output_directory, hist, y_pred, y_true, duration)
        tf.keras.backend.clear_session()

def predict(self, x_test, y_true,return_df_metrics = True):
    model_path = self.output_directory + 'best_model.hdf5'
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(x_test)
    if return_df_metrics:
        y_pred = np.argmax(y_pred, axis=1)
        df_metrics = calculate_metrics(y_true, y_pred, 0.0)
        return df_metrics
    else:
        return y_pred

def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 2), dtype=np.float64), index=[0],
                       columns=['accuracy', 'duration'])
    score = accuracy_score(y_true, y_pred)
    print(score)
    res['accuracy'] = score

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)
    res['duration'] = duration
    return res

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    return

def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float64), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics