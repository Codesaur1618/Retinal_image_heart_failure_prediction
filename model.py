import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from aucmedi import input_interface, DataGenerator, Neural_Network, Image_Augmentation
from aucmedi.utils.class_weights import compute_multilabel_weights
from retinal_crop import Retinal_Crop
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.sampling import sampling_kfold
from aucmedi.neural_network.loss_functions import multilabel_focal_loss
from aucmedi.neural_network.architectures import architecture_dict, supported_standardize_mode
from aucmedi.utils.callbacks import MinEpochEarlyStopping

# Configurations
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
path_riadd = "/storage/riadd2021/Upsampled_Set/"
k_fold = 5
processes = 8
batch_queue_size = 16
threads = 16
arch = "InceptionV3"
input_shape = (224, 224)

# Data Paths
path_images = os.path.join(path_riadd, "images")
path_csv = os.path.join(path_riadd, "data.csv")

# Initialize input data reader
cols = ["DR", "ARMD", ...]  # Define your column names here
ds = input_interface(interface="csv", path_imagedir=path_images, path_data=path_csv, ohe=True, col_sample="ID", ohe_range=cols)
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Other setup tasks...

# Sample dataset via k-fold cross-validation
subsets = sampling_kfold(index_list, class_ohe, n_splits=k_fold, stratified=True, iterative=True, seed=0)

# Iterate over each fold of the CV
for i, fold in enumerate(subsets):
    # Obtain data samplings
    (x_train, y_train, x_val, y_val) = fold

    # Compute class weights
    class_weights = compute_multilabel_weights(ohe_array=y_train)

    # Initialize architecture
    nn_arch = architecture_dict[arch](channels=3, input_shape=input_shape)

    # Initialize model
    model = Neural_Network(nclasses, channels=3, architecture=nn_arch,
                           workers=processes,
                           batch_queue_size=batch_queue_size,
                           activation_output="sigmoid",
                           loss=multilabel_focal_loss(class_weights),
                           metrics=["binary_accuracy", AUC(100)],
                           pretrained_weights=True, multiprocessing=True)
    model.tf_epochs = 10  # Modify number of transfer learning epochs with frozen model layers

    # Initialize training and validation Data Generators
    train_gen = DataGenerator(x_train, path_images, labels=y_train,
                              batch_size=48, img_aug=None, shuffle=True,
                              subfunctions=[], resize=input_shape,
                              standardize_mode=supported_standardize_mode[arch],
                              grayscale=False, prepare_images=False,
                              sample_weights=None, seed=None,
                              image_format=image_format, workers=threads)
    val_gen = DataGenerator(x_val, path_images, labels=y_val, batch_size=48,
                            img_aug=None, subfunctions=[], shuffle=False,
                            standardize_mode=supported_standardize_mode[arch], resize=input_shape,
                            grayscale=False, prepare_images=False, seed=None,
                            sample_weights=None,
                            image_format=image_format, workers=threads)

    # Define callbacks
    cb_mc = ModelCheckpoint(os.path.join("models", "classifier_" + arch, "cv_" + str(i) + ".model.best.hdf5"),
                            monitor="val_loss", verbose=1, save_best_only=True, mode="min")
    cb_cl = CSVLogger(os.path.join("models", "classifier_" + arch, "cv_" + str(i) + ".logs.csv"),
                      separator=',', append=True)
    cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, mode='min', min_lr=1e-7)
    cb_es = MinEpochEarlyStopping(monitor='val_loss', patience=20, verbose=1, start_epoch=60)
    callbacks = [cb_mc, cb_cl, cb_lr, cb_es]

    # Train model
    model.train(train_gen, val_gen, epochs=300, iterations=250,
                callbacks=callbacks, transfer_learning=True)

    # Dump latest model
    model.dump(os.path.join("models", "classifier_" + arch, "cv_" + str(i) + ".model.last.hdf5"))

    # Garbage collection
    del train_gen
    del val_gen
    del model
