import os
import gc
import h5py
import warnings
import configparser
import numpy as np
import tensorflow as tf
# Local Libraries
from utils.helper_functions import *
from utils.tf_losses import TFLosses
from utils.tf_metrics import TFMetrics
from utils.tf_optimizers import TFOptimizers
from utils.DataGenerator import CustomDataGenerator
from models.model_selector import model_selector
warnings.filterwarnings("ignore")

'''CONFIGURATIONS'''
## Data Configurations
config_file = configparser.ConfigParser()
config_file.read('Train_Configs.ini')
train_dir = config_file["TRAIN"]["train_dir"]  # Train Directory
val_dir = config_file["TRAIN"]["val_dir"]  # Validation Directory
independent_val_set = config_file["TRAIN"].getboolean("independent_val_set")  # True: Independent Validation Set | False: Validation Set randomly splitted from the Training Set
validation_portion = np.float_(config_file["TRAIN"]["validation_portion"])  # 0 to 1 [Default: 0; when validation set is independent, otherwise created randomly while training based on "validation_portion"]
imlength = np.int_(config_file["TRAIN"]["imlength"])  # Length or Height of the Image | Image Size: [imwidth, imlength]
imwidth = np.int_(config_file["TRAIN"]["imwidth"])  # Width of the Image
image_color_mode = config_file["TRAIN"]["image_color_mode"]  # Color Mode of the images [rgb, rgba (rgb with transparent alpha channel), grayscale (black and white single channel image)]
mask_color_mode = config_file["TRAIN"]["mask_color_mode"]  # Color Mode of the masks [rgb or grayscale (black and white single channel image)]
num_channels = np.int_(config_file["TRAIN"]["num_channels"])  # Number of Input Channels in the Model [rgb:3, rgba:4, grayscale:1]
normalizing_factor_img = np.int_(config_file["TRAIN"]["normalizing_factor_img"])  # 255.0 for images with pixel values varying between 0 to 255. If it is between 0 to 1, change it to 1
normalizing_factor_msk = np.int_(config_file["TRAIN"]["normalizing_factor_msk"])  # 255.0 for masks with pixel values varying between 0 to 255. If it is between 0 to 1, change it to 1
## Model Configurations
model_genre = config_file["TRAIN"]["model_genre"]  # model_genre: Generation or Genre of the Model: UNet, FPN, LinkNet, etc.
# Encoder
encoder_mode = config_file["TRAIN"]["encoder_mode"]  # Transfer Learning: "pretrained_encoder" | Train from scratch: "from_scratch"
encoder_name = config_file["TRAIN"]["encoder_name"]  # Select an Encoder from a pool of ImageNet trained Models available from TensorFlow, default: ResNet50
encoder_trainable = config_file["TRAIN"].getboolean("encoder_trainable")  # Fine Tuning ON/OFF [True/False] | Start with OFF, Fine Tune later in the 2nd stage, which is optional
# Decoder
decoder_name = config_file["TRAIN"]["decoder_name"]  # Select a Model from the list to train from scratch, UNet is kept as default
model_width = np.int_(config_file["TRAIN"]["model_width"])  # Number of Filters or Kernels of the Input Layer, subsequent layers start from here
model_depth = np.int_(config_file["TRAIN"]["model_depth"])  # Number of Layers in the Model [For the "pretrained_encoder" mode: Maximum 5, Minimum 1]
output_nums = np.int_(config_file["TRAIN"]["output_nums"])  # Number of Outputs for the model
A_E = np.int_(config_file["TRAIN"]["A_E"])  # Turn on AutoEncoder Mode for Feature Extraction [Default: 0]
A_G = np.int_(config_file["TRAIN"]["A_G"])  # Turn on for Guided Attention [Default: 0]
LSTM = np.int_(config_file["TRAIN"]["LSTM"])  # Turn on for LSTM [Default: 0]
dense_loop = np.int_(config_file["TRAIN"]["dense_loop"])  # Number of Densely Connected Residual Blocks in the BottleNeck Layer [Default: 2]
feature_number = np.int_(config_file["TRAIN"]["feature_number"])  # Number of Features to be Extracted [Only required for the AutoEncoder (A_E) Mode]
is_transconv = config_file["TRAIN"].getboolean("is_transconv")  # True: Transposed Convolution | False: UpSampling in the Decoder layer
alpha = np.float_(config_file["TRAIN"]["alpha"])  # Alpha parameter, required for MultiResUNet models [Default: 1]
q_onn = np.int_(config_file["TRAIN"]["q_onn"])  # 'q' for Self-ONN' [Default: 3, set 1 to get CNN]
final_activation = config_file["TRAIN"]["final_activation"]  # Activation Function for the Final Layer: "Linear", "Sigmoid", "Softmax", etc. depending on the problem type
class_number = np.int_(config_file["TRAIN"]["class_number"])  # Number of Output Classes [e.g., here for Kidney Tumor segmentation, Class 1: Kidney | Class 2: Tumor]
target_class_ids = [np.int_(config_file["TRAIN"]["target_class"])]
## Training Configurations
batch_size = np.int_(config_file["TRAIN"]["batch_size"])  # Batch Size of the Images being loaded for training
learning_rate = np.float_(config_file["TRAIN"]["learning_rate"])  # During Fine-Tuning, the Learning Rate should be very low (e.g., 1e-5), otherwise more (e.g., 1e-4, 1e-3)
start_fold = np.int_(config_file["TRAIN"]["start_fold"])  # Fold to Start Training, can be varied from 1 to the last fold
end_fold = np.int_(config_file["TRAIN"]["end_fold"])  # Fold to End Training, can be any value from the start_fold [Number of Folds + 1]
monitor_param = config_file["TRAIN"]["monitor_param"]
patience_amount = np.int_(config_file["TRAIN"]["patience_amount"])
patience_amount_RLROnP = np.int_(config_file["TRAIN"]["patience_amount_RLROnP"])
patience_mode = config_file["TRAIN"]["patience_mode"]
RLROnP_factor = np.float_(config_file["TRAIN"]["RLROnP_factor"])
num_epochs = np.int_(config_file["TRAIN"]["num_epochs"])
loss_function_name = config_file["TRAIN"]["loss_function"]
optimizer_function_name = config_file["TRAIN"]["optimizer_function"]
metric_list = config_file["TRAIN"]["metric_list"]
save_history = config_file["TRAIN"].getboolean("save_history")
load_weights = config_file["TRAIN"].getboolean("load_weights")
save_dir = config_file["TRAIN"]["save_dir"]
task_name = config_file["TRAIN"]["task_name"]
seed = np.int_(config_file["TRAIN"]["seed"])
# Patchify
ispatchify = config_file["TRAIN"].getboolean("patchify")
patch_width = np.int_(config_file["TRAIN"]["patch_width"])  # Length or Height of the Image | Image Size: [imwidth, imlength]
patch_height = np.int_(config_file["TRAIN"]["patch_height"])  # Width of the Image
overlap_ratio = np.float_(config_file["TRAIN"]["overlap_ratio"])
# Deep Supervision
D_S = np.int_(config_file["TRAIN"]["D_S"])  # Turn on Deep Supervision [Default: 0]
ds_type = config_file["TRAIN"]["ds_type"]  # "UNet" or "UNetPP"; only required when Deep Supervision (D_S) is on

config_file = open("Train_Configs.ini", "r")
content = config_file.read()
print("Current Configurations:\n")
print(content)
config_file.flush()
config_file.close()


## Set or Assert Default Conditions
# Validation Set
if independent_val_set == True:
    assert validation_portion == 0.0
# Image Color Mode and Input Channels
if image_color_mode == 'rgb':
    assert num_channels == 3
elif image_color_mode == 'rgba':
    assert num_channels == 4
elif image_color_mode == 'grayscale':
    assert num_channels == 1
# Mask Color Mode and Input Channels
num_channels_msk = 1
if mask_color_mode == 'rgb':
    assert num_channels_msk == 3
elif mask_color_mode == 'rgba':
    assert num_channels_msk == 4
elif mask_color_mode == 'grayscale':
    assert num_channels_msk == 1
# TF Metrics - name selection
if D_S == 0:
    if metric_list == "MeanSquaredError":
        monitor_param = "val_mean_squared_error"
    elif metric_list == "MeanSquaredLogarithmicError":
        monitor_param = "val_mean_squared_logarithmic_error"
    elif metric_list == "MeanAbsoluteError":
        monitor_param = "val_mean_absolute_error"
    elif metric_list == "Accuracy":
        monitor_param = "val_accuracy"
    elif metric_list == "IoU":
        monitor_param = "val_io_u"
elif D_S == 1:
    if metric_list == "MeanSquaredError":
        monitor_param = "val_out_mean_squared_error"
    elif metric_list == "MeanSquaredLogarithmicError":
        monitor_param = "val_out_mean_squared_logarithmic_error"
    elif metric_list == "MeanAbsoluteError":
        monitor_param = "val_out_mean_absolute_error"
    elif metric_list == "Accuracy":
        monitor_param = "val_out_accuracy"
    elif metric_list == "IoU":
        monitor_param = "val_out_io_u"
if ispatchify == True:
    imwidth = patch_width
    imlength = patch_height
    
# Load 2D Segmentation Model
Segmentation_Model = model_selector(model_genre,          
                                encoder_name,
                                decoder_name, 
                                imwidth, 
                                imlength, 
                                model_width, 
                                model_depth, 
                                num_channels=num_channels,
                                output_nums=output_nums,
                                ds=D_S, 
                                ae=A_E, 
                                ag=A_G, 
                                lstm=LSTM, 
                                dense_loop=dense_loop,
                                feature_number=feature_number,  
                                is_transconv=is_transconv,
                                final_activation=final_activation, 
                                train_mode=encoder_mode,
                                is_base_model_trainable=encoder_trainable,
                                alpha=alpha,
                                q=q_onn).segmentation_model()

# Main Training Loop
num_iter = 0
for i in range(start_fold, (end_fold + 1)):
    # Import Train Dataset using the Custom Image Data Generator
    print(f'Fold {i}\n')
    train_image_dir = train_dir + f'/Images/Fold_{i}'
    train_mask_dir = train_dir + f'/Masks/Fold_{i}'
    val_image_dir = val_dir + f'/Images/Fold_{i}'
    val_mask_dir = val_dir + f'/Masks/Fold_{i}'
    train_ds = []
    val_ds = []
    history = []
    train_ds = CustomDataGenerator(img_dir=f'{train_image_dir}/Images',
                                            msk_dir=f'{train_mask_dir}/Masks',
                                            img_size=(imlength,imwidth),
                                            batch_size=batch_size,
                                            num_img_channel=num_channels,
                                            num_msk_channel=num_channels_msk,
                                            norm_factor_img=normalizing_factor_img,
                                            norm_factor_msk=normalizing_factor_msk,
                                            num_class=class_number,
                                            is_train=True,
                                            patchify=ispatchify,
                                            patch_shape=(patch_height,patch_width),
                                            overlap_ratio=overlap_ratio,
                                            deep_supervision=D_S,
                                            model_depth=model_depth,
                                            ds_type=ds_type
                                            )
    if independent_val_set == True:
        val_ds = CustomDataGenerator(img_dir=f'{val_image_dir}/Images',
                                        msk_dir=f'{val_mask_dir}/Masks',
                                        img_size=(imlength,imwidth),
                                        batch_size=batch_size,
                                        num_img_channel=num_channels,
                                        num_msk_channel=num_channels_msk,
                                        norm_factor_img=normalizing_factor_img,
                                        norm_factor_msk=normalizing_factor_msk,
                                        num_class=class_number,
                                        is_train=False,
                                        patchify=ispatchify,
                                        patch_shape=(patch_height,patch_width),
                                        overlap_ratio=overlap_ratio,
                                        deep_supervision=D_S,
                                        model_depth=model_depth,
                                        ds_type=ds_type
                                        )
    print(' ')
    # Reinitialize Model for the New Fold, Change Function depending on Cases
    model = Segmentation_Model
    metrics_dic = [TFMetrics(metric_list, 
                             class_number, 
                             target_class_ids=target_class_ids).metric()]
    if D_S == 1:
        for i in range(0, model_depth):
            metrics_dic.append([TFMetrics(metric_list, 
                                         class_number, 
                                         target_class_ids=target_class_ids).metric()])
    # Compile Model
    model.compile(loss=TFLosses(loss_function_name).loss(),
                  optimizer=TFOptimizers(optimizer_function_name, 
                                         learning_rate).optimizer(),
                  metrics=metrics_dic
                  )
    # Model name based on final configs
    if encoder_mode == "pretrained_encoder":
        if D_S == 0:
            if ispatchify == False:
                model_name = encoder_name + '_' + decoder_name
            elif ispatchify == True:
                model_name = encoder_name + '_' + decoder_name + '_patched'
        elif D_S == 1:
            if ispatchify == False:
                model_name = encoder_name + '_' + decoder_name + '_DS'
            elif ispatchify == True:
                model_name = encoder_name + '_' + decoder_name + '_DS_patched'
    elif encoder_mode == "from_scratch":
        if D_S == 0:
            if ispatchify == False:
                model_name = decoder_name + '_from_scratch'
            elif ispatchify == True:
                model_name = decoder_name + '_from_scratch_patched'
        elif D_S == 1:
            if ispatchify == False:
                model_name = decoder_name + '_from_scratch_DS'
            elif ispatchify == True:
                model_name = decoder_name + '_from_scratch_DS_patched'
    if task_name == "None":
        task_name = model_name
    # Print Model Info
    if num_iter == 0:
        # Model Summary
        # print(model.summary())
        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
        non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        print(f'Model name: {model_name}')
        print(f'Trainable Params: {trainable_params}')
        print(f'Non-trainable Params: {non_trainable_params}')
        print(f'Total Params: {total_params}\n')
    # Load Pre-trained Weights to continue training; delete the trained model from the directory in case of any change causing mismatch
    if os.path.exists(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras') and (load_weights == True) and (encoder_trainable == False):
        print('Loading PreTrained Weights...')
        model.load_weights(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras')
    if encoder_trainable == True:
        if os.path.exists(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras'):
            print('Loading PreTrained Weights for Finetuning...')
            model.load_weights(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras')
        else:
            print('Previously trained model is not available for finetuning. ImageNet weights need to be used.')
    print(' ')
    # Callbacks
    callbacks_ = [tf.keras.callbacks.EarlyStopping(monitor=monitor_param,
                               patience=patience_amount,
                               mode=patience_mode),
                 tf.keras.callbacks.ModelCheckpoint(f'{save_dir}/{task_name}/Fold_{i}/'+model_name+'_'+str(imwidth)+'_Fold_'+str(i)+'.keras',
                                 verbose=1,
                                 monitor=monitor_param,
                                 save_best_only=True,
                                 mode=patience_mode),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_param,
                                   factor=RLROnP_factor,
                                   patience=patience_amount_RLROnP,
                                   verbose=1,
                                   mode=patience_mode,
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0),
                 tf.keras.callbacks.TerminateOnNaN()]
    # Train Model
    if independent_val_set == True:
        model_history = model.fit(train_ds,
                            epochs=num_epochs, 
                            verbose=1, 
                            validation_data=val_ds, 
                            callbacks=callbacks_)
    elif independent_val_set == False:
        assert validation_portion > 0.0
        assert validation_portion < 1.0
        model_history = model.fit(train_ds, 
                            epochs=num_epochs, 
                            verbose=1, 
                            validation_split=validation_portion, 
                            callbacks=callbacks_)
    # Save Training History
    if save_history:
        print('Saving History...')
        history_dict = model_history.history
        history_list = list(history_dict)
        history_len = len(history_list)
        history_path = f'{save_dir}/{task_name}/Fold_{i}'
        if not os.path.exists(history_path):
            os.makedirs(history_path)
        hf = h5py.File(f'{history_path}/{model_name}_Fold_{i}_History.h5', 'w')
        for j in range(0,history_len):
            history_item_name = history_list[j]
            history_item_val = history_dict[history_item_name]
            hf.create_dataset(f'{history_item_name}', data=history_item_val)
        hf.close()
        # Get the dictionary containing each metric and the loss for each epoch
        plot_history(history_dict, history_path, i)
    print('\n')
    num_iter = num_iter + 1
    print('=======================================================================================')
    # Delect any existing Model from the Memory to avoid Reuse in the next iteration
    del model, model_history, train_ds, val_ds
    gc.collect()
