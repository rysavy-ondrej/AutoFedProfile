from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Lambda, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from sklearn.model_selection import train_test_split
import seaborn as sns
from viz_helper import make_image_from_sample
from dataclasses import dataclass

@dataclass
class AETrainResult:
    model: object
    avg_error: float
    std_error: float
 
def train_autoencoder(x_train, 
                      x_test, 
                      encoding_dim, 
                      max_epochs: Optional[int] = 50,
                      debug: Optional[bool] = False
                      ) -> Model:
    # 1. Dataset split done in advance to ensure k-fold cross validation
    print(f'train shape={x_train.shape}')
    print(f'test shape={x_test.shape}')
    
    # 2. Define dimensions and architecture
    input_dim = x_train.shape[1]
        
    conv_input_size = 20
    intermediate_dim = 64
    
    # Input placeholder
    input_img = Input(shape=(input_dim,))
    # Slice the first 20 values for convolution
    def slice_first_20(x):
        return tf.reshape(x[:, :conv_input_size], (-1, conv_input_size, 1))  # shape: (batch, steps, channels)
    sliced_input = Lambda(slice_first_20, output_shape=(conv_input_size, 1))(input_img)
    
    # Convolutional processing
    conv_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(sliced_input)
    pool_layer = MaxPooling1D(pool_size=2)(conv_layer)
    flattened_conv = Flatten()(pool_layer)
    
    # Remaining input: slice from index 20 onward
    def slice_remaining(x):
        return x[:, conv_input_size:]
    
    remaining_input = Lambda(slice_remaining, output_shape=(input_dim - conv_input_size,))(input_img)
    dense_input = Dense(intermediate_dim, activation='relu')(remaining_input)
    
    # Combine processed convolutional and dense features
    combined = Concatenate()([flattened_conv, dense_input])
    # Encoder layers
    hidden = Dense(intermediate_dim, activation='relu')(combined)
    encoded = Dense(encoding_dim, activation='relu')(hidden)
    
    # Decoder layers
    hidden_decoded = Dense(intermediate_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(hidden_decoded)
    
    # Autoencoder model
    autoencoder = Model(input_img, decoded)
    
    # Encoder model for later use
    encoder = Model(input_img, encoded)
    
    # Decoder model setup
    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer1 = autoencoder.layers[-2](encoded_input)
    decoder_layer2 = autoencoder.layers[-1](decoder_layer1)  #zde bylo -1
    decoder = Model(encoded_input, decoder_layer2)
    
    # 3. Compile and train the autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=max_epochs)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(x_train, x_train,
                    epochs=max_epochs*10,                      # use some more reasonable number here (>50)
                    callbacks = [early_stop],
                    batch_size=16,
                    shuffle=True,                
                    validation_data=(x_test, x_test))
    
    # 4. Visualize the reconstructed images
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    
    # Assuming x_test contains the original test data
    # and decoded_imgs are the autoencoder's reconstructed outputs
    reconstruction_errors = np.mean(np.square(x_test - decoded_imgs), axis=1)
    
    # Average reconstruction error across all samples
    avg_error = np.mean(reconstruction_errors)
    max_error = np.max(reconstruction_errors)
    min_error = np.min(reconstruction_errors)
    std_error = np.std(reconstruction_errors)
    print(f"Reconstruction error for each sample {reconstruction_errors}")
    print(f"Average={avg_error}, Max={max_error}, Min={min_error} reconstruction errors.")
    print(f"Standard deviation of reconstruction error = {std_error}.")
    if debug:
        #false positives evaluation
        FP_1_sigma = np.sum(reconstruction_errors > (avg_error + 1*std_error))
        print(f"Number of false positives for 1 sigma is {FP_1_sigma} with threshold {avg_error + 1*std_error}.")
        FP_2_sigma = np.sum(reconstruction_errors > (avg_error + 2*std_error))
        print(f"Number of false positives for 2 sigma is {FP_2_sigma}.")
        FP_3_sigma = np.sum(reconstruction_errors > (avg_error + 3*std_error))
        print(f"Number of false positives for 3 sigma is {FP_3_sigma}.")
        
        
        sns.violinplot(data= reconstruction_errors)
        
        # Get indices that would sort the array in ascending order
        worst20 = np.argsort(reconstruction_errors)[-20:][::-1]
        worst50 = np.argsort(reconstruction_errors)[-50:][::-1]
        
        n = 20  # Number of digits to display
        i = 0
        
        print("Worst reconstructed:")
        plt.figure(figsize=(20, 4))
        for j in worst20:
            reconstruction_error = reconstruction_errors[j]
            original = x_test[j]
            original = make_image_from_sample(original) 
            reconstructed = decoded_imgs[j]
            reconstructed = make_image_from_sample(reconstructed) 
            # Original images
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(original, cmap='gray')
            plt.title("Original")
            plt.axis('off')
            
            # Reconstructed images
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed, cmap='gray')
            plt.title(f"RE {reconstruction_error:.3f}")
            plt.axis('off')
            i+=1
        plt.show()

    return AETrainResult(model=autoencoder, 
                         avg_error=avg_error,
                         std_error=std_error)

def compute_ae_reconstruction_error(model, X):
    """
    Compute reconstruction error for an autoencoder model.

    Assumptions:
        - model.predict(X) returns reconstructed data
        - error = mean squared error per sample
    """

    # 1) Keras / TensorFlow: model.predict(X)
    try:
        recon = model.predict(X)
        return np.mean((X - recon) ** 2, axis=1)
    except Exception:
        pass

    # 2) PyTorch-style: model.reconstruct(X)
    if hasattr(model, "reconstruct"):
        recon = model.reconstruct(X)
        return np.mean((X - recon) ** 2, axis=1)

    # 3) Custom model: model(X) returns recon
    if callable(model):
        recon = model(X)
        return np.mean((X - recon) ** 2, axis=1)

    raise RuntimeError("Unsupported autoencoder API for computing reconstruction error.")