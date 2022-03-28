from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from keras import layers


def make_Xception_model(input_shape, num_classes):
    """
    Creates an Xception style model for the input shape and class number of the current dataset.
    Based largely on a tutorial I found in the Keras documentation for image processing, and then modified to work
    on the current dataset.
    :param input_shape: the input shape for this dataset
    :param num_classes: the number of classes being used to train the model
    :return: a Keras model object based on the inputs and outputs of this structure
    """
    inputs = keras.Input(shape=input_shape)
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def train_Xception_model(input_shape, num_classes, train_dataset, valid_dataset):
    """
    Creates the Xception model using an earlier function, then trains it for 125 epochs.
    At the end of the training, it also displays some graphs about the training and validation accuracy data,
    to help the programmer understand how successful the training was.  it also saves each iteration of the model, so
    the programmer needs to make sure they have at least 38GB of memory open or reduce the number of iterations to match
    :param input_shape: input shape of data in the dataset
    :param num_classes: number of classes in the dataset
    :param train_dataset: the dataset used for training
    :param valid_dataset: the dataset used for validation
    :return:
    """
    model = make_Xception_model(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    epochs = 125
    callbacks = [keras.callbacks.ModelCheckpoint("Xception_model_iterations/save_at_{epoch}.h5")]
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks
    ) # 4 min per epoch of training on my machine
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    pyplot.figure(figsize=(8, 8))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(epochs_range, acc, label='Training Accuracy')
    pyplot.plot(epochs_range, val_acc, label='Validation Accuracy')
    pyplot.legend(loc='lower right')
    pyplot.title('Training and Validation Accuracy')

    pyplot.subplot(1, 2, 2)
    pyplot.plot(epochs_range, loss, label='Training Loss')
    pyplot.plot(epochs_range, val_loss, label='Validation Loss')
    pyplot.legend(loc='upper right')
    pyplot.title('Training and Validation Loss')
    pyplot.show()
    print("Didn't Crash")


def make_AlexNet_model(input_shape, num_classes):
    """
    Creates an AlexNet style model for the input shape and class number of the current dataset.
    I used a diagram of the AlexNet model structure and tweaked my version until it worked for the current dataset.
    :param input_shape: the input shape for this dataset
    :param num_classes: the number of classes being used to train the model
    :return: a Keras model object based on the inputs and outputs of this structure
    """
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_AlexNet_model(input_shape, num_classes, train_dataset, valid_dataset):
    """
    Creates the AlexNet model using an earlier function, then trains it for 125 epochs.
    At the end of the training, it also displays some graphs about the training and validation accuracy data,
    to help the programmer understand how successful the training was.  it also saves each iteration of the model, so
    the programmer needs to make sure they have at least 38GB of memory open or reduce the number of iterations to match
    :param input_shape: input shape of data in the dataset
    :param num_classes: number of classes in the dataset
    :param train_dataset: the dataset used for training
    :param valid_dataset: the dataset used for validation
    """
    model = make_AlexNet_model(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    epochs = 125
    callbacks = [keras.callbacks.ModelCheckpoint("AlexNet_model_iterations/save_at_{epoch}.h5")]
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    pyplot.figure(figsize=(8, 8))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(epochs_range, acc, label='Training Accuracy')
    pyplot.plot(epochs_range, val_acc, label='Validation Accuracy')
    pyplot.legend(loc='lower right')
    pyplot.title('Training and Validation Accuracy')

    pyplot.subplot(1, 2, 2)
    pyplot.plot(epochs_range, loss, label='Training Loss')
    pyplot.plot(epochs_range, val_loss, label='Validation Loss')
    pyplot.legend(loc='upper right')
    pyplot.title('Training and Validation Loss')
    pyplot.show()
    print("Didn't Crash")

def ResNet_identity_block(X, f, filters, stage, block):
    """
    This function defines the identity block of the ResNet structure, which is called throughout the ResNet structure
    :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network
    :return: output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    return X

def ResNet_convolutional_block(X, f, filters, stage, block, s = 2):
    """
    This function defines the convolutional block of the ResNet structure,
    which is called throughout the ResNet structure
    :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network
    :param s: Integer, specifying the stride to be used
    :return: output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path
    X = layers.Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',
                      kernel_initializer = keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = layers.Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = layers.Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b',
                      kernel_initializer = keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid',
                      name = conv_name_base + '2c', kernel_initializer = keras.initializers.glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = layers.Activation('relu')(X)
    return X

def make_ResNet_model(input_shape, num_classes):
    """
    This function returns a ResNet style model by using a combination of predefined identity and convolutional blocks
    and the input shape/number of classes from the target dataset.

    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    :param input_shape: shape of the images of the dataset
    :param num_classes: integer, number of classes
    :return: a Keras Model object with the defined ResNet structure
    """
    """
    Developer Notes: I found this model type the hardest to understand, and I am still the least confident in my ability
                    to improve this type of model when compared to the other two model types.  I ended up following an
                    online tutorial pretty closely with this one, and reading the operation of it step-by-step did help
                    me understand the ResNet model a bit better, but I still avoided tweaking this one the most just in 
                    case.  It also ended up being my least accurate model, which might be partly due to my avoidance on 
                    editing the tutorial code in places.  If I end up revisiting this code, I leave this note as a 
                    reminder to read some online resources before changing too many things.  
    """
    # Define the input as a tensor with shape input_shape
    x_input = layers.Input(input_shape)

    # Zero-Padding
    x = layers.ZeroPadding2D((3, 3))(x_input)

    # Stage 1
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
                      kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    x = layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = ResNet_convolutional_block(x, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    x = ResNet_identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = ResNet_identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    x = ResNet_convolutional_block(x, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    x = ResNet_identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = ResNet_identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = ResNet_identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    x = ResNet_convolutional_block(x, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    x = ResNet_identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = ResNet_identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = ResNet_identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = ResNet_identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = ResNet_identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    x = ResNet_convolutional_block(x, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    x = ResNet_identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = ResNet_identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "x = AveragePooling2D(...)(x)"
    x = layers.AveragePooling2D((2, 2), name="avg_pool")(x)

    # output layer
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, activation='softmax', name='fc' + str(num_classes), kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)

    # Create model
    model = keras.Model(inputs=x_input, outputs=x, name='ResNet50')

    return model

def train_ResNet_model(input_shape, num_classes, train_dataset, valid_dataset):
    """
    Creates the ResNet model using an earlier function, then trains it for 125 epochs.
    At the end of the training, it also displays some graphs about the training and validation accuracy data,
    to help the programmer understand how successful the training was.  it also saves each iteration of the model, so
    the programmer needs to make sure they have at least 38GB of memory open or reduce the number of iterations to match
    :param input_shape: input shape of data in the dataset
    :param num_classes: number of classes in the dataset
    :param train_dataset: the dataset used for training
    :param valid_dataset: the dataset used for validation
    """
    model = make_ResNet_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 125
    callbacks = [keras.callbacks.ModelCheckpoint("ResNet_model_iterations/save_at_{epoch}.h5")]
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    pyplot.figure(figsize=(8, 8))
    pyplot.subplot(1, 2, 1)
    pyplot.plot(epochs_range, acc, label='Training Accuracy')
    pyplot.plot(epochs_range, val_acc, label='Validation Accuracy')
    pyplot.legend(loc='lower right')
    pyplot.title('Training and Validation Accuracy')

    pyplot.subplot(1, 2, 2)
    pyplot.plot(epochs_range, loss, label='Training Loss')
    pyplot.plot(epochs_range, val_loss, label='Validation Loss')
    pyplot.legend(loc='upper right')
    pyplot.title('Training and Validation Loss')
    pyplot.show()
    print("Didn't Crash")


if __name__ == '__main__':
    image_size = (150, 150)
    input_shape = (150, 150, 3)
    batch_size = 64
    num_classes = 350

    train_directory = 'C:/Users/Brian/Desktop/Capstone/KaggleData/train'
    valid_directory = 'C:/Users/Brian/Desktop/Capstone/KaggleData/valid'
    test_directory = 'C:/Users/Brian/Desktop/Capstone/KaggleData/test'

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_directory,
        labels="inferred",
        label_mode="categorical",
        seed=1128,
        image_size=image_size,
        batch_size=batch_size,
    )
    valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        valid_directory,
        labels="inferred",
        label_mode="categorical",
        seed=3995,
        image_size=image_size,
        batch_size=batch_size,
    )
    train_dataset = train_dataset.prefetch(batch_size)
    valid_dataset = valid_dataset.prefetch(batch_size)
    # train_Xception_model(input_shape, num_classes, train_dataset, valid_dataset)
    # train_AlexNet_model(input_shape, num_classes, train_dataset, valid_dataset)
    # train_ResNet_model(input_shape, num_classes, train_dataset, valid_dataset)