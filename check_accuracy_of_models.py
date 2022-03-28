import os

import tensorflow as tf

import model_maker_rig

def test_Xception(test_dataset, input_shape, num_classes):
    """
    Tests all the iterations of the Xception model that are created by the model_maker_rig.py functions

    WARNING: this is designed to be used when the target folder has all 125 iterations, which most users will have
    deleted after testing once to make more room on the hard drive.  check for target files before use.
    :param test_dataset: the dataset used for testing the model
    :param input_shape: input shape of the test dataset
    :param num_classes: number of classes in the test dataset
    :return: does not return any values
    """
    for num in range(1, 126):
        name = "Xception_model_iterations/save_at_" + str(num) + ".h5"
        model = model_maker_rig.make_Xception_model(input_shape=input_shape, num_classes=num_classes)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        model.load_weights(name)
        print(name)
        model.evaluate(test_dataset)
        """
        After testing each model from the original set, here are the top models:
        124: 94.3%
        113: 93.2%
        100: 93.2%
        4 min per training round
        """

def test_AlexNet(test_dataset, input_shape, num_classes):
    """
    Tests all the iterations of the AlexNet model that are created by the model_maker_rig.py functions

    WARNING: this is designed to be used when the target folder has all 125 iterations, which most users will have
    deleted after testing once to make more room on the hard drive.  check for target files before use.
    :param test_dataset: the dataset used for testing the model
    :param input_shape: input shape of the test dataset
    :param num_classes: number of classes in the test dataset
    :return: does not return any values
    """
    for num in range(1, 126):
        name = "AlexNet_model_iterations/save_at_" + str(num) + ".h5"
        model = model_maker_rig.make_AlexNet_model(input_shape=input_shape, num_classes=num_classes)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        model.load_weights(name)
        print(name)
        model.evaluate(test_dataset)
        """
        After testing each model from the original set, here are the top models:
        65: 87.1%
        69: 87.1%
        116: 86.8%
        40-60 seconds per training round
        """

def test_ResNet(test_dataset, input_shape, num_classes):
    """
    Tests all the iterations of the ResNet model that are created by the model_maker_rig.py functions

    WARNING: this is designed to be used when the target folder has all 125 iterations, which most users will have
    deleted after testing once to make more room on the hard drive.  check for target files before use.
    :param test_dataset: the dataset used for testing the model
    :param input_shape: input shape of the test dataset
    :param num_classes: number of classes in the test dataset
    :return: does not return any values
    """
    for num in range(1, 126):
        name = "ResNet_model_iterations/save_at_" + str(num) + ".h5"
        model = model_maker_rig.make_ResNet_model(input_shape=input_shape, num_classes=num_classes)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy']
                      )
        model.load_weights(name)
        print(name)
        model.evaluate(test_dataset)
        """
        After testing each model from the original set, here are the top models:
        84: 85%
        83: 84.9%
        71: 84.4%
        4ish min per training round
        """

def test_best(test_dataset, input_shape, num_classes):
    """
    Tests the best models of each type, which I manually copied and renamed after using the above functions to find the
    best models from each model type.  This is used to double check the accuracies of each model or as a demonstration
    of these models.
    :param test_dataset: the dataset used for testing the model
    :param input_shape: input shape of the test dataset
    :param num_classes: number of classes in the test dataset
    :return: does not return any values
    """
    name = "Best_Xception.h5"
    xcm = model_maker_rig.make_Xception_model(input_shape=input_shape, num_classes=num_classes)
    xcm.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    xcm.load_weights(name)
    print(name)
    xcm.evaluate(test_dataset)

    name = "Best_AlexNet.h5"
    aln = model_maker_rig.make_AlexNet_model(input_shape=input_shape, num_classes=num_classes)
    aln.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )
    aln.load_weights(name)
    print(name)
    aln.evaluate(test_dataset)

    name = "Best_ResNet.h5"
    rnn = model_maker_rig.make_ResNet_model(input_shape=input_shape, num_classes=num_classes)
    rnn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )
    rnn.load_weights(name)
    print(name)
    rnn.evaluate(test_dataset)


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
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory,
        labels="inferred",
        label_mode="categorical",
        seed=3995,
        image_size=image_size,
        batch_size=batch_size,
    )
    # test_Xception(test_dataset, input_shape, num_classes)
    # test_AlexNet(test_dataset, input_shape, num_classes)
    # test_ResNet(test_dataset, input_shape, num_classes)
    test_best(test_dataset, input_shape, num_classes)

