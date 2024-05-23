
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, saved_weights_path=None):
        # Initialize the Model
        model = Sequential()

        # First CONV => RELU => POOL Layer
        model.add(Conv2D(20, (5, 5), padding="same", activation="relu", input_shape=(height, width, depth)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second CONV => RELU => POOL Layer
        model.add(Conv2D(50, (5, 5), activation="relu", padding="same"))
        # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third CONV => RELU => POOL Layer
        # Convolution -> ReLU Activation Function -> Pooling Layer
        model.add(Conv2D(100, (5, 5), activation="relu", padding="same"))
        # model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # FC => RELU layers
        #  Fully Connected Layer -> ReLU Activation Function
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        # model.add(Dense(500))
        # model.add(Activation("relu"))

        # Using Softmax Classifier for Linear Classification
        model.add(Dense(total_classes, activation="softmax"))
        # model.add(Activation("softmax"))

        # If the saved_weights file is already present i.e model is pre-trained, load that weights
        if saved_weights_path is not None:
            model.load_weights(saved_weights_path)
        return model
# --------------------------------- EOC ------------------------------------
