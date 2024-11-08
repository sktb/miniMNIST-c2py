# %%
import math

# %%
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
EPOCHS = 20
BATCH_SIZE = 64
IMAGE_SIZE = 28
TRAIN_SPLIT = 0.8
PRINT_INTERVAL = 1000
RAND_MAX = 32767

TRAIN_IMG_PATH = "data/train-images.idx3-ubyte"
TRAIN_LBL_PATH = "data/train-labels.idx1-ubyte"

# %%
class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = [0.0] * (input_size * output_size)  # Placeholder initialization
        self.biases = [0.0] * output_size
        self.weight_momentum = [[0.0] * input_size] * output_size
        self.bias_momentum = [0.0] * output_size

# %%
class Network:
    def __init__(self, hidden_input_size, hidden_output_size, output_input_size, output_output_size):
        self.hidden = Layer(hidden_input_size, hidden_output_size)
        self.output = Layer(output_input_size, output_output_size)

# %%
class InputData:
    def __init__(self, images, labels, count):
        self.images = images
        self.labels = labels
        self.count = count

# %%
import random

def init_layer(layer: Layer, in_size: int, out_size: int):
    n = in_size * out_size
    scale = math.sqrt(2.0 / in_size)

    layer = Layer(in_size, out_size)

    for i in range(0, n):
        layer.weights[i] = (random.random() / RAND_MAX - 0.5) * 2 * scale

# %%
import struct
import numpy as np

def load_images(filename, image_size):
    with open(filename, "rb") as file:
        # Read and ignore the first integer (`temp`)
        temp = struct.unpack('>i', file.read(4))[0]
        
        # Read number of images
        n_images = struct.unpack('>i', file.read(4))[0]
        
        # Read number of rows and columns, ensuring byte swap with big-endian format
        rows = struct.unpack('>i', file.read(4))[0]
        cols = struct.unpack('>i', file.read(4))[0]
        
        # Read raw pixel data for all images
        images_data = file.read(n_images * image_size * image_size)
    
    # Convert raw data to numpy array of the appropriate shape
    images_array = np.frombuffer(images_data, dtype=np.uint8)
    # images_array = images_array.reshape((n_images, image_size, image_size))
    
    return n_images, rows, cols, images_array


# %%
import struct
import numpy as np

def load_labels(filename):
    with open(filename, "rb") as file:
        # Read and ignore the first integer (`temp`)
        temp = struct.unpack('>i', file.read(4))[0]
        
        # Read number of images
        n_labels = struct.unpack('>i', file.read(4))[0]
        
        # Read raw pixel data for all images
        labels_data = file.read(n_labels)
    
    # Convert raw data to numpy array of the appropriate shape
    labels_array = np.frombuffer(labels_data, dtype=np.uint8)
    labels_array = labels_array.reshape((n_labels))
    
    return n_labels, labels_array


# %%
import random

def shuffle_data(images, labels, n: int):
    for i in range(n - 1, 0):
        j = random.randint(i, INPUT_SIZE)
        for k in range(0, INPUT_SIZE):
            temp = images[i * INPUT_SIZE + k]
            images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k]
            images[j * INPUT_SIZE + k] = temp

        temp = labels[i]
        labels[i] = labels[j]
        labels[j] = temp

# %%
def softmax(input, size):
    max = input[0]
    sum = 0

    for i in range (1, size):
        if input[i] > max:
            max = input[i]
        
    for i in range(0, size):
        input[i] = expf(input[i] - max)
        sum += input[i]
    
    for i in range(0, size):
        input[i] /= sum

# %%
def forward(layer: Layer, input, output):
    for i in range(0, layer.output_size):
        output[i] = layer.biases[i]

    for j in range(0, layer.input_size):
        in_j = input[j];
        #weight_row = layer.weights[j * layer.output_size];
        for i in range(0, layer.output_size):
            output[i] += in_j * layer.weights[i * j]

    for i in range(0, layer.output_size):
        print(output[i])
        output[i] = output[i] if output[i] > 0 else 0

# %%
def train(net: Network, input, label, lr):
    final_output = [0] * OUTPUT_SIZE
    hidden_output= [0] * HIDDEN_SIZE
    output_grad = [0] * OUTPUT_SIZE
    hidden_grad = [0] * HIDDEN_SIZE

    forward(net.hidden, input, hidden_output);
    forward(net.output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    for i in range(0, OUTPUT_SIZE):
        output_grad[i] = final_output[i] - (i == label)

    backward(net.output, hidden_output, output_grad, hidden_grad, lr)

    for i in range(0, HIDDEN_SIZE):
        hidden_grad[i] *= 1 if hidden_output[i] > 0 else 0 # ReLU derivative

    backward(net.hidden, input, hidden_grad, NULL, lr)

    return final_output

# %%
def __init__():
    net = Network(INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    
    init_layer(net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    (n_images, rows, cols, images_array) = load_images('./data/train-images.idx3-ubyte', IMAGE_SIZE)
    (n_labels, labels_array) = load_labels('./data/train-labels.idx1-ubyte')

    data = InputData(images_array, labels_array, n_images)
    shuffle_data(data.images, data.labels, data.count)

    learning_rate = LEARNING_RATE
    img = [0] * INPUT_SIZE

    train_size =int(data.count * TRAIN_SPLIT)
    test_size = int(data.count - train_size)

    for epoch in range (0, EPOCHS):
        total_loss = 0
        for i in range(0, train_size):
            for k in range(0, INPUT_SIZE):
                print(i,k)
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0

            final_output = train(net, img, data.labels[i], learning_rate)
            total_loss += -math.log(final_output[data.labels[i]] + 0.0000000001)
    
        correct = 0
        for i in range(train_size, data.count):
            for k in range(0, INPUT_SIZE):
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0
            if predict(net, img) == data.labels[i]:
                correct += 1
    
        print("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f, Time: %.2f seconds\n", epoch + 1, correct / test_size * 100, total_loss / train_size)


# %%
__init__()

# %%


# %%



