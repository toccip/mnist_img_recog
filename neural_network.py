import numpy as np
import random

def sig(x):
    x = np.clip( x, -600, 600 )
    return 1.0 / (1.0 + np.exp(-x))
    
def sig_d(x):
    return sig(x) * (1.0 - sig(x))


def read_images(filename, data_size):
    img_data = list()
    with open(filename, 'rb') as f:
        f.read(4)
        img_num = f.read(4)
        row_num = f.read(4)
        col_num = f.read(4)
        for i in range(data_size):
            a_img = np.empty((784, 1))
            for j in range(784):
                a_img[j] = int.from_bytes(f.read(1), 'big') / 255.0
            img_data.append(a_img)
    return img_data

def read_labels(filename, data_size):
    label_data = list()
    with open(filename, 'rb') as f:
        f.read(4)
        count = f.read(4)
        for i in range(data_size):
            a_label = int.from_bytes(f.read(1), 'big')
            a_label_arr = np.full((10, 1), 0.0)
            a_label_arr[a_label] = 1.0
            label_data.append(a_label_arr)
    return label_data

def read_mnist(train_size, test_size):
    train_img_data = read_images('train-images', train_size)
    train_label_data = read_labels('train-labels', train_size)
    train_data = list(zip(train_img_data, train_label_data))

    test_img_data = read_images('test-images', test_size)
    test_label_data = read_labels('test-labels', test_size)
    test_data = list(zip(test_img_data, test_label_data))

    return [train_data, test_data]

class NeuralNetwork():

    def __init__(self, counts):
        
        self.counts = counts

        #One bias per node in each layer, skipping the first layer which does not need a bias
        self.biases = list()
        for i in counts[1:]:
            #self.biases.append(2 * np.random.random((i, 1)) - 1)
            self.biases.append(np.random.randn(i, 1))

        #2d array of weights between each layer. n number of weights for each node in the previous layer
        self.weights = list()
        for input_count, n_count in zip(counts[:-1], counts[1:]):
            #self.weights.append(2 * np.random.random((n_count, input_count)) - 1)
            self.weights.append(np.random.randn(n_count, input_count))

        #print('biases: \n{}'.format(self.biases))
        #print('weights: \n{}'.format(self.weights))
        
        print('---------------------')
        self.LEARNING_RATE = 0.1

    def activate(self, a):
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sig(z)
        return a

    def activate_with_layers(self, a):
        z_vals = list()
        a_vals = list()
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sig(z)
            z_vals.append(z)
            a_vals.append(a)
        return z_vals, a_vals

    #backpropagation using gradient descent
    def backprop(self, a_input, y):
        dw_arr = list()
        db_arr = list()

        a_vals, z_vals = self.activate_with_layers(a_input)
        a_vals.reverse()
        z_vals.reverse()

        for i, [al, zl] in enumerate(zip(a_vals, z_vals)):
            if (i+1) != len(a_vals):
                al_prev = a_vals[i+1]
            else:
                al_prev = a_input

            if i == 0:  
                dc_dal = np.subtract(al, y)
            
            db = dc_dal * sig_d(zl)
            dw = np.dot(db, al_prev.T)
            dc_dal = np.dot(self.weights[-(i+1)].T, db)
            dw_arr.insert(0, dw)
            db_arr.insert(0, db)
        return [dw_arr, db_arr]
        
    
    def backprop_batch(self, batch, batch_size):

        dw_final = list()
        db_final = list()
        for [a_input, y] in batch:
            
            [dw_arr, db_arr] = self.backprop(a_input, y)
            if not dw_final:
                dw_final = dw_arr
                db_final = db_arr
            else:
                dw_final = list(map(np.add, dw_final, dw_arr))
                db_final = list(map(np.add, db_final, db_arr))
        term = self.LEARNING_RATE / batch_size
        dw_final = [term * x for x in dw_final]
        db_final = [term * x for x in db_final]
        self.weights = list(map(np.subtract, self.weights, dw_final ))
        self.biases = list(map(np.subtract, self.biases, db_final))

    def train(self, inputs, outputs):

        for i in range(1):
            print('Epoch {}'.format(i))
            for in_data, out_data in zip(inputs, outputs):
                self.backprop(in_data, out_data)

    def train_batch(self, train_data, test_data=None):
        batch_size = 10
        for i in range(30):
            print('Epoch {}'.format(i))
            
            batched_data = list(zip(*(iter(train_data),) * batch_size))
            for a_batch in batched_data:
                self.backprop_batch(a_batch, batch_size)

            if test_data is not None:
                self.evaluate(test_data)

            random.shuffle(train_data)

    def evaluate(self, test_data):
        good = 0
        bad = 0
        for [img, label] in test_data:
            out = self.activate(img)
            if np.argmax(out) == np.argmax(label):
                good += 1
            else:
                bad += 1
        print('good: {}\nbad: {}'.format(good, bad))
        print('{} percent correct'.format(good / (good + bad) * 100))
        

        

if __name__ == "__main__":
    
    [train_data, test_data] = read_mnist(60000, 10000)

    my_net = NeuralNetwork([784, 25, 25, 10])
    my_net.train_batch(train_data, test_data=test_data)
    
    
