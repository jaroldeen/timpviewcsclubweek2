"""
TicTacToe Analyst

Problem Statement:
Given a Tic Tac Toe board, use a neural net to determine which person wins.
Output 1 if "X" wins, -1 if "O" wins, and 0 if no one wins.

Note: you will probably need a wrapper function to convert the output from
your neural net into -1, 0, or 1, as an answer of 0.9997 will be considered
wrong if it is supposed to be 1.
"""

import random
import numpy

def sigmoid(x):
    return 1/(1+numpy.exp(-x))

def sigmoid_prime(x):
    e = numpy.exp(-x)
    return e/(1+e)**2

def leaky_relu(x):
    return numpy.maximum(x, 0.1*x)


def generate_random_board():
    """Returns the inputs and the winner. The inputs are defined below."""

    # List of winning indices; if one player has all of a group of three, they win.
    winning_positions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
                         (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    # Blank board
    inputs = [0 for i in range(9)]

    # List of which spots are open to place in.
    open = list(range(9))

    # X starts the game
    player = 1

    # Letter to input value.
    while len(open) > 0:
        # Choose a random spot and place in it.
        i = random.choice(open)
        inputs[i] = player
        
        # This spot is no longer open
        open.remove(i)

        # Check if they win
        for pos in winning_positions:
            if all(inputs[p] == player for p in pos):
                return inputs, player
        
        # Switch whose turn it is.
        player *= -1
    
    # No one won, it must have been a draw.
    return inputs, 0
            

def test_function(f):
    """
    f(inputs) should give the correct output (-1, 0, or 1) depending on the
    inputs. The inputs will look similar to:

        [1, -1, -1, -1, 1, 1, 0, 0, 1]

    The first 3 values correspond to the first row of the tictactoe board,
    the next 3 values correspond to the middle row, and the last 3 values
    correspond to the last row. A 0 means no one placed in that square, a
    1 means X placed in the square, and a -1 means O placed in the square.

    So, the example above would correspond to this tictactoe board:
    
        X|O|O
        ------
        O|X|X
        ------
         | |X

    In this example X wins, so your function should output 1.

    """
    inputs, output = generate_random_board()
    return f(inputs) == output

class NeuralNet():
    def generate_biases(layers):
        biases = []
        for layer in layers[1:]:
            biases.append(numpy.random.rand(layer))
        return biases

    def generate_weights(layers):
        weights = []
        for i, layer in enumerate(layers[:-1]):
            weights.append(numpy.random.rand(layer, layers[i+1]))
        return weights
    
    def __init__(self, layers, function):
        self.layers = layers
        self.function = function
        self.biases = [numpy.random.rand(layers[i])*2-1 for i in range(1, len(layers))]
        self.weights = [numpy.random.rand(layers[i], layers[i+1])*2-1
                   for i in range(len(layers)-1)]

    def evaluate(self, inputs):
        for i, layer_weights in enumerate(self.weights):
            inputs = self.function(numpy.matmul(inputs, layer_weights) + self.biases[i])
        return inputs

    def error(self, inputs_set, outputs_set):
        responses = numpy.array([self.evaluate(inputs) for inputs in inputs_set])
        print(responses)
        error = numpy.linalg.norm(responses - outputs_set)
        return error

    def train(self, inputs_set, outputs_set, step, dx):
        for i, layer_weights in enumerate(self.weights):
            for j, node_weights in enumerate(layer_weights):
                for k, weight in enumerate(node_weights):
                    error = self.error(inputs_set, outputs_set)
                    self.weights[i][j][k] += dx
                    new_error = self.error(inputs_set, outputs_set)
                    self.weights[i][j][k] -= dx + step*(new_error - error)/dx
        for i, layer_biases in enumerate(self.biases):
            error = self.error(inputs_set, outputs_set)
            self.biases[i] += dx
            new_error = self.error(inputs_set, outputs_set)
            self.biases[i] -= dx + step*(new_error - error)/dx

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    numpy.random.seed(1)
    nn = NeuralNet([9,5,3,1], function=leaky_relu)
    inputs_set = []
    outputs_set = []
    for i in range(15):
        inputs, output = generate_random_board()
        inputs_set.append(inputs)
        outputs_set.append(output)
        i += 1
    
    def output(nn):
        l = []
        for inputs in inputs_set:
            l.append('{}'.format(str(nn.evaluate(inputs)[0])))
        print(", ".join(l))

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Error of Neural Net")
    x, y = [], []
    Ln, = ax.plot(x, y)

    plt.draw()

    i = 0
    while True:
        error = nn.error(inputs_set, outputs_set)
        nn.train(inputs_set, outputs_set, step=error/100, dx=error/1000)

        x.append(i)
        y.append(error)

        if(i%100 == 0):            
            Ln.set_xdata(x)
            Ln.set_ydata(y)
            k = min(y[0], y[max(0, i-100)]*10)
            start = 0
            while y[start] > k:
                start += 1
            plt.xlim(start, i+5)
            plt.ylim(0, k)
            fig.canvas.draw_idle()
            plt.pause(0.001)
        
        if(i%300 == 0):
            output(nn)
        i += 1