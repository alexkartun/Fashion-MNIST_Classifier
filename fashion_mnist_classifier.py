import numpy as np

np.random.seed(42)      # setting the seed

# hyper parameters
n_class = 10
n_iter = 8192
lr = 5e-4
dropout = .7
reg = 8e-2
mu = .9
minibatch_size = 4096


def make_network(n_feature, n_hidden=128, std=1e-4):
    """
    creating mlp, NN with 1 hidden layer
    :param n_feature: number of features
    :param n_hidden: number of hidden units
    :param std: normalization term
    :return: dictionary with all network's layers
    """
    return dict(
        W1=std*np.random.randn(n_feature, n_hidden),
        b1=np.zeros((1, n_hidden)),
        W2=std*np.random.randn(n_hidden, n_class),
        b2=np.zeros((1, n_class))
    )


def forward(model, X):
    """
    feed forward step which feeding outputs of each layer to the next layers
    :param model: nn model
    :param X: data
    :return: dictionary with all outputs of the layers
    """
    h = np.dot(X, model['W1']) + model['b1']
    h[h < 0] = 0
    u = np.random.binomial(1, dropout, size=h.shape)
    h *= u
    probs = softmax(np.dot(h, model['W2']) + model['b2'])
    return {'h': h, 'u': u, 'probs': probs}


def NLL(model, cache, y):
    """
    negative log likelihood is network's loss function that we want to minimize
    :param model: nn model
    :param cache: cache data computed in feed forward step
    :param y: labels
    :return: loss + regularization loss
    """
    correct_logprobs = -np.log(cache['probs'][range(y.shape[0]), y])
    data_loss = np.sum(correct_logprobs) / y.shape[0]
    reg_loss = (reg / (2 * y.shape[0])) * (np.sum(model['W1'] * model['W1']) + np.sum(model['W2'] * model['W2']))
    return data_loss + reg_loss


def backward(model, cache, X, y):
    """
    back propagation step which calculating the gradients of each one the network's parameters with respect
    to the loss function
    :param model: nn model
    :param cache: cache data computed in feed forward step
    :param X: data
    :param y: labels
    :return: gradients
    """
    dScores = cache['probs']
    dScores[range(X.shape[0]), y] -= 1
    dScores /= X.shape[0]
    dW2 = np.dot(cache['h'].T, dScores)
    db2 = np.sum(dScores, axis=0, keepdims=True)

    dh = np.dot(dScores, model['W2'].T)
    dh[cache['h'] <= 0] = 0
    dh *= cache['u']
    dW1 = np.dot(X.T, dh)
    db1 = np.sum(dh, axis=0, keepdims=True)

    dW2 += reg * model['W2']
    dW1 += reg * model['W1']

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


def train(model, X, y):
    """
    training step of your network, loop for fixed number of iteration, choose random minibatch, feed forward the data,
    back propagate the error and calculate the gradients and finally optimize the network parameters by SGD with
    momentum
    :param model: nn model
    :param X: data
    :param y: labels
    :return: None
    """
    velocity = {k: np.zeros_like(v) for k, v in model.items()}

    minibatches = get_minibatches(X, y)
    for _ in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        cache = forward(model, X_mini)
        _ = NLL(model, cache, y_mini)
        grads = backward(model, cache, X_mini, y_mini)

        for layer in grads:
            velocity[layer] = mu * velocity[layer] - lr * grads[layer]
            model[layer] += velocity[layer]


def get_minibatches(X, y):
    """
    shuffle the training dataset and extract all minibatches from training dataset with respect to minibatch size that
    was set before
    :param X: data
    :param y: labels
    :return: list of all minibatches
    """
    minibatches = []
    X, y = shuffle(X, y)
    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i: i + minibatch_size]
        y_mini = y[i: i + minibatch_size]
        minibatches.append((X_mini, y_mini))
    return minibatches


def shuffle(X, y):
    """
    shuffling the training dataset
    :param X: data
    :param y: labels
    :return: new shuffled training dataset
    """
    Z = np.column_stack((X, y))
    np.random.shuffle(Z)
    return Z[:, :-1], Z[:, -1]


def predict(model, X):
    """
    predicting step of our network on dataset X, feed forward the data to the last layer, and now take the maximal
    predicted class, do not calculate softmax cause of expensive operation
    :param model: nn model
    :param X: dataset
    :return: all predicted values for each example in X
    """
    h = np.dot(X, model['W1']) + model['b1']
    h[h < 0] = 0
    y_hat = np.argmax(np.dot(h, model['W2']) + model['b2'], axis=1)
    return y_hat


def softmax(scores):
    """
    softmax function, taking scores from last nn layer and normalizing the vector
    :param scores: scores we want to calculate probs for
    :return: probs of scores
    """
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def load():
    """
    loading the training dataset and test set
    :return: training dataset and test set
    """
    train_x = np.loadtxt("train_x", dtype=np.uint8)
    train_y = np.loadtxt("train_y", dtype=np.uint8)
    test_x = np.loadtxt("test_x", dtype=np.uint8)
    return train_x, train_y, test_x


def save(test_y):
    """
    saving predicted values of test set
    :param test_y: model's predicted values on test set
    :return: None
    """
    np.savetxt('test_y', test_y, fmt='%u')


def main():
    """
    main function of our exercise 3
    :return: None
    """
    train_x, train_y, test_x = load()

    model = make_network(train_x.shape[1])
    train(model, train_x, train_y)
    test_y = predict(model, test_x)
    save(test_y)


if __name__ == '__main__':
    main()
