import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def relu(x):
    return np.maximum(x, 0.)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def element_predict(input_data, weights):
    act = [relu for _ in weights]
    act[-1] = sigmoid
    inp = input_data.copy()
    for d in range(0, len(weights)):
        res = np.zeros((inp.shape[0], weights[d][0].shape[1]))
        for i in range(0, inp.shape[0]):
            for j in range(0, weights[d][0].shape[1]):
                s = 0
                for k in range(0, inp.shape[1]):
                    s += inp[i][k] * weights[d][0][k][j]
                res[i][j] = act[d](s + weights[d][1][j])
        inp = res
    return inp


def numpy_predict(input_data, weights):
    act = [relu for _ in weights]
    act[-1] = sigmoid
    res = input_data.copy()
    for i in range(0, len(weights)):
        res = act[i](np.dot(res, weights[i][0]) + weights[i][1])
    return res


def print_predicts(_model, dataset):
    weights = []
    for layer in _model.layers:
        weights.append(layer.get_weights())
    element_wise_res = element_predict(dataset, weights)
    tensor_res = numpy_predict(dataset, weights)
    model_res = _model.predict(dataset)
    assert np.isclose(element_wise_res, model_res).all()
    assert np.isclose(tensor_res, model_res).all()
    print("Поэлементно")
    print(element_wise_res)
    print("Из NumPy")
    print(tensor_res)
    print("Предсказание модели")
    print(model_res)


def logic_func(a, b, c):
    return (a or b) != (not (b and c))


train_data = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]])
validation_data = np.array([int(logic_func(*x)) for x in train_data])
model = Sequential()
model.add(Dense(6, activation='relu', input_shape=(3,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print_predicts(model, train_data)
model.fit(train_data, validation_data, epochs=150, batch_size=1)
print_predicts(model, train_data)
