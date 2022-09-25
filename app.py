from functions import *

bias = 0.5
l_rate = 0.1


data, weights = generate_data(4, 3)
for i in range(len(data)):
    feature = data.loc[i][:-1]
    target = data.loc[i][-1]
    w_sum = get_weighted_sum(feature, weights, bias)
    prediction = sigmoid(w_sum)
    loss = cross_entropy(target, prediction)
    # Gradient descent
    weights = update_weights(weights, l_rate, target, prediction, feature)
    bias = update_bias(bias, l_rate, target, prediction)



