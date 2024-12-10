import numpy as np


# GRADIENTDESCENTMULTI Performs gradient descent to learn theta
#   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
#   taking num_iters gradient steps with learning rate alpha
from src.utils.compute_cost import compute_cost


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.
        h = X.dot(theta)
        error = h - y
        gradient = 1 / m * X.T.dot(error)
        theta = theta - alpha * gradient  # update theta

        J_history[i] = compute_cost(X, y, theta)    # save the cost

    return theta, J_history
