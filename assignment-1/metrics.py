def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    # assert(y_hat.size == y.size)
    assert(len(y_hat) == len(y))

    # TODO: Write here
    accuracy_of_data = 0.0
    # print(type(accuracy_of_data))

    n = len(y_hat)
    # n =
    for i in range(n):
        accuracy_of_data = accuracy_of_data + (y_hat[i] == y[i])

    accuracy_of_data = accuracy_of_data/n

    # print(accuracy_of_data)
    return accuracy_of_data

    pass

# COMPLETE FROM HERE


def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """

    # assert(y_hat.size == y.size)
    assert(len(y_hat) == len(y))

    final_precision = 0.0

    n = len(y_hat)
    count_of_desired_classes = 0
    for i in range(n):
        count_of_desired_classes = count_of_desired_classes + (y_hat[i] == cls)
        final_precision = final_precision + \
            (y[i] == y_hat[i] and y_hat[i] == cls)

    # print("NODC IS", count_of_desired_classes)
    # print("FINAL PREC is ", final_precision)
    if (final_precision == count_of_desired_classes and final_precision == 0):
        return 0
    final_precision = final_precision/count_of_desired_classes

    return final_precision


def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """

    # assert(y_hat.size == y.size)
    assert(len(y_hat) == len(y))

    n = len(y_hat)

    final_recall = 0.0
    relevant_instances = 0

    for i in range(n):
        relevant_instances = relevant_instances + (y[i] == cls)
        final_recall = final_recall + (y[i] == cls and y[i] == y_hat[i])

    final_recall = final_recall/relevant_instances
    return final_recall

    pass


def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    # assert(y_hat.size == y.size)
    # print(y)
    # print(y_hat)
    assert(len(y_hat) == len(y))

    root_mean_squared_error = 0.0
    n = len(y_hat)

    mean_squared_error = 0.0

    for i in range(n):
        to_be_added = (y[i] - y_hat[i])**2
        mean_squared_error = mean_squared_error + to_be_added

    mean_squared_error = mean_squared_error/n

    root_mean_squared_error = (mean_squared_error)**0.5

    return root_mean_squared_error

    pass


def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """

    # assert(y_hat.size == y.size)

    assert(len(y_hat) == len(y))

    n = len(y_hat)
    mean_absolute_error = 0.0
    for i in range(n):
        mean_absolute_error = mean_absolute_error + abs(y[i] - y_hat[i])

    mean_absolute_error = mean_absolute_error/n

    return mean_absolute_error
    pass

# DONE


def getPredictionMostOccuringClass(y):
    findMax = {}
    maxOccurence = -1
    for i in y:
        if i in findMax:
            findMax[i] += 1
        else:
            findMax[i] = 1

        maxOccurence = max(maxOccurence, findMax[i])

    ans = -1
    for i in findMax:
        if (findMax[i] == maxOccurence):
            ans = i

    return ans