def linclass(weight, bias, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    # Perform linear classification i.e. class prediction
    class_pred = []
    for i in range(len(data)):
        res = 0
        for j in range(len(data[i][:])):
            res += weight[j]*data[i][j]
        res += bias
        if res > 0:
            classified = 1
        else:
            classified = -1
        class_pred.append(classified)


    return class_pred


