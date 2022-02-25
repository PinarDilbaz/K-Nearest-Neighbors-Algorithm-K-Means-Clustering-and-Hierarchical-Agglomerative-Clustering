import numpy as np
import matplotlib.pyplot as plt

def calculate_distances(train_data, test_instance, distance_metric):
    """
    Calculates Manhattan (L1) / Euclidean (L2) distances between test_instance and every train instance.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data.
    :param test_instance: A (D, ) shaped numpy array.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: An (N, ) shaped numpy array that contains distances.
    """
    distances = []
    summ = 0
    if (distance_metric == 'L1'):
        for i in range(len(train_data)):
            #manhattan_distance = np.abs(train_data[i][0] - test_instance[0]) + np.abs(train_data[i][1] - test_instance[1])
            manhattan_distance = np.abs(train_data[i] - test_instance).sum()
            distances.append(manhattan_distance)
            
    elif (distance_metric == 'L2'):
        
        for i in range(len(train_data)):
            #euclidean_distance = ((train_data[i][0]-test_instance[0])**2 + (train_data[i][1] - test_instance[1])**2)**(1/2)
            euclidean_distance = np.linalg.norm(train_data[i] - test_instance)
            distances.append(euclidean_distance)
            
       
    return np.asarray(distances)


def majority_voting(distances, labels, k):
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """
    distance_index = []
    label_arr = []
    d = []
    sorted_distances = sorted(distances)
    for distance in distances:
        d.append(distance)
    for dist in sorted_distances:
        distance_index.append(d.index(dist))
    distance_index = distance_index [:k]
    for index in distance_index:
        x = labels[index]
        label_arr.append(x)
    sorted_labels = sorted(label_arr)
    label = max(sorted_labels, key=sorted_labels.count)
    
    return label

def knn(train_data, train_labels, test_data, test_labels, k, distance_metric):
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. The calculated accuracy.
    """
    summ = 0
    arr = []
    numberOfTestLabels = len(test_labels)

    for i in range(len(test_data)):
        distances = calculate_distances(train_data, test_data[i], distance_metric)
        label = majority_voting(distances, train_labels, k)
        arr.append(label)
    
    for i in range(len(test_data)):
        if test_labels[i] == arr[i]:
            summ = summ + 1

    accuracy = float(summ/len(test_data))
    
    return accuracy


def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """
    size = int(len(whole_train_data)/k_fold)
    d = np.shape(whole_train_data)[1]
    number_of_labels = len(whole_train_labels)
    
    #split to whole_train_data and whole_train_labels into k_fold 
    train_split = np.array_split(whole_train_data, k_fold)
    label_split = np.array_split(whole_train_labels, k_fold)
    #for validation dataset we take validation_indexth value
    validation_data = train_split[validation_index]
    validation_labels = label_split[validation_index]
    #remove the item which is index validation index in splitting dataset for creating training dataset
    train_data = np.delete(train_split, validation_index, axis=0)
    train_labels = np.delete(label_split, validation_index, axis=0)
    
    train_shape = number_of_labels - int(number_of_labels/k_fold)
    validation_shape = int(number_of_labels/k_fold)
    #for reshape
    train_labels = train_labels.reshape(train_shape, )
    train_data = train_data.reshape(train_shape, d)
    validation_data = validation_data.reshape(validation_shape, d)
    validation_labels = validation_labels.reshape(validation_shape, )
    
    return train_data, train_labels, validation_data, validation_labels

def cross_validation(whole_train_data, whole_train_labels, k_fold, k, distance_metric):
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k_fold: An integer.
    :param k: An integer. The number of nearest neighbor to be selected.
    :param distance_metric: A string which indicates the distance metric, it can be either 'L1' or 'L2'
    :return: A float. Average accuracy calculated.
    """
    accuracy = []
    summ = 0
    for validation_index in range(k_fold):
        train_data, train_labels, validation_data, validation_labels = split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold)
        acc = knn(train_data, train_labels, validation_data, validation_labels, k, distance_metric)
        summ = summ + acc
    
    average_accuracy = float(summ/k_fold)   
    return average_accuracy


 #main method starts   
if __name__ == "__main__":

    #download 4 unlabeled datasets
    
    test_labels = np.load("C:/Users/ASUS/Desktop/hw2/knn/test_labels.npy")
    test_data = np.load("C:/Users/ASUS/Desktop/hw2/knn/test_set.npy")
    train_labels = np.load("C:/Users/ASUS/Desktop/hw2/knn/train_labels.npy")
    train_data = np.load("C:/Users/ASUS/Desktop/hw2/knn/train_set.npy")

    arr_accuracy = []
    k = list(range(1, 180))
    print("Press 1 to use Manhattan distance")
    print("Press 2 to use Euclidean distance")
    distance_metric = int(input("Enter your choice: "))

    if (distance_metric == 1):
        print("Please wait for the graph, it may take a while!")
        for i in range(1,180):
            #kfold is 10 and k is changing 1 to 179
            accuracy = cross_validation(train_data, train_labels, 10, i, 'L1')
            arr_accuracy.append(accuracy)
        #drow a plot showing x-axis is k value and y-axis is average accuracy  
        plt.plot(k, arr_accuracy, color='lime', linestyle='dashed', linewidth = 2,
            marker='o', markerfacecolor='darkviolet', markersize=4)
        #name of the x-axis
        plt.xlabel('K Value', color='magenta')
        #name of the y-axis
        plt.ylabel('Average Accuracy', color='magenta')
        #name of the my plot
        plt.title('K-Nearest Neighbors with L1', color='deeppink')
        #show the plot
        plt.show()

        #find the maximum average accuracies and the best k value
        maximum_accuracy = max (arr_accuracy)
        index = arr_accuracy.index(maximum_accuracy)
        best_k = k[index]
        test_accuracy = knn(train_data, train_labels, test_data, test_labels, best_k, 'L1')
        print("The test accuracy is: ",test_accuracy)
        print("The best K value is: ",best_k)

    elif (distance_metric == 2):
        print("Please wait for the graph, it may take a while!")
        for i in range(1,180):
            #kfold is 10 and k is changing 1 to 179
            accuracy = cross_validation(train_data, train_labels, 10, i, 'L2')
            arr_accuracy.append(accuracy)
        #drow a plot showing x-axis is k value and y-axis is average accuracy  
        plt.plot(k, arr_accuracy, color='lime', linestyle='dashed', linewidth = 2,
            marker='o', markerfacecolor='darkviolet', markersize=4)
        #name of the x-axis
        plt.xlabel('K Value', color='magenta')
        #name of the y-axis
        plt.ylabel('Average Accuracy', color='magenta')
        #name of the my plot
        plt.title('K-Nearest Neighbors with L2', color='deeppink')
        #show the plot
        plt.show()

        #find the maximum average accuracies and the best k value
        maximum_accuracy = max (arr_accuracy)
        index = arr_accuracy.index(maximum_accuracy)
        best_k = k[index]
        test_accuracy = knn(train_data, train_labels, test_data, test_labels, best_k, 'L2')
        print("The test accuracy is: ",test_accuracy)
        print("The best K value is: ",best_k)
        
    


