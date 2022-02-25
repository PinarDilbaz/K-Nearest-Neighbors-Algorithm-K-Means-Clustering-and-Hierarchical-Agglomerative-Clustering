import numpy as np
import matplotlib.pyplot as plt

def single_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the single linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = []
    for i in c1:
        for j in c2:
            distance = (i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2
            distances.append(distance)
    min_distance = min(distances)
    min_distance = min_distance**(1/2)

    return min_distance


def complete_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the complete linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    distances = []
    for i in c1:
        for j in c2:
            distance = (i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2
            distances.append(distance)
    max_distance = max(distances)
    max_distance = max_distance**(1/2)

    return max_distance


def average_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the average linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    summ = 0
    number_of_c1 = len(c1)
    number_of_c2 = len(c2)
    total = number_of_c1 * number_of_c2
    for i in c1:
        for j in c2:
            distance = (i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2
            distance = distance**(1/2)
            summ = summ + distance
    avg = summ/total

    return avg


def centroid_linkage(c1, c2):
    """
    Given clusters c1 and c2, calculates the centroid linkage criterion.
    :param c1: An (N, D) shaped numpy array containing the data points in cluster c1.
    :param c2: An (M, D) shaped numpy array containing the data points in cluster c2.
    :return: A float. The result of the calculation.
    """
    x = 0
    y = 0
    x1 = 0
    y1 = 0
    number_of_c1 = len(c1)
    number_of_c2 = len(c2)
    
    for i in c1:
        x = x + i[0]
    for i in c1:
        y = y + i[1]
    x = x/number_of_c1
    y = y/number_of_c1

    for i in c2:
        x1 = x1 + i[0]
    for i in c2:
        y1 = y1 + i[1]
    x1 = x1/number_of_c2
    y1 = y1/number_of_c2

    distance = (x - x1) ** 2 + (y - y1) ** 2
    distance = distance**(1/2)

    return distance


def hac(data, criterion, stop_length):
    """
    Applies hierarchical agglomerative clustering algorithm with the given criterion on the data
    until the number of clusters reaches the stop_length.
    :param data: An (N, D) shaped numpy array containing all of the data points.
    :param criterion: A function. It can be single_linkage, complete_linkage, average_linkage, or
    centroid_linkage
    :param stop_length: An integer. The length at which the algorithm stops.
    :return: A list of numpy arrays with length stop_length. Each item in the list is a cluster
    and a (Ni, D) sized numpy array.
    """
    cluster = []
    temp = list(data.flatten())
    n = int(len(temp)/2)
    x = 0
    for i in range(n):
        cluster.append([[temp[x],temp[x+1]]])
        x = x + 2
        
    number_of_cluster = len(cluster)
    while stop_length != number_of_cluster:
        distance = 0.0
        index_1 = 0
        index_2 = 0
        for i in range(number_of_cluster):
            for j in range(i + 1, number_of_cluster):

                if   criterion == single_linkage:
                    linkage = single_linkage  (cluster[i], cluster[j])

                elif criterion == complete_linkage:
                    linkage = complete_linkage(cluster[i], cluster[j])

                elif criterion == average_linkage:
                    linkage = average_linkage (cluster[i], cluster[j])

                elif criterion == centroid_linkage:
                    linkage = centroid_linkage(cluster[i], cluster[j])

                if (linkage < distance or distance == 0):
                    index_1 = i
                    index_2 = j
                    distance = linkage

        #combine clusters ​​
        combine = cluster[index_1] + cluster[index_2]
        #remove the combined clusters ​​from the cluster 
        cluster.pop(index_1)
        cluster.pop(index_2 - 1)
        #add their combined states to the cluster 
        cluster.append(combine)
        #assign new number of cluster
        number_of_cluster = len(cluster)

    for i in range(len(cluster)):
        cluster[i] = np.asarray(cluster[i])

    return cluster

if __name__ == "__main__":

    #download 4 unlabeled datasets
    
    dataset1 = np.load("C:/Users/ASUS/Desktop/hw2/hac/dataset1.npy")
    dataset2 = np.load("C:/Users/ASUS/Desktop/hw2/hac/dataset2.npy")
    dataset3 = np.load("C:/Users/ASUS/Desktop/hw2/hac/dataset3.npy")
    dataset4 = np.load("C:/Users/ASUS/Desktop/hw2/hac/dataset4.npy")

    
    ####################
    #for dataset1, k = 2
    stop_length = 2
    data = dataset1
    criterion = [single_linkage, complete_linkage, average_linkage ,centroid_linkage]
    for i in range(4):
        cluster = hac(data, criterion[i], stop_length)
        cluster1 = cluster[0]
        cluster2 = cluster[1]
        #for first cluster       
        plt.scatter(cluster1[:, 0], cluster1[:, 1], c='cyan', s=20,  cmap='viridis')
        #for second cluster 
        plt.scatter(cluster2[:, 0], cluster2[:, 1], c='hotpink', s=20,  cmap='viridis')
        #name of the x-axis
        plt.xlabel('X-axis', color='blue')
        #name of the y-axis
        plt.ylabel('Y-axis', color='blue')
        #name of the my plot
        if (i == 0):
            plt.title('Dataset-1 Single Linkage', color='darkblue')
        elif (i == 1):
            plt.title('Dataset-1 Complete Linkage', color='darkblue')
        elif (i == 2):
            plt.title('Dataset-1 Average Linkage', color='darkblue')
        elif (i == 3):
            plt.title('Dataset-1 Centroid Linkage', color='darkblue')
        #show my plot
        plt.show()
    

    ####################
    #for dataset2, k = 2
    stop_length = 2
    data = dataset2
    criterion = [single_linkage, complete_linkage, average_linkage ,centroid_linkage]
    for i in range(4):
        cluster = hac(data, criterion[i], stop_length)
        cluster1 = cluster[0]
        cluster2 = cluster[1]
        #for first cluster       
        plt.scatter(cluster1[:, 0], cluster1[:, 1], c='lime', s=20,  cmap='viridis')
        #for second cluster 
        plt.scatter(cluster2[:, 0], cluster2[:, 1], c='darkviolet', s=20,  cmap='viridis')
        #name of the x-axis
        plt.xlabel('X-axis', color='blue')
        #name of the y-axis
        plt.ylabel('Y-axis', color='blue')
        #name of the my plot
        if (i == 0):
            plt.title('Dataset-2 Single Linkage', color='darkblue')
        elif (i == 1):
            plt.title('Dataset-2 Complete Linkage', color='darkblue')
        elif (i == 2):
            plt.title('Dataset-2 Average Linkage', color='darkblue')
        elif (i == 3):
            plt.title('Dataset-2 Centroid Linkage', color='darkblue')
        #show my plot
        plt.show()
    

    ####################
    #for dataset3, k = 2
    stop_length = 2
    data = dataset3
    criterion = [single_linkage, complete_linkage, average_linkage ,centroid_linkage]
    for i in range(4):
        cluster = hac(data, criterion[i], stop_length)
        cluster1 = cluster[0]
        cluster2 = cluster[1]
        #for first cluster       
        plt.scatter(cluster1[:, 0], cluster1[:, 1], c='forestgreen', s=20,  cmap='viridis')
        #for second cluster 
        plt.scatter(cluster2[:, 0], cluster2[:, 1], c='blue', s=20,  cmap='viridis')
        #name of the x-axis
        plt.xlabel('X-axis', color='blue')
        #name of the y-axis
        plt.ylabel('Y-axis', color='blue')
        #name of the my plot
        if (i == 0):
            plt.title('Dataset-3 Single Linkage', color='darkblue')
        elif (i == 1):
            plt.title('Dataset-3 Complete Linkage', color='darkblue')
        elif (i == 2):
            plt.title('Dataset-3 Average Linkage', color='darkblue')
        elif (i == 3):
            plt.title('Dataset-3 Centroid Linkage', color='darkblue')
        #show my plot
        plt.show()
    

    ####################
    #for dataset4, k = 4
    stop_length = 4
    data = dataset4
    criterion = [single_linkage, complete_linkage, average_linkage ,centroid_linkage]
    for i in range(4):
        cluster = hac(data, criterion[i], stop_length)
        cluster1 = cluster[0]
        cluster2 = cluster[1]
        cluster3 = cluster[2]
        cluster4 = cluster[3]
        #for first cluster       
        plt.scatter(cluster1[:, 0], cluster1[:, 1], c='orange', s=20,  cmap='viridis')
        #for second cluster 
        plt.scatter(cluster2[:, 0], cluster2[:, 1], c='mediumspringgreen', s=20,  cmap='viridis')
        #for third cluster 
        plt.scatter(cluster3[:, 0], cluster3[:, 1], c='fuchsia', s=20,  cmap='viridis')
        #for fourth cluster 
        plt.scatter(cluster4[:, 0], cluster4[:, 1], c='skyblue', s=20,  cmap='viridis')
        #name of the x-axis
        plt.xlabel('X-axis', color='blue')
        #name of the y-axis
        plt.ylabel('Y-axis', color='blue')
        #name of the my plot
        if (i == 0):
            plt.title('Dataset-4 Single Linkage', color='darkblue')
        elif (i == 1):
            plt.title('Dataset-4 Complete Linkage', color='darkblue')
        elif (i == 2):
            plt.title('Dataset-4 Average Linkage', color='darkblue')
        elif (i == 3):
            plt.title('Dataset-4 Centroid Linkage', color='darkblue')
        #show my plot
        plt.show()