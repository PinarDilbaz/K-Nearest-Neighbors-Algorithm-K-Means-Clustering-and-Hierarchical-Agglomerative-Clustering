import numpy as np
import matplotlib.pyplot as plt

def initialize_clusters(data, k):
    """
    I used "Forgy Initialization Method" to initialize my center clusters
    """
    initial_cluster_centers = data [np.random.choice(range(len(data)), replace = False,     size = k), :]
    print(initial_cluster_centers)
    return initial_cluster_centers

def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    arr = []
    distances = []
    arr_index = []
    m = 0
    d = 10.0
    number_of_data = len(data)
    number_of_clusters = len(cluster_centers)
    for i in range(number_of_data):
        m = m - number_of_clusters
        d = 10.0
        for j in range(number_of_clusters):
            m = m + 1
            #calculate euclidean distance using linalg.norm
            euclidean_distance = np.linalg.norm(data[i] - cluster_centers[j])
            distances.append(euclidean_distance)
            if (euclidean_distance < d):
                d = euclidean_distance
                #for index
                keep = j
            
            if (m == 0) :
                minimum_dist = min(distances)
                #minimum distances array
                arr.append(minimum_dist)
                #index of the closest center array
                arr_index.append(keep)

    return np.asarray(arr_index) 

def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    new_cluster_centers = []
    m = 0  
    
    number_of_data = len(data)

    for i in range(k):
        m = m - number_of_data
        value = 0
        summ = 0
        for j in range(number_of_data):
            m = m + 1
            if (i == assignments[j]):
                summ = summ + data[j]
                value = value + 1
            if (m ==0):
                if(value == 0):
                    new_cluster_centers.append(cluster_centers[i])
                else:
                    avg = summ/value
                    new_cluster_centers.append(avg)

    return (np.asarray(new_cluster_centers))

def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
    number_of_data = len(data)
    number_of_clusters = len(initial_cluster_centers)
    temp = initial_cluster_centers
    change = 0
    summ = 0
    cluster_centers = initial_cluster_centers
    while (change != 1):
        assignments = assign_clusters(data, cluster_centers)
        cluster_centers = calculate_cluster_centers(data, assignments, cluster_centers, number_of_clusters)

        if (np.array_equal(cluster_centers,temp)):
            change = 1   
        temp = cluster_centers
    
    for i in range(number_of_data):
        #summ = summ + (np.linalg.norm(data[i] - cluster_centers[assignments[i]])) **2
        distance = (data[i][0] - cluster_centers[assignments[i]][0])**2 + (data[i][1] - cluster_centers[assignments[i]][1])**2
        summ = summ + distance
        objective_function = float(summ/2)
    #print(objective_function) 
    return np.asarray(cluster_centers), objective_function

def elbow(objective_list, k):

    #drow a plot showing x-axis is k value and y-axis is objective function value
    plt.plot(k, objective_list, color='magenta', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='green', markersize=4)
    #name of the x-axis
    plt.xlabel('K Value', color='blue')
    #name of the y-axis
    plt.ylabel('Objective Function Value', color='blue')
    #name of the my plot
    plt.title('Elbow Method', color='black')
    #show the plot
    plt.show()


if __name__ == "__main__":

    #download 4 unlabeled datasets
    
    dataset1 = np.load("C:/Users/ASUS/Desktop/hw2/kmeans/dataset1.npy")
    dataset2 = np.load("C:/Users/ASUS/Desktop/hw2/kmeans/dataset2.npy")
    dataset3 = np.load("C:/Users/ASUS/Desktop/hw2/kmeans/dataset3.npy")
    dataset4 = np.load("C:/Users/ASUS/Desktop/hw2/kmeans/dataset4.npy")

    k = list(range(1, 11))
    objective_list = []

    """
    We use the elbow method to understand which k value is the most appropriate and I applied it for all datasets in the code below. 
    I leave it as a comment because it takes too long.

    for i in k:
        initial_cluster_centers = initialize_clusters(dataset1, i)
        cluster_centers, objective_function = kmeans(dataset1, initial_cluster_centers)
        objective_list.append(objective_function)
    elbow(objective_list, k)
    
    for i in k:
        initial_cluster_centers = initialize_clusters(dataset2, i)
        cluster_centers, objective_function = kmeans(dataset2, initial_cluster_centers)
        objective_list.append(objective_function)
    elbow(objective_list, k)

    for i in k:
        initial_cluster_centers = initialize_clusters(dataset3, i)
        cluster_centers, objective_function = kmeans(dataset3, initial_cluster_centers)
        objective_list.append(objective_function)
    elbow(objective_list, k)
    
    for i in k:
        initial_cluster_centers = initialize_clusters(dataset4, i)
        cluster_centers, objective_function = kmeans(dataset4, initial_cluster_centers)
        objective_list.append(objective_function)   
    elbow(objective_list, k)
    """
    
    
    ####################################################################################################
    #for dataset-1, with the elbow method, we understood that the optimal k value is 2, so I take k as 2. 
    initial_cluster_centers = initialize_clusters(dataset1, 2)
    cluster_centers, objective_function = kmeans(dataset1, initial_cluster_centers)
    index = assign_clusters(dataset1, cluster_centers)
    cluster1 = []
    cluster2 = []
    for i in range(len(index)):
        if (index[i] == 0):
            cluster1.append(dataset1[i])
            cluster_c1 = cluster_centers[index[i]]
        elif (index[i] == 1):
            cluster2.append(dataset1[i])
            cluster_c2 = cluster_centers[index[i]]
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    #for first cluster       
    plt.scatter(cluster1[:, 0], cluster1[:, 1], c='cyan', s=20,  cmap='viridis')
    #for first cluster center
    plt.plot(cluster_c1[0], cluster_c1[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for second cluster 
    plt.scatter(cluster2[:, 0], cluster2[:, 1], c='pink', s=20,  cmap='viridis')
    #for second cluster center
    plt.plot(cluster_c2[0], cluster_c2[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)

    #name of the x-axis
    plt.xlabel('X-axis', color='blue')
    #name of the y-axis
    plt.ylabel('Y-axis', color='blue')
    #name of the my plot
    plt.title('Colorization of the Dataset-1', color='magenta')

    #show my plot
    plt.show()
    

    ####################################################################################################
    #for dataset-2, with the elbow method, we understood that the optimal k value is 2, so I take k as 3. 
    initial_cluster_centers = initialize_clusters(dataset2, 3)
    cluster_centers, objective_function = kmeans(dataset2, initial_cluster_centers)
    index = assign_clusters(dataset2, cluster_centers)
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i in range(len(index)):
        if (index[i] == 0):
            cluster1.append(dataset2[i])
            cluster_c1 = cluster_centers[index[i]]
        elif (index[i] == 1):
            cluster2.append(dataset2[i])
            cluster_c2 = cluster_centers[index[i]]
        elif (index[i] == 2):
            cluster3.append(dataset2[i])
            cluster_c3 = cluster_centers[index[i]]
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    cluster3 = np.asarray(cluster3)
    #for first cluster       
    plt.scatter(cluster1[:, 0], cluster1[:, 1], c='skyblue', s=20, cmap='viridis')
    #for first cluster center
    plt.plot(cluster_c1[0], cluster_c1[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for second cluster 
    plt.scatter(cluster2[:, 0], cluster2[:, 1], c='tomato', s=20 , cmap='viridis')
    #for second cluster center
    plt.plot(cluster_c2[0], cluster_c2[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for third cluster 
    plt.scatter(cluster3[:, 0], cluster3[:, 1], c='greenyellow', s=20, cmap='viridis')
    #for third cluster center
    plt.plot(cluster_c3[0], cluster_c3[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)

    #name of the x-axis
    plt.xlabel('X-axis', color='blue')
    #name of the y-axis
    plt.ylabel('Y-axis', color='blue')
    #name of the my plot
    plt.title('Colorization of the Dataset-2', color='darkblue')

    #show my plot
    plt.show()
    

    ####################################################################################################
    #for dataset-3, with the elbow method, we understood that the optimal k value is 2, so I take k as 4. 
    initial_cluster_centers = initialize_clusters(dataset3, 4)
    cluster_centers, objective_function = kmeans(dataset3, initial_cluster_centers)
    index = assign_clusters(dataset3, cluster_centers)
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    for i in range(len(index)):
        if (index[i] == 0):
            cluster1.append(dataset3[i])
            cluster_c1 = cluster_centers[index[i]]
        elif (index[i] == 1):
            cluster2.append(dataset3[i])
            cluster_c2 = cluster_centers[index[i]]
        elif (index[i] == 2):
            cluster3.append(dataset3[i])
            cluster_c3 = cluster_centers[index[i]]
        elif (index[i] == 3):
            cluster4.append(dataset3[i])
            cluster_c4 = cluster_centers[index[i]]
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    cluster3 = np.asarray(cluster3)
    cluster4 = np.asarray(cluster4)
    #for first cluster       
    plt.scatter(cluster1[:, 0], cluster1[:, 1], c='deeppink', s=20, cmap='viridis')
    #for first cluster center
    plt.plot(cluster_c1[0], cluster_c1[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for second cluster 
    plt.scatter(cluster2[:, 0], cluster2[:, 1], c='orange', s=20 , cmap='viridis')
    #for second cluster center
    plt.plot(cluster_c2[0], cluster_c2[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for third cluster 
    plt.scatter(cluster3[:, 0], cluster3[:, 1], c='lightcoral', s=20, cmap='viridis')
    #for third cluster center
    plt.plot(cluster_c3[0], cluster_c3[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for fourth cluster 
    plt.scatter(cluster4[:, 0], cluster4[:, 1], c='darkkhaki', s=20, cmap='viridis')
    #for fourth cluster center
    plt.plot(cluster_c4[0], cluster_c4[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)

    #name of the x-axis
    plt.xlabel('X-axis', color='blue')
    #name of the y-axis
    plt.ylabel('Y-axis', color='blue')
    #name of the my plot
    plt.title('Colorization of the Dataset-3', color='darkblue')

    #show my plot
    plt.show()

    ####################################################################################################
    #for dataset-4, with the elbow method, we understood that the optimal k value is 2, so I take k as 4. 
    initial_cluster_centers = initialize_clusters(dataset4, 4)
    cluster_centers, objective_function = kmeans(dataset4, initial_cluster_centers)
    index = assign_clusters(dataset4, cluster_centers)
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    #cluster5 = []
    for i in range(len(index)):
        if (index[i] == 0):
            cluster1.append(dataset4[i])
            cluster_c1 = cluster_centers[index[i]]
        elif (index[i] == 1):
            cluster2.append(dataset4[i])
            cluster_c2 = cluster_centers[index[i]]
        elif (index[i] == 2):
            cluster3.append(dataset4[i])
            cluster_c3 = cluster_centers[index[i]]
        elif (index[i] == 3):
            cluster4.append(dataset4[i])
            cluster_c4 = cluster_centers[index[i]]
        """
        elif (index[i] == 4):
            cluster5.append(dataset4[i])
            cluster_c5 = cluster_centers[index[i]]
        """
    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    cluster3 = np.asarray(cluster3)
    cluster4 = np.asarray(cluster4)
    #cluster5 = np.asarray(cluster5)
    #for first cluster       
    plt.scatter(cluster1[:, 0], cluster1[:, 1], c='springgreen', s=20, cmap='viridis')
    #for first cluster center
    plt.plot(cluster_c1[0], cluster_c1[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for second cluster 
    plt.scatter(cluster2[:, 0], cluster2[:, 1], c='gold', s=20 , cmap='viridis')
    #for second cluster center
    plt.plot(cluster_c2[0], cluster_c2[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for third cluster 
    plt.scatter(cluster3[:, 0], cluster3[:, 1], c='turquoise', s=20, cmap='viridis')
    #for third cluster center
    plt.plot(cluster_c3[0], cluster_c3[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for fourth cluster 
    plt.scatter(cluster4[:, 0], cluster4[:, 1], c='mediumorchid', s=20, cmap='viridis')
    #for fourth cluster center
    plt.plot(cluster_c4[0], cluster_c4[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)
    #for fifth cluster 
    #plt.scatter(cluster5[:, 0], cluster5[:, 1], c='coral', s=20, cmap='viridis')
    #for fifth cluster center
    #plt.plot(cluster_c5[0], cluster_c5[1], color='magenta', marker='d', markeredgecolor='black', markersize=13)

    #name of the x-axis
    plt.xlabel('X-axis', color='blue')
    #name of the y-axis
    plt.ylabel('Y-axis', color='blue')
    #name of the my plot
    plt.title('Colorization of the Dataset-4', color='darkblue')

    #show my plot
    plt.show()
 