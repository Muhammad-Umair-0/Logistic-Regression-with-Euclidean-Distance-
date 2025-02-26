import pandas as pd
import numpy as np
from collections import Counter


# defining the Eclidean Distance 
def eculidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# Knn prediction Function
def knn_predict(training_data, training_label, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = eculidean_distance(test_point,training_data[i])
        distances.append((dist, training_label[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_label = [label for _, label in distances[:k]]
        return Counter(k_nearest_label).most_common(1)[0][0]
    

training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3

prediction = knn_predict(training_data, training_labels, test_point, k)
print(prediction)
print("ali")