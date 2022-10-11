import numpy as np
import numpy.random as rd
import numpy.linalg
import time

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ADD_CONST = False

GROUP_PIVOTS = (
    [33, 33, 46, 107, 66, 113, 110, 124, 36, 81, 82, 75, 39, 44, 69, 58, 124, 52, 110, 100, 115, 88, 52, 98, 61, 117, 124, 118, 56, 91, 89, 49, 35, 38, 74, 67, 71, 67, 48, 67, 121, 43, 87, 44, 65, 66, 107, 109, 36, 65, 55, 125, 91, 121, 41, 118, 117, 47, 115, 126, 51, 52, 86, 87, 113, 83, 52, 34, 94, 56, 117, 55, 118, 79, 63, 121, 73, 108, 48, 96, 43, 88, 96, 114, 50, 65, 72, 48, 70, 108, 73, 94, 44, 49, 43, 42, 46, 39, 37, 54, 59, 122, 47, 80, 80, 77, 44, 69, 84, 39, 84, 48, 119, 80, 56, 58, 102, 102, 105, 36, 95, 78, 72, 116, 52, 55, 43, 36],
    [34, 123, 62, 35, 69, 120, 43, 125, 56, 51, 74, 105, 72, 97, 83, 77, 99, 66, 62, 77, 40, 34, 62, 109, 84, 99, 109, 44, 119, 84, 43, 114, 51, 124, 88, 63, 61, 45, 68, 59, 55, 100, 64, 104, 38, 89, 106, 67, 115, 92, 108, 83, 97, 50, 97, 58, 114, 77, 73, 74, 100, 71, 50, 58, 104, 120, 99, 96, 55, 34, 114, 106, 124, 101, 55, 45, 42, 58, 99, 59, 83, 101, 125, 111, 113, 69, 110, 92, 51, 46, 41, 44, 114, 109, 46, 88, 110, 110, 70, 100, 70, 35, 118, 95, 62, 105, 34, 106, 115, 93, 108, 77, 110, 62, 79, 108, 34, 69, 36, 123, 51, 82, 62, 49, 82, 121, 93, 64],
)

MIN_ASCII = 33
MAX_ASCII = 126

def ana_linear_regression(matrix, vector):
    start = time.time()
    ret = numpy.linalg.lstsq(matrix, vector, rcond=-1)[0]
    end = time.time()
    return ret, end - start

def sgd_linear_regression(matrix, vector):
    start = time.time()
    regressor = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.001,
        fit_intercept=False,
        max_iter=10000,
        tol=1e-3,
        epsilon=0.1,
        #learning_rate='constant',
        #average=10
    )
    scaler = StandardScaler()
    pipeline = make_pipeline(scaler, regressor)
    pipeline.fit(matrix, vector)
    end = time.time()
    return pipeline, end - start

def generate_matrix(m, n, i):
    if ADD_CONST:
        matrix = np.zeros((m, n + 1))
    else:
        matrix = np.zeros((m, n))

    min_string = GROUP_PIVOTS[i]
    max_string = GROUP_PIVOTS[i+1]

    for i in range(0, m):
        status = 0
        for j in range(0, n):
            if status == 0:
                gen = float(rd.randint(min_string[j], max_string[j] + 1))
                matrix[i, j] = gen
                if gen == min_string[j]:   status = 1
                elif gen == max_string[j]: status = 3
                else:                      status = 2
            elif status == 1:
                gen = float(rd.randint(min_string[j], MAX_ASCII + 1))
                matrix[i, j] = gen
                if gen == min_string[j]:   status = 1
                elif gen == max_string[j]: status = 3
                else:                      status = 2
            elif status == 2:
                gen = float(rd.randint(MIN_ASCII, MAX_ASCII + 1))
                matrix[i][j] = gen
            elif status == 3:
                gen = float(rd.randint(MIN_ASCII, max_string[j] + 1))
                matrix[i, j] = gen
                if gen == min_string[j]:   status = 1
                elif gen == max_string[j]: status = 3
                else:                      status = 2
    if ADD_CONST:
        for i in range(0, m):
            matrix[i, n] = 1
    return matrix

def sort_matrix(matrix):
    _, n = matrix.shape
    for i in range(n-1, -1, -1):
        matrix = matrix[matrix[:, i].argsort(kind='mergesort')]
    return matrix

def err_through_sampling(a_matrix, b_vector, model, sample_num):
    err_sum = 0
    err_count = 0
    max_err = 0
    for i in range(sample_num):
        key = a_matrix[i, :]
        index = b_vector[i]
        assert(len(model) == len(key))
        predicted = np.dot(model, key)
        
        if max_err < abs(index - predicted):
            max_err = abs(index - predicted) 
        err_sum += abs(index - predicted)
        err_count += 1
    return max_err, (err_sum / err_count)

def err_through_sampling_sgd(a_matrix, b_vector, regressor, sample_num):
    err_sum = 0
    err_count = 0
    max_err = 0
    for i in range(sample_num):
        key = a_matrix[i, :]
        index = b_vector[i]
        predicted = regressor.predict(np.array([key]))
        
        if max_err < abs(index - predicted):
            max_err = abs(index - predicted) 
        err_sum += abs(index - predicted)
        err_count += 1
    return max_err, (err_sum / err_count)

if __name__ == "__main__":
    m = 1000
    n = 32
    start_index = 0
    pivot_num = 0
    sample_num = m
    
    test_iter = 100
    ana_time_sum = 0.0
    sgd_time_sum = 0.0
    ana_max_sum = 0.0
    sgd_max_sum = 0.0
    ana_avg_sum = 0.0
    sgd_avg_sum = 0.0

    for i in range(test_iter):
        key_matrix = generate_matrix(m, n, pivot_num)
        key_matrix = sort_matrix(key_matrix)
        index = np.array([i for i in range(start_index, start_index + m)])
        ana_result, ana_time = ana_linear_regression(key_matrix, index)
        sgd_regressor, sgd_time = sgd_linear_regression(key_matrix, index)

        ana_max_err, ana_avg_err = err_through_sampling(key_matrix, index, ana_result, sample_num)
        sgd_max_err, sgd_avg_err = err_through_sampling_sgd(key_matrix, index, sgd_regressor, sample_num)

        ana_time_sum += ana_time
        sgd_time_sum += sgd_time
        ana_max_sum += ana_max_err
        ana_avg_sum += ana_avg_err
        sgd_max_sum += sgd_max_err
        sgd_avg_sum += sgd_avg_err
    
    print("Time to analytic lr: ", ana_time_sum / test_iter, "sec")
    print("Time to sgd lr: ", sgd_time_sum / test_iter, "sec")
    print("Analytic lr Average err: ", ana_avg_sum / test_iter)
    print("Analytic lr Max err: ", ana_max_sum / test_iter)
    print("SGD lr Average err: ", sgd_avg_sum / test_iter)
    print("SGD lr Max err: ", sgd_max_sum / test_iter)