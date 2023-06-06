import numpy as np

# st_pt = np.array([10, 8, 6, 4])
# nd_pt = np.array([20, 16, 12, 8])
# num = [20, 24, 30, 40]
def simple_permutator(start_point: np.ndarray,
                      end_point: np.ndarray,
                      num: np.ndarray) -> np.ndarray:
    start_point = start_point.astype(np.float64)
    end_point = end_point.astype(np.float64)
    dm2 = np.max(num)
    dm1 = np.shape(start_point)[0]
    permutation = np.zeros((dm1, dm2, dm1))
    for start_ind, start_candidate in enumerate(start_point):
        lnsps = np.linspace(start_candidate,
                            end_point[start_ind],
                            num[start_ind])
        tp_perm = np.copy(start_point)
        for iter_num in range(num[start_ind]):
            tp_perm[start_ind] = lnsps[iter_num] 
            permutation[start_ind][iter_num] = tp_perm
    is_zero_vector = np.all(permutation == np.zeros(dm1), axis=2)
    result = permutation[~is_zero_vector]
    return permutation, result

# print(simple_permutator(start_point=st_pt,
# end_point=nd_pt,
# num= num)[1])