import numpy as np
def convert_central_line_to_wall(pts: np.ndarray):
    # create two sides.
    pts_lft = []
    pts_rgt = []
    loop_whole = np.array(pts_lft+pts_rgt)
    vectors = np.hstack((loop_whole, np.zeros((loop_whole.shape[0], 1))))
    
    for i,pt in loop_whole:
        