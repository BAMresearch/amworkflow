import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def shortest_distance(line1, line2):
    pt11, pt12 = line1
    pt21, pt22 = line2
    s1 = pt12 - pt11
    s2 = pt22 - pt21
    s1square = np.dot(s1, s1)
    s2square = np.dot(s2, s2)
    lmbda1 = (np.dot(s1,s2) * np.dot(pt11 - pt21,s2) - s2square * np.dot(pt11 - pt21, s1)) / (s1square * s2square - (np.dot(s1, s2)**2))
    lmbda2 = -(np.dot(s1,s2) * np.dot(pt11 - pt21,s1) - s1square * np.dot(pt11 - pt21, s2)) / (s1square * s2square - (np.dot(s1, s2)**2))
    condition1 = lmbda1 >= 1
    condition2 = lmbda1 <= 0
    condition3 = lmbda2 >= 1
    condition4 = lmbda2 <= 0
    if condition1 or condition2 or condition3 or condition4:
        choices = [[line2, pt11,s2], [line2, pt12, s2], [line1, pt21, s1], [line1, pt22, s1]]
        result = np.zeros((4,2))
        for i in range(4):   
            result[i] = shortest_distance_p(choices[i][0], choices[i][1])
        shortest_index = np.argmin(result.T[1])
        shortest_result = result[shortest_index]
        pti1 = shortest_result[0] * choices[shortest_index][2] + choices[shortest_index][0][0]
        pti2 = choices[shortest_index][1]
        print(result)
    else:
        pti1 = pt11 + lmbda1 * s1
        pti2 = pt21 + lmbda2 * s2
    print(lmbda1, lmbda2)
    print(pti1, pti2)
    print(np.dot(s1,pti2 - pti1), np.dot(s2,pti2 - pti1))
    return np.array([pti1, pti2])
    # return np.array([lmbda1, lmbda2])

def shortest_distance_p(line, p):
    pt1, pt2 = line
    s = pt2 - pt1
    lmbda = (p - pt1).dot(s) / s.dot(s)
    pt_compute = pt1 + lmbda * s
    if lmbda < 1 and lmbda > 0:
        distance = np.linalg.norm(pt_compute - p)
        return lmbda, distance
    elif lmbda <= 0:
        distance = np.linalg.norm(pt1 - p)
        return 0, distance
    else:
        distance = np.linalg.norm(pt2 - p)
        return 1, distance
    
def random_pnt_gen(xmin, xmax, ymin, ymax, zmin = 0, zmax = 0):
    random_x = np.random.randint(xmin, xmax)
    random_y = np.random.randint(ymin, ymax)
    if zmin == 0 and zmax == 0:
        random_z = 0
    else:
        random_z = np.random.randint(zmin, zmax)
    return np.array([random_x, random_y, random_z])

def random_line_gen(xmin, xmax, ymin, ymax, zmin = 0, zmax = 0):
    pt1 = random_pnt_gen(xmin, xmax,ymin,ymax,zmin,zmax)
    pt2 = random_pnt_gen(xmin, xmax,ymin,ymax,zmin,zmax)
    return np.array([pt1, pt2])

lin1 = random_line_gen(0,10,0,10)
lin2 = random_line_gen(0,10,0,10)
lini = shortest_distance(lin1, lin2)


def plot(lin1,lin2, lini):
    # Define the coordinates of the two points for each of the three segments
    x_segment1 = lin1.T[0]
    y_segment1 = lin1.T[1]

    x_segment2 = lin2.T[0]
    y_segment2 = lin2.T[1]

    x_segment3 = lini.T[0]
    y_segment3 = lini.T[1]
    # Create the plot
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    plt.plot(x_segment1, y_segment1, label='Segment 1', linestyle='-', marker='o', color='blue')
    plt.plot(x_segment2, y_segment2, label='Segment 2', linestyle='--', marker='s', color='green')
    plt.plot(x_segment3, y_segment3, label='Segment 3', linestyle=':', marker='^', color='red')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')    
    
    # Add labels, a legend, and other plot elements
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Three Line Segments Plot')
    plt.legend()

    # Display the plot
    plt.grid(True)  # Optional: Add a grid
    plt.show()
    
plot(lin1=lin1, lin2=lin2, lini= lini)