import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def shortest_distance(line1, line2):
    pt11, pt12 = line1
    pt21, pt22 = line2
    s1 = pt12 - pt11
    s2 = pt22 - pt21
    s1square = np.dot(s1, s1)
    s2square = np.dot(s2, s2)
    lmbda1 = (np.dot(s1,s2) * np.dot(pt11 - pt21,s2) - s2square * np.dot(pt11 - pt21, s1)) / (s1square * s2square - (np.dot(s1, s2)**2))
    lmbda2 = -(np.dot(s1,s2) * np.dot(pt11 - pt21,s1) - s1square * np.dot(pt11 - pt21, s2)) / (s1square * s2square - (np.dot(s1, s2)**2))
    pti1 = pt11 + lmbda1 * s1
    pti2 = pt21 + lmbda2 * s2
    print(lmbda1, lmbda2)
    print(pti1, pti2)
    # return np.array([pti1, pti2])
    return np.array([lmbda1, lmbda2])

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
# lini = shortest_distance(lin1, lin2)

class Problem():
    def __init__(self, x, line1, line2, mu):
        self.x = np.array(x)
        self.mu = mu
        pt11, pt12 = line1
        pt21, pt22 = line2
        x1 = self.x[0]
        x2 = self.x[1]
        s1 = pt12 - pt11
        s2 = pt22 - pt21
        s = np.array([s1, -s2]).T
        s1square = np.dot(s1, s1)
        s2square = np.dot(s2, s2)
        self.A = np.array(pt11 - pt21)
        self.f = (np.transpose(self.A.reshape((3,-1)) + s@self.x.reshape((2,-1)))@(self.A.reshape((3,-1)) + s@self.x.reshape((2,-1))))[0][0]
        self.df = (2*(self.A.reshape((3,-1)) + s@self.x.reshape((2,-1))).T@s)[0]
        self.Hf = 2*np.array([[s1.dot(s1),-s2.dot(s1)],[-s1.dot(s2), s2.dot(s2)]])
        # self.Hf = 2*np.array([[s1.dot(s1),-s1.dot(s2)],[-s2.dot(s1), s2.dot(s2)]])
        self.g = np.array([np.log(1 - x1), np.log(x1), np.log(1 - x2), np.log(x2)])
        self.g = np.sum(np.where(self.g, self.g is np.nan, 1e6))
        # self.g = np.log(1 - x1) + np.log(x1) + np.log(1 - x2) + np.log(x2)
        # if self.g is np.nan:
        #     self.g = 1e6
        self.dg = (-1 / (1 - x1)) + (1 / x1) + (-1 / (1 - x2)) + (1 / x2)
        self.d2g = (-1 / (1 - x1)**4) + (1 / x1**4) + (-1 / (1 - x2)**4) + (1 / x2**4)
        self.B = self.f - self.mu * self.g
        self.dB = self.df -self.mu * self.dg
        self.H = self.Hf + self.mu * self.d2g

# B = Problem(lin1, lin2)
# print(B.H)

def newton_solver(mu, problem: callable):
    x = [0.1, 0.1]
    B = problem(x, lin1, lin2, mu)
    qa_plus = 1.2
    qa_minus = 0.9
    qls = 0.01
    lmbda =1e-2
    alpha = 1
    while True:
        delta = -np.linalg.inv(B.H + lmbda * np.identity(2)) @ B.dB
        print(delta)
        B_linsearch = problem(x + alpha*delta, lin1, lin2,mu)
        line_search = B_linsearch.B > B.B + qls * (B.dB@(alpha * delta))
        print(line_search)
        # while line_search:
        #     alpha *= qa_minus
        #     B_linsearch = problem(x + alpha*delta, lin1, lin2,mu)
        x += alpha*delta
        B = problem(x, lin1, lin2,mu)
        alpha = min(qa_plus*alpha, 1)
        if np.linalg.norm(alpha*delta, np.inf) < 1e-6:
            break
    return x

# result = newton_solver(Problem)
# print(result)
# B = Problem([1.22,1.2], lin1, lin2)
# print(B.H)

def log_barrier(problem: callable):
    q_mu_minus = 0.5
    mu = 1
    i = 0
    while True:
        if i == 0:
            result = newton_solver(mu, problem)
            mu *= q_mu_minus
            i += 1
            last_result = np.copy(result)
        else:
            result = newton_solver(mu, problem)
            mu *= q_mu_minus
            i += 1
            if np.linalg.norm(result - last_result) < 1e-6:
                break
            else:
                last_result = np.copy(result)
    return result,mu

            
    
    

def plot(lin1,lin2, lini = []):
    # Define the coordinates of the two points for each of the three segments
    x_segment1 = lin1.T[0]
    y_segment1 = lin1.T[1]

    x_segment2 = lin2.T[0]
    y_segment2 = lin2.T[1]

    # x_segment3 = lini.T[0]
    # y_segment3 = lini.T[1]

    # Create the plot
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    plt.plot(x_segment1, y_segment1, label='Segment 1', linestyle='-', marker='o', color='blue')
    plt.plot(x_segment2, y_segment2, label='Segment 2', linestyle='--', marker='s', color='green')
    # plt.plot(x_segment3, y_segment3, label='Segment 3', linestyle=':', marker='^', color='red')

    # Add labels, a legend, and other plot elements
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Three Line Segments Plot')
    plt.legend()

    # Display the plot
    plt.grid(True)  # Optional: Add a grid
    plt.show()
    
def plot_3d(x1min,x1max, x2min,x2max, problem: callable,optimum_point,mu):
    # Define the range and density of your grid
    x = np.linspace(x1min, x1max, 100)
    y = np.linspace(x2min, x2max, 100)
    # X, Y = np.meshgrid(x, y)
    X = []
    Y = []
    Z = []
    for x0 in x:
        for y0 in y:     
            B = problem([x0,y0],lin1, lin2,1)
            z = B.B
            Z.append(z)
            X.append(x0)
            Y.append(y0)
    Z = np.array(Z)
    X = np.array(X)
    Y = np.array(Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot
    surf = ax.plot_trisurf(X, Y, Z, cmap='viridis')
    B_optimum = Problem(optimum_point, lin1, lin2,mu)
    ax.scatter([optimum_point[0]], [optimum_point[1]], [B_optimum.B], c='red', s=100, label='Specific Point')
    # Add a color bar which maps values to colors
    fig.colorbar(surf)

    # Set labels for the axes
    ax.set_xlabel('X1-axis')
    ax.set_ylabel('X2-axis')
    ax.set_zlabel('Z-axis')

    # Set a title for the plot
    ax.set_title('3D Plot of Your Function')

    # Show the plot
    plt.show()

# optimum_point = shortest_distance(lin1, lin2)   
# print(log_barrier(Problem))
optimum_point,mu =  log_barrier(Problem)
plot(lin1, lin2)
plot_3d(0.001,0.999,0.001,0.999,Problem,optimum_point,mu)
