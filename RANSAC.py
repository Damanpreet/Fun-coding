'''
Code from https://salzis.wordpress.com/2014/06/10/robust-linear-model-estimation-using-ransac-python-implementation/
'''
# import packages
import numpy as np 
import scipy 
import matplotlib.pyplot as plt
import math
import sys
from sklearn import linear_model
# from sklearn.datasets import make_regression


def find_line_model_params(points):
    '''
    returns line model
    given two points y1 and y2, 
        m(slope) = (y2-y1)/(x2-x1)
        c = y2 - mx2
    '''
    m = (points[1, 1] - points[0, 1])/(points[1, 0] - points[0, 0])
    c = points[1, 1] - m*points[1, 0]
    return m, c


'''
https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
'''
def find_intercept_point_line(m, c, x0, y0):
    x = (x0 + m*y0 - m*c)/(m**2 + 1)
    y = m * x + c
    dist = math.sqrt((x-x0)**2 + (y-y0)**2)
    return x, y, dist


'''
Plot the graph on each iteration
'''
def ransac_plot(step, x, y, m, c, out_dir, final=False, x_inliers=(), y_inliers=(), points=()):
    title = "Iteration"+str(step)
    figname = out_dir+title+'.png'
    plt.figure("Ransac", figsize=(15., 15.))
    
    # grid for the plot
    grid = [min(x[:,0])-10, max(x[:,0])+10, min(y[:,0])-20, max(y[:,0])+20]
    plt.axis(grid)
    
    plt.grid(b = True, which = 'major', color = '0.75', linestyle = '--')
    plt.xticks([i for i in range(math.floor(min(x[:,0])-10), math.ceil(max(x[:,0])+10), 5)])
    plt.yticks([i for i in range(math.floor(min(y[:,0])-20), math.ceil(max(y[:,0])+20), 10)])

    # draw the model line and points selected to choose the model line
    plt.plot(x, m*x+c, 'r', label='line model', color='#0080ff', linewidth=5)
    plt.plot(x[:, 0], y[:, 0], marker = 'o', label = 'Input points', color='#00cc00', linestyle='', alpha=0.4)

    if not final:
        plt.plot(x_inliers, y_inliers[:], marker='o', label='inliers', linestyle='None', color='#ff0000', alpha=0.6) #inliers
        plt.plot(points[:,0], points[:,1], marker='o', label='Picked points', color='#0000cc', linestyle='None', alpha=0.6) #selected points to draw the line

    if not final:
        plt.title(title)
    if final:
        plt.title('Choosen Model')
    plt.legend()
    # plt.show()
    plt.savefig(figname)
    plt.close()
    

def main():
    ransac_iters = 100
    ransac_thresh = 5      # threshold of distance between inliers and the line fit
    ransac_ratio = 0.9    # ratio of inliers (min. value)

    # generate random points
    n_samples = 500
    outliers_ratio = 0.2 # we don't know this, but we randomly assign some value.

    # generate samples 
    x = 30*np.random.randn(n_samples, 1)

    line_fit = 0.5*np.random.normal(size = (1, 1))

    y = scipy.dot(x, line_fit)

    # Add a little gaussian noise
    x_noise = x + np.random.normal(size=x.shape)
    y_noise = y + np.random.normal(size=y.size)

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    no_outliers = math.floor(outliers_ratio*n_samples)
    outlier_indices = indices[:no_outliers]

    x_noise[outlier_indices] = 30 * np.random.randn(no_outliers, 1)
    y_noise[outlier_indices] = 30 * np.random.randn(no_outliers, 1)

    data = np.hstack((x_noise, y_noise))

    # best fit model parameters
    model_ratio = 0
    model_m = 0
    model_c = 0

    print("Running RANSAC for iterations ", ransac_iters)

    for i in range(ransac_iters):
        print("Iteration - ", i+1)
        #1. choose 2 random points.
        n = 2

        #2. randomly choose 2 points for fitting the line and remaining points for testing.
        all_indices = np.arange(x.shape[0])
        np.random.shuffle(all_indices)
        sel_points = data[all_indices[:n], :]
        test_points = data[all_indices[n:], :]

        #3. find a line model for these points
        #use the function find_line_model_params to find the parameters of the line
        m, c = find_line_model_params(sel_points)

        x_list = [] #list of inliers
        y_list = []
        num = 0

        for pt in range(test_points.shape[0]):
            x0 = test_points[pt, 0]
            y0 = test_points[pt, 1]

            _, _, dist = find_intercept_point_line(m, c, x0, y0)

            if dist < ransac_thresh:
                x_list.append(x0)
                y_list.append(y0)
                num += 1

        if num/float(n_samples) > model_ratio:
            model_ratio = num/float(n_samples)
            model_m = m
            model_c = c

            print("New model params: ")
            print("Ransac ratio: ", model_ratio)
            print("Slope: ", m)
            print("Intercept: ", c)

        # Plot the graph in each iteration.
        ransac_plot(i+1, x_noise, y_noise, m, c, 'ransac_out/', False, np.array(x_list), np.array(y_list), sel_points)

        # Stop iterating as soon as the model is above the threshold.
        if num > ransac_ratio*n_samples:
            print("Model found!")
            break

    # Plot the final result.
    ransac_plot(0, x_noise, y_noise, model_m, model_c, 'ransac_out/', True)

    print("Final params: ")
    print("Ransac ratio: ", model_ratio)
    print("Slope: ", m)
    print("Intercept: ", c)

    ####################################################################################
    # Using sklearn
    ####################################################################################
    # X, y = make_regression(
    #      n_samples=500, n_features=2, noise=4.0, random_state=0)

    line_ransac = linear_model.RANSACRegressor(stop_probability=0.99, stop_score=0.99, max_trials=100)
    X = data[:,0]
    y = data[:,1]
    line_ransac.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    line_ransac.predict(X.reshape(-1, 1))

    inlier_mask = line_ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    print("Score: ", line_ransac.score(X.reshape(-1, 1), y.reshape(-1, 1))) # Prediction score
    print("Coefficients - m: ", line_ransac.estimator_.coef_, ", c: ", line_ransac.estimator_.intercept_)
    print("No of trials to obtain the result: ", line_ransac.n_trials_)

    lineX = np.arange(-5, 5)
    lineY = line_ransac.predict(lineX[:, np.newaxis])

    plt.figure("Ransac", figsize=(15., 15.))
    plt.plot(X[inlier_mask], y[inlier_mask], marker='o', linestyle='None', label='Inliers', color='#ff0000', alpha=0.6)
    plt.plot(X[outlier_mask], y[outlier_mask], marker='o', linestyle='', label='Outliers', color='#0000cc', alpha=0.6)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
