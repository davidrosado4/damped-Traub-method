# Function definitions for the project
import numpy as np
import matplotlib.pyplot as plt
def damped_traub(f, df, z, tol = 1e-15, delta = 1):
    """
    Damped Newton's method with Traub's modification
    :param f: Function to find root of
    :param df: Derivative of f
    :param z: Initial guess
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :return: None if no convergence, otherwise root and number of iterations
    """
    # Maximum 50 iterations
    for i in range(100):
        if abs(df(z)) > tol:
            newt_step = z - f(z)/df(z)
            z_new = newt_step - delta * f(newt_step)/df(z)
        else:
            return None, 100 # Return None if derivative is too small

        # Stop the method when two iterates are close or f(z) = 0
        if abs(f(z)) < tol or abs(z_new - z) < tol:
            return z_new, i
        else:
            # Update z and continue
            z = z_new
    # If no convergence, return None
    return None, 100

def plot_damped_traub(f, df, tol = 1e-15, delta = 1, N = 2000, xmin = -1, xmax = 1, ymin = -1, ymax = 1):
    """
    Plots the convergence of damped Traub's method
    :param f: Function to find root of
    :param df: Derivative of f
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param N: Number of points to plot
    :param xmin: Minimum x value for plot
    :param xmax: Maximum x value for plot
    :param ymin: Minimum y value for plot
    :param ymax: Maximum y value for plot
    :return: None
    """
    # List to store unique roots
    roots = []

    # Define the ranges for z_x and z_y
    z_x_range = np.linspace(xmin, xmax, N)
    z_y_range = np.linspace(ymin, ymax, N)

    # Create a meshgrid from the ranges
    z_x, z_y = np.meshgrid(z_x_range, z_y_range)

    # Create an array to store the number of iterations
    iterations_array = np.zeros_like(z_x)

    # Iterate over the meshgrid
    for i in range(N):
        for j in range(N):

            # Create a complex number from the meshgrid
            point = complex(z_x[i,j], z_y[i,j])

            # Apply damped Traub's method
            root, iterations = damped_traub(f, df, point, tol, delta)

            # Store the number of iterations
            iterations_array[i,j] = iterations

            # Check if the root is found
            if root:
                flag = False
                # Check if the root is already in the list
                for test_root in roots:
                    if abs(test_root - root) < tol*1e7:
                        root = test_root
                        flag = True
                        break
                # If the root is not in the list, append it
                if not flag:
                    roots.append(root)

    # Define the maximum number of iterations for normalization
    max_iterations = np.max(iterations_array)
    min_iterations = np.min(iterations_array)

    # Plot the colored picture
    plt.figure(figsize = (10,10))
    plt.imshow(iterations_array, extent = [xmin, xmax, ymin, ymax], cmap = 'hsv', vmax = max_iterations, vmin = min_iterations)

    # Plot the roots
    root_markers = np.array(roots)
    plt.scatter(root_markers.real, root_markers.imag, marker = 'o', color = 'black', s = 20)

    # Remove the axes
    plt.axis('off')

    # Show the plot
    plt.show()





