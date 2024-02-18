# Function definitions for the project
import numpy as np
import matplotlib.pyplot as plt
def damped_traub(f, df, z, tol = 1e-15, delta = 1, max_iter = 100):
    """
    Damped Newton's method with Traub's modification
    :param f: Function to find root of
    :param df: Derivative of f
    :param z: Initial guess
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :return: None if no convergence, otherwise root and number of iterations
    """
    # Maximum 50 iterations
    for i in range(max_iter):
        if abs(df(z)) > tol:
            newt_step = z - f(z)/df(z)
            z_new = newt_step - delta * f(newt_step)/df(z)
        else:
            return None, max_iter # Return None if derivative is too small

        # Stop the method when two iterates are close or f(z) = 0
        if abs(f(z)) < tol or abs(z_new - z) < tol:
            return z_new, i
        else:
            # Update z and continue
            z = z_new
    # If no convergence, return None
    return None, max_iter

def plot_damped_traub(f, df, tol = 1e-15, delta = 1, N = 2000, xmin = -1, xmax = 1, ymin = -1, ymax = 1, max_iter = 100):
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
    :param max_iter: Maximum number of iterations
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
            root, iterations = damped_traub(f, df, point, tol, delta, max_iter)

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
    plt.figure(figsize=(10,10))
    plt.imshow(iterations_array, extent = [xmin, xmax, ymin, ymax], cmap = 'hsv', vmax = max_iterations, vmin = min_iterations, origin='lower')

    # Plot the roots
    root_markers = np.array(roots)
    plt.scatter(root_markers.real, root_markers.imag, marker = 'o', color = 'black', s = 20)

    # Remove the axes
    plt.axis('off')

    # Show the plot
    plt.show()
# --------------------------------------------------------------------------------------------
# Function for Damped Traub's method colored plot
def darken(color, fraction):
    """
    Darkens a color by a fraction
    :param color: Color to darken
    :param fraction: Fraction to darken by
    :return: Darkened color
    """
    return [p * (1 - fraction) for p in color]

def normalize(bounds, perc):
    """
    Normalizes the bounds by a percentage
    :param bounds: Bounds to normalize
    :param perc: Percentage to normalize by
    :return: Normalized bounds
    """
    a = bounds[0]
    b = bounds[1]

    return (b-a) * perc + a

def pixel_color(x,y, bounds_x, bounds_y, width, height, f, df, roots, colors,tol, delta, max_iter):
    """
    Returns the color of a pixel
    :param x: x-coordinate of the pixel
    :param y: y-coordinate of the pixel
    :param bounds_x: Bounds for the x-axis
    :param bounds_y: Bounds for the y-axis
    :param width: Width of the plot
    :param height: Height of the plot
    :param f: Function to find root of
    :param df: Derivative of f
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :return: Color of the pixel
    """
    real = normalize(bounds_x, x/ width)
    imag = normalize(bounds_y, y/ height)
    return point_color(complex(real, imag), f, df, roots, colors, delta=delta, tol=tol, max_iter=max_iter)

def point_color(z, f, df, roots, colors, tol = 1e-15, delta = 1, max_iter = 100):
    """
    Returns the color of a point
    :param z: Point to find the color of
    :param f: Function to find root of
    :param df: Derivative of f
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :return: Color of the point
    """
    # Perform the damped Traub's method assigning color intensity to the point
    for i in range(max_iter):
        if abs(df(z)) < tol:
            return [0,0,0]
        else:
            newt_step = z - f(z)/df(z)
            z_new = newt_step - delta * f(newt_step)/df(z)
            for root_id, root in enumerate(roots):
                diff = abs(z_new - root)
                if diff > tol:
                    z = z_new
                    continue
                # Found which attractor the point converges to
                color_intensity = max(min(i / (1 << 5), 0.95), 0)
                return darken(colors[root_id], color_intensity)
    return [0,0,0]

def plot_colored_damped_traub(f, df, bounds_x, bounds_y, width, height, roots, colors, tol = 1e-15, delta = 1, max_iter = 100):
    """
    Plots the convergence of damped Traub's method
    :param f: Function to find root of
    :param df: Derivative of f
    :param bounds_x: Bounds for the x-axis
    :param bounds_y: Bounds for the y-axis
    :param width: Width of the plot
    :param height: Height of the plot
    :param tol: Tolerance for convergence
    :param delta: Damping parameter
    :param max_iter: Maximum number of iterations
    :param roots: Roots of the function
    :param colors: Colors of the roots
    :return: None
    """
    data = np.zeros((height, width, 3), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            # Assign the color to the pixel
            data[y, x] = pixel_color(x, y, bounds_x, bounds_y, width, height, f, df, roots, colors, delta= delta, tol=tol, max_iter=max_iter)

    # Plot the colored picture
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(data, extent = [bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]], origin='lower')
    # Plot the roots
    ax.scatter([root.real for root in roots], [root.imag for root in roots], marker = 'o', color = 'black', s = 20)
    # Plotting initial conditions Hubbard et al.
    '''
    r = 2.283
    N = 67
    circle = np.zeros(N, dtype=complex)
    for i in range(N):
        theta = 2*np.pi*i/N
        circle[i] = r*np.exp(1j*theta)
    ax.scatter([c.real for c in circle], [c.imag for c in circle], marker = 'o', color = 'white', s = 20)
    '''
    plt.axis('off')
    plt.show()



