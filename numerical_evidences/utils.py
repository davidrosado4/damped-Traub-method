# Function definitions for the project
import numpy as np
import cmath
import matplotlib.pyplot as plt
def damped_traub(f, df, z, tol = 1e-10, delta = 1, max_iter = 100):
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

def find_free_critical(a):
    """
    Find the critical points of the Traub's method
    :param a: Complex parameter
    :return: The three critical points
    """
    # First critical value
    term1 = (-92*a**3 + 138*a**2 + 3 * cmath.sqrt(69) * cmath.sqrt(9*a**6 - 27*a**5 - 38*a**4 + 121*a**3 - 38*a**2 - 27*a + 9) + 138*a - 92)**(1/3) / (3 * 23**(2/3))
    term2 = (-115*a**2 + 115*a - 115) / (69 * 23**(1/3) * (-92*a**3 + 138*a**2 + 3 * cmath.sqrt(69) * cmath.sqrt(9*a**6 - 27*a**5 - 38*a**4 + 121*a**3 - 38*a**2 - 27*a + 9) + 138*a - 92)**(1/3))
    term3 = (a + 1) / 3
    z1 = term1 - term2 + term3
    # Second critical value
    term1 = -((1 - 1j * cmath.sqrt(3)) * (-92*a**3 + 138*a**2 + 3 * cmath.sqrt(69) * cmath.sqrt(9*a**6 - 27*a**5 - 38*a**4 + 121*a**3 - 38*a**2 - 27*a + 9) + 138*a - 92)**(1/3)) / (6 * 23**(2/3))
    term2 = ((1 + 1j * cmath.sqrt(3)) * (-115*a**2 + 115*a - 115)) / (138 * 23**(1/3) * (-92*a**3 + 138*a**2 + 3 * cmath.sqrt(69) * cmath.sqrt(9*a**6 - 27*a**5 - 38*a**4 + 121*a**3 - 38*a**2 - 27*a + 9) + 138*a - 92)**(1/3))
    term3 = (a + 1) / 3
    z2 = term1 + term2 + term3
    # Third critical value
    term1 = -((1 + 1j * cmath.sqrt(3)) * (-92*a**3 + 138*a**2 + 3 * cmath.sqrt(69) * cmath.sqrt(9*a**6 - 27*a**5 - 38*a**4 + 121*a**3 - 38*a**2 - 27*a + 9) + 138*a - 92)**(1/3)) / (6 * 23**(2/3))
    term2 = ((1 - 1j * cmath.sqrt(3)) * (-115*a**2 + 115*a - 115)) / (138 * 23**(1/3) * (-92*a**3 + 138*a**2 + 3 * cmath.sqrt(69) * cmath.sqrt(9*a**6 - 27*a**5 - 38*a**4 + 121*a**3 - 38*a**2 - 27*a + 9) + 138*a - 92)**(1/3))
    term3 = (a + 1) / 3
    z3 = term1 + term2 + term3
    return z1, z2, z3


def evidences_critical_points_control(tol=1e-10, max_iter=1000, N=1000, delta=1):
    """
    Function to observe if critical points for Traub's method are under control
    :param tol: Tolerance for convergence
    :param max_iter: Maximum number of iterations
    :param N: Number of points in the meshgrid
    :param delta: Damping parameter
    :return: The number of critical points that are not under control and the data of the critical points
    """

    # Define the ranges for a_x and a_y
    a_x_range = np.linspace(-1, 1, N)
    a_y_range = np.linspace(-1, 1, N)

    # Create a meshgrid from the ranges
    a_x, a_y = np.meshgrid(a_x_range, a_y_range)

    # Create a complex number from the meshgrid
    count_no_conv = 0
    data = []
    for i in range(N):
        for j in range(N):
            # Define the polynomial for that value of a
            a = complex(a_x[i, j], a_y[i, j])

            def f(z):
                return z * (z - 1) * (z - a)

            def der_f(z):
                return 3 * z ** 2 - 2 * (1 + a) * z + a

            # Set the free critical points for Traub method
            c1, c2, c3 = find_free_critical(a)
            # Iterate the free critical points
            for c in [c1, c2, c3]:
                root, iterations = damped_traub(f, der_f, c, tol=tol, delta=delta, max_iter=max_iter)
                if abs(root-0)>tol and abs(root-1)>tol and abs(root-a)>tol:
                    count_no_conv += 1
                    print({'a': a, 'crit': c, 'root': root, 'iterations': iterations})
                else:
                    # Save a dict with the value of the critical point, the rooot and the number of iterations
                    data_dict = {'a': a, 'crit': c, 'root': root, 'iterations': iterations}
                    # Append the dict to the list
                    data.append(data_dict)
    return count_no_conv, data

def parameter_plane_a(tol=1e-10, max_iter=100, N=1000, delta=1, xmin=0, xmax=1, ymin=0, ymax=1):
    """
    Function that plots the parameter plane for the Traub's method with respect the cubic polynomial
    :param tol: Tolerance for convergence
    :param max_iter: Maximum number of iterations
    :param N: Number of points in the meshgrid
    :param delta: Damping parameter
    :param xmin: Minimum value for a_x
    :param xmax: Maximum value for a_x
    :param ymin: Minimum value for a_y
    :param ymax: Maximum value for a_y
    :return: The plot of the parameter plane
    """
    # List to store unique roots
    roots = []

    # Define the ranges for a_x and a_y
    a_x_range = np.linspace(xmin, xmax, N)
    a_y_range = np.linspace(ymin, ymax, N)

    # Create a meshgrid from the ranges
    a_x, a_y = np.meshgrid(a_x_range, a_y_range)

    # Create an array to store the number of iterations
    iterations_array = np.zeros_like(a_x)

    # Create a complex number from the meshgrid
    for i in range(N):
        for j in range(N):
            # Define the polynomial for that value of a
            a = complex(a_x[i, j], a_y[i, j])

            def f(z):
                return z * (z - 1) * (z - a)

            def der_f(z):
                return 3 * z ** 2 - 2 * (1 + a) * z + a
            # Iterate the critical point (a+1)/3
            root, iterations = damped_traub(f, der_f, (a+1)/3, tol=tol, delta=delta, max_iter=max_iter)

            iterations_array[i, j] = iterations

    # Define the maximum number of iterations for normalization
    max_iterations = np.max(iterations_array)
    min_iterations = np.min(iterations_array)

    # Plot the colored picture
    plt.figure(figsize=(10, 10))
    plt.imshow(iterations_array, extent=[xmin, xmax, ymin, ymax], cmap='hsv', vmax=max_iterations,
               vmin=min_iterations, origin='lower')

    # Remove the axes
    plt.axis('off')

    # Show the plot
    plt.show()


def damped_traub(f, df, z, tol=1e-15, delta=1, max_iter=100):
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
            newt_step = z - f(z) / df(z)
            z_new = newt_step - delta * f(newt_step) / df(z)
        else:
            return None, max_iter  # Return None if derivative is too small

        # Stop the method when two iterates are close or f(z) = 0
        if abs(f(z)) < tol or abs(z_new - z) < tol:
            return z_new, i
        else:
            # Update z and continue
            z = z_new
    # If no convergence, return None
    return None, max_iter

def plot_damped_traub(f, df, tol=1e-15, delta=1, N=2000, xmin=-1, xmax=1, ymin=-1, ymax=1, max_iter=100):
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
            point = complex(z_x[i, j], z_y[i, j])

            # Apply damped Traub's method
            root, iterations = damped_traub(f, df, point, tol, delta, max_iter)

            # Store the number of iterations
            iterations_array[i, j] = iterations

            # Check if the root is found
            if root:
                flag = False
                # Check if the root is already in the list
                for test_root in roots:
                    if abs(test_root - root) < tol * 1e7:
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
    plt.figure(figsize=(10, 10))
    plt.imshow(iterations_array, extent=[xmin, xmax, ymin, ymax], cmap='hsv', vmax=max_iterations,
               vmin=min_iterations, origin='lower')

    # Plot the roots
    root_markers = np.array(roots)
    plt.scatter(root_markers.real, root_markers.imag, marker='o', color='black', s=20)

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

    return (b - a) * perc + a

def pixel_color(x, y, bounds_x, bounds_y, width, height, f, df, roots, colors, tol, delta, max_iter):
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
    real = normalize(bounds_x, x / width)
    imag = normalize(bounds_y, y / height)
    return point_color(complex(real, imag), f, df, roots, colors, delta=delta, tol=tol, max_iter=max_iter)

def point_color(z, f, df, roots, colors, tol=1e-15, delta=1, max_iter=100):
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
            return [0, 0, 0]
        else:
            newt_step = z - f(z) / df(z)
            z_new = newt_step - delta * f(newt_step) / df(z)
            for root_id, root in enumerate(roots):
                diff = abs(z_new - root)
                if diff > tol:
                    z = z_new
                    continue
                # Found which attractor the point converges to
                color_intensity = max(min(i / (1 << 5), 0.95), 0)
                return darken(colors[root_id], color_intensity)
    return [0, 0, 0]

def plot_colored_damped_traub_2left(f, df, bounds_x, bounds_y, width, height, roots, colors, tol=1e-15, delta=1,
                          max_iter=100, print_free_fixed=False):
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
            data[y, x] = pixel_color(x, y, bounds_x, bounds_y, width, height, f, df, roots, colors, delta=delta,
                                     tol=tol, max_iter=max_iter)

    # Plot the colored picture
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(data, extent=[bounds_x[0], bounds_x[1], bounds_y[0], bounds_y[1]], origin='lower')
    # Plot the fixed and free critical points
    if print_free_fixed:
        if delta == 0.01:
            crit = [0.362376,0.45311 - 0.0380033j, 0.45311 + 0.0380033j]
            fixed = [0.441468 - 0.0243827j,0.441468 + 0.0243827j,0.385174]
            ax.scatter([c.real for c in crit], [c.imag for c in crit], marker='x', color='black', s=40)
            ax.scatter([c.real for c in fixed], [c.imag for c in fixed], marker='x', color='white', s=40)
        if delta == 0.1:
            crit = [0.273742,0.500527 - 0.0616454j, 0.500527 + 0.0616454j]
            fixed = [0.468745 - 0.0413892j,0.468745 + 0.0413892j,0.33205]
            ax.scatter([c.real for c in crit], [c.imag  for c in crit], marker='x', color='black', s=40)
            ax.scatter([c.real for c in fixed], [c.imag  for c in fixed], marker='x', color='white', s=40)
        if delta == 0.5:
            crit = [0.122415,0.552786, 0.639662]
            fixed = [0.511082 - 0.039393j,0.511082 + 0.039393j,0.253488]
            ax.scatter([c.real for c in crit], [c.imag  for c in crit], marker='x', color='black', s=40)
            ax.scatter([c.real for c in fixed], [c.imag  for c in fixed], marker='x', color='white', s=40)
        if delta == 0.99:
            crit = [0.0225223,0.535078, 0.856152]
            fixed = [0.533897 - 0.0133846j,0.533897 + 0.0133846j,0.213562]
            ax.scatter([c.real for c in crit], [c.imag  for c in crit], marker='x', color='black', s=40)
            ax.scatter([c.real for c in fixed], [c.imag  for c in fixed], marker='x', color='white', s=40)

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

