# Function definitions for the project
import numpy as np
import cmath

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
    a_x_range = np.linspace(-4, 4, N)
    a_y_range = np.linspace(-4, 4, N)

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
                if root is None:
                    count_no_conv += 1
                    print({'a': a, 'crit': c, 'root': root, 'iterations': iterations})
                else:
                    # Save a dict with the value of the critical point, the rooot and the number of iterations
                    data_dict = {'a': a, 'crit': c, 'root': root, 'iterations': iterations}
                    # Append the dict to the list
                    data.append(data_dict)
    return count_no_conv, data