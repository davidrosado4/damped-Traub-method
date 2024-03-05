# On the Basins of Attraction of Root-Finding Algorithms

This repository contains the Python implementation for visualizing the basins of attraction in the complex plane for the Damped Traub's Method family. The Damped Traub's Method is a numerical algorithm utilized for finding roots of complex polynomials. The basins of attraction offer insights into how the method converges to different roots for various initial guesses.

## Introduction

The Damped Traub's Method is a widely-used numerical algorithm for finding roots of complex polynomials. Understanding the behavior of this method in the complex plane is crucial for analyzing its convergence properties. This repository provides tools to visualize the basins of attraction, aiding in the comprehension of the method's behavior. The Damped Traub's Method family encompasses a range of algorithms utilized for root finding of complex polynomials. Within this family, $\delta=0$ corresponds to the well-known Newton Method, while $\delta=1$ corresponds to the Traub Method.

## Example Pictures
![newt_5](https://github.com/davidrosado4/damped-Traub-method/assets/114001733/69dced5b-1f06-4dcf-b3ba-862c7be6cadd)
![traub_per_orb](https://github.com/davidrosado4/damped-Traub-method/assets/114001733/1993104c-9617-418b-9737-08c4d59ba268)
![parameter-plane](https://github.com/davidrosado4/damped-Traub-method/assets/114001733/b32efca2-8811-4ef5-a0b2-eac1c50933de)

## Features
- Implementation in Python
- Visualization of basins of attraction in the complex plane
- Two options for plotting basins of attraction: with different colors or with a unique palette of colors

## File Structure
- **utils.py**: Contains the definitions of the necessary functions.
- **colored-newton.ipynb**: Jupyter notebook for plotting basins of attraction with different colors.
- **damped-traub.ipynb**: Jupyter notebook for plotting basins of attraction with a unique palette of colors.
- **parameter-plane.ipynb**: Jupyter notebook for visualizing the parameter plane of a Blaschke product.
- **numerical_evidences/**: Folder containing Python implementations to demonstrate numerical evidences, particularly focusing on proving that Damped Traub's Method basins of attractions are unbounded and simple connected sets.
  
## Usage
To utilize this repository, simply clone it to your local machine and execute the provided Jupyter notebooks. Make sure to have the necessary dependencies installed.

## License
This project is licensed under the MIT License - see the [MIT License](LICENSE) file for details.

## Acknowledgments
Special thanks to [Dr. Xavier Jarque i Ribera](https://mat.ub.edu/departament/professors/jarque-i-ribera-xavier/) for their contributions and inspirations.

## Contact
For any inquiries or suggestions, please feel free to reach out to [rosadodav4@gmail.com](mailto:rosadodav4@gmail.com).
