"""
===============================================================================
Title: TLS Flow Sample Visualization Utilities
Description:
    This module provides helper functions for transforming one-dimensional 
    TLS flow samples (e.g., sequences of record lengths) into two-dimensional 
    image-like matrices for visualization and feature learning.

    The process involves:
      - Finding the nearest composite number greater than or equal to 
        the sample length.
      - Determining an optimal (d1, d2) divisor pair such that d1 * d2 
        ≈ sample length and the resulting 2D shape is as close to square 
        as possible.
      - Padding the sample with zeros to fit this shape.
      - Reshaping the sample into a 2D array suitable for visualization 
        or input to convolutional models.

Functions:
    - is_composite(x): Check if a number is composite.
    - find_nearest_composites(n): Find nearby composite numbers greater than n.
    - greatest_divisor_pair(x): Find the closest divisor pair of a composite number.
    - get_padding_and_dim(x): Compute target padding and 2D reshape dimensions.
    - make_image_from_sample(sample): Pad and reshape a 1D sample into a 2D image.
===============================================================================
"""

import math
import numpy as np
def is_composite(x):
    """Return True if x is composite (not prime) and x >= 4; otherwise False."""
    if x < 4:
        return False  # 2 and 3 are prime; 1 is neither prime nor composite
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return True
    return False

def find_nearest_composites(n):
    """Return the composite numbers greater than n."""
    candidates = []
    for i in range(n, int(n * 3 / 2)):
        if is_composite(i):
            candidates.append(i)
    return candidates

def greatest_divisor_pair(x):
    """
    Return the pair of divisors (d, x//d) for composite x such that
    d is the greatest divisor not exceeding sqrt(x). This pair is closest to each other.
    """
    d = int(math.sqrt(x))
    while d > 1:
        if x % d == 0:
            return (d, x // d)
        d -= 1
    return (1, x)

def get_padding_and_dim(x):
    """
    Determine the nearest composite number ≥ x and compute its 
    optimal 2D reshape dimensions (d1, d2) with minimal difference 
    between sides. Returns (new_length, dim_x, dim_y), where:
        new_length - padded length
        dim_x, dim_y - target image dimensions
    """
    dif = x
    val_x = x
    val_d1 = 0
    val_d2 = 0
    for nearest in find_nearest_composites(x):
        d1, d2 = greatest_divisor_pair(nearest)  
        if (math.fabs(d1-d2) > dif):
            return (val_x, val_d1, val_d2)
        else:
            val_x = nearest
            val_d1 = d1
            val_d2 = d2
            dif = math.fabs(d1-d2) 



def make_image_from_sample(sample):
    """
    Pad and reshape a 1D sample into a 2D image representation.

    Parameters:
        sample (array-like): The input 1D sequence (e.g., TLS record lengths).

    Returns:
        np.ndarray: A 2D array representation of the sample suitable 
                    for visualization or convolutional model input.
    """
    sample_len = len(sample)
    (newrow_len, IMAGE_DIM_X, IMAGE_DIM_Y) = get_padding_and_dim(sample_len)
    IMAGE_PAD = newrow_len - sample_len
    return np.pad(
        sample,
        pad_width=(0, IMAGE_PAD),
        mode='constant',
        constant_values=0
    ).reshape(IMAGE_DIM_X, IMAGE_DIM_Y)