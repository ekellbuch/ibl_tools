import numpy as np


def quantile_scaling(x, min_per=5, max_per=95):
    """
    Scale data using max and min quantiles
    """
    # see quantile_transform sklearn
    x_min = np.nanpercentile(x, min_per)
    x_max = np.nanpercentile(x, max_per)
    xrmp = (x - x_min) / (x_max - x_min)
    return xrmp, x_min, x_max




def get_area_inscribed_quadrilateral_from_side_lengths(side_LT,side_LB,side_TR,side_BR):
    """
    Calculate the area of a inscribed quadrilateral given 
    Given the length of four sides of a quadrilateral: a, b, c, and d
    p = (a + b + c + d)/2
    Area = sqrt((p-a)(p-b)(p-c)(p-d))
    
    Inputs:
    --------
    side_LT: T, array
        length of distance between left and top coordinates
    side_LB: T, array
        length of distance between left and bottom coordinates
    side_TR: T, array
        length of distance between right and top coordinates
    side_BR: T, array
        length of distance between right and bottom coordinates
        
    Outputs:
    --------
    Area: T, array
        area of inscribed quadrilateral
    """
    # Use coordinates to calculate sides
    p = (side_LT + side_LB + side_TR + side_BR )/2
    area = np.sqrt((p-side_LT)*(p-side_LB)*(p-side_TR)*(p-side_BR))
    return area


def distance_2markers(x, y):
    """
    Calculate distance between two points in 2D coordinate system
    point 1 (a_x, b_x)
    point 2 (a_y, b_y)
    
    Params:
    x 2, T array
    y 2, T array
    
    Return:
    d(x,y): T, array
    """
    assert len(x) == 2
    assert len(y) == 2
    a_x, b_x = x[0], x[1]
    a_y, b_y = y[0], y[1]
    return np.sqrt((b_x - a_x)**2 + (b_y - a_y)**2)


def get_quadrilateral_lengths(x, y, top = 0, bottom = 1, left = 2, right = 3):
    """
    Calculate the lenghts of 4 sides of a quadrilateral
    from its x y coordinates
    
    Inputs:
    -------
    x : 4, T array
    y : 4, T array
    top: int 
        index of top coordinate
    bottom: int
        index of bottom coordinate
    left: int
        index of left coordinate
    right: int 
        index of right coordinate
    
    Outputs:
    --------
    side_LT: T, array
        length of distance between left and top coordinates
    side_LB: T, array
        length of distance between left and bottom coordinates
    side_TR: T, array
        length of distance between right and top coordinates
    side_BR: T, array
        length of distance between right and bottom coordinates
    """
    side_LT = distance_2markers(x[[left, top]], y[[left, top]])
    side_LB = distance_2markers(x[[left, bottom]], y[[left, bottom]])
    side_TR = distance_2markers(x[[right, top]], y[[right, top]])
    side_BR = distance_2markers(x[[right, bottom]], y[[right, bottom]])
    return side_LT, side_LB, side_TR, side_BR

def get_area_inscribed_quadrilateral_from_coordinates(x, y, top = 0, bottom = 1, left = 2, right = 3):
    """
    Calculate the area of circumscribed quadrilateral from coordinates
    """
    side_LT, side_LB, side_TR, side_BR = get_quadrilateral_lengths(x, y,
                                                                         top=top,
                                                                         bottom=bottom,
                                                                         left=left,
                                                                         right=right)
    
    return get_area_inscribed_quadrilateral_from_side_lengths(side_LT,side_LB,side_TR,side_BR)


def from_index_to_time(idx_samples, fps):
    """
    Calculates time given index of frame in samples
    Inputs:
    ------
    idx_samples: int
        sample 
    fps: float
        sampling frequency
    """
    idx_seconds = (idx_samples /fps)
    num_hours = idx_seconds // (60*60)
    num_add_min = idx_seconds % (60*60)
    num_min  = num_add_min // 60
    num_add_sec = num_add_min % (60)
    print('{} hr {} min {} sec'.format(num_hours, num_min, num_add_sec))
    return 

