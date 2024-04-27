import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import bisect

DISPLAY = False


def display_image(image, title):
    global DISPLAY
    if not DISPLAY:
        return
    cv2.imshow("title", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def draw_lines_by_polar(image, rho_theta, color=(0, 0, 255), thickness=1):
    if rho_theta is not None:
        for rho, theta in rho_theta:
            # Calculate line endpoints
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = (a * rho).item()
            y0 = (b * rho).item()
            # Extend the line to the image border
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # Draw the line on the image
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def display_canny(image, low_threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    display_image(edges, "Canny edges")
    return edges


def hough_lines_for_theta(image, edges, theta: int, headline: str, hough_threshold: int):
    # Convert the angle range to radians
    theta_rad = np.deg2rad(theta)

    # Define the angle range (in radians)
    min_angle = np.deg2rad(theta - 20)  # Convert to radians
    max_angle = np.deg2rad(theta + 20)  # Convert to radians

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, rho=2, theta=np.deg2rad(1), threshold=hough_threshold, min_theta=min_angle,
                           max_theta=max_angle)
    # lines = cv2.HoughLines(edges, 2,  np.pi / 180,  100)
    # Draw the detected lines on the original image
    draw_lines_by_polar(image, lines[:, 0], thickness=2)
    # Display the image with detected lines
    display_image(image, headline)
    return lines


def check_threshold_if_load_lines(lines, theta):
    """
        This function check whether the detected lines are loaded in small amount of bins.

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta (float): The angle we want the lines to be on (approximately).

        Returns:
        True if there is a bin with more than 20 lines on it, means too many lines in one bin
        and False if there isn't.

    """
    bins_size = []
    for i in range(0, 400, 20):  #Dividing to 20 bins
        rhos = []
        for line in lines:
            _rho, _theta = line[0]  #Gets rho and theta to line
            x = get_x_by_rho_theta(_rho, _theta, theta)
            if (x >= i and x <= i + 20):  #If x value is in the bin
                rhos.append(_rho)
        bins_size.append(len(rhos))
    check = False
    for i in range(0, len(bins_size) - 1, 1):
        if (bins_size[i] > 20):  #If there are too much lines in one bin
            check = True
            break
    return check


def fix_threshold_if_load_lines(image, theta, hough_params):
    """
        This function fix the threshold of Hough Transform if it is too low.

        Parameters:
        image (matrix): The matrix of the image.
        theta (float): The angle we want the lines to be on (approximately).
        hough_params: The thresholds we want to check (Hough Transform and Canny, Canny is not changing in
                      this function).

        Returns:
        The fixed thresholds.

    """
    threshold = hough_params[0]
    min_angle = np.deg2rad(theta - 20)
    max_angle = np.deg2rad(theta + 20)
    #Calculate Canny with Canny's threshold (hough_params[1])
    edges = display_canny(image, hough_params[1])
    #Calculating the hough lines with the Hough Transform's threshold parameter (hough_params[0])
    lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=hough_params[0], min_theta=min_angle,
                           max_theta=max_angle)
    iterations = 0  #In case we won't be in the loop
    # We have too much lines and number of bins is correct
    while check_threshold_if_load_lines(lines, theta) and check_by_bins(lines, theta) == 0:
        threshold = threshold + 1
        # Calculating the hough lines with the new threshold
        lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=threshold, min_theta=min_angle,
                               max_theta=max_angle)
        iterations += 1
    if iterations == 0:  # We don't get into the loop
        return hough_params
    else:
        return threshold - 1, hough_params[1]


def get_hough_params(image, theta):
    """
        This function calculate the best thresholds for Canny and Hough Transform.

        Parameters:
        image (matrix): The matrix of the image.
        theta (float): The angle we want the lines to be on (approximately).

        Returns:
        Canny and Hough Transform thresholds.

    """
    params_thres_min = get_hough_params_by_base_threshold(image, theta, 50)
    params_thres_max = get_hough_params_by_base_threshold(image, theta, 200)
    params_res = 0  # The result by the base thresholds
    # If the Cannys' threshold is bigger in the smaller base_threshold
    if params_thres_min[1] > params_thres_max[1]:
        params_res = params_thres_max  # Choose the smaller Canny threshold
        # If the Cannys' threshold is bigger in the bigger base_threshold
    elif params_thres_min[1] < params_thres_max[1]:
        params_res = params_thres_min  # Choose the smaller Canny threshold
    elif params_thres_max[0] == 200:  # If the initial base_threshold is the best threshold
        # The result will be a weighted average of the bigger base_threshold and the smallest , with
        # more weight to the smallest
        params_res = int((0.58 * params_thres_min[0] + 0.42 * params_thres_max[0])), params_thres_min[1]
    elif params_thres_min[0] == 50:  # If the initial base_threshold is the best threshold
        # The result will be a weighted average of the bigger base_threshold and the smallest , with
        # more weight to the biggest
        params_res = int((0.42 * params_thres_min[0] + 0.58 * params_thres_max[0])), params_thres_max[1]
    else:
        # The result will be a weighted average of the bigger base_threshold and the smallest , with
        # more weight to the smallest (more lines preferred from missing lines)
        params_res = int((0.77 * params_thres_min[0] + 0.23 * params_thres_max[0])), params_thres_min[1]
    # Result of thresholds after decreasing some lines, if needed
    params_if_load_lines = fix_threshold_if_load_lines(image, theta, params_res)
    # If the difference between the original params and the new params is low (less than 5)
    if abs(params_if_load_lines[0] - params_res[0]) <= 5:
        return params_res  # Choose the original params (more lines preferred from missing lines)
    else:
        # The result will be a weighted average of the original params and the new ones
        return int(0.5 * params_if_load_lines[0] + 0.5 * params_res[0]), params_res[1]


def get_hough_params_by_base_threshold(image, theta, base_threshold):
    """
        This function calculate the best thresholds for Canny and Hough Transform according to
        initial threshold of Hough Transform (Canny's initial threshold is 50).

        Parameters:
        image (matrix): The matrix of the image.
        theta (float): The angle we want the lines to be on (approximately).
        base_threshold (float): The initial threshold we want the Hough Transform to be.

        Returns:
        Canny and Hough Transform thresholds.

    """
    original_base_threshold = base_threshold
    edges = display_canny(image)
    min_angle = np.deg2rad(theta - 20)  # Convert to radians
    max_angle = np.deg2rad(theta + 20)
    canny_threshold = 50
    flagForMissing = False  # If we decrease the threshold
    flagForMany = False  # If we increase the threshold
    while base_threshold > 0:
        if canny_threshold <= 0:
            break
        if flagForMany and flagForMissing:  # The correct threshold is between 2 thresholds with difference of 5
            canny_threshold = canny_threshold - 5
            # Calculating Canny with the canny_threshold
            edges = display_canny(image, canny_threshold)
            flagForMissing = False  # Back to initial data
            flagForMany = False
            base_threshold = original_base_threshold
        # Caluclating the hough lines with base_threshold
        lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=base_threshold, min_theta=min_angle,
                               max_theta=max_angle)
        if lines is None:
            base_threshold = base_threshold - 1
            continue
        bins_status = check_by_bins(lines, theta)  # Check if there are correct number of bins
        if bins_status == 1:  # Too many lines
            base_threshold = base_threshold + 5  # Strengthen the threshold
            flagForMany = True
        elif bins_status == -1:  # Too little lined
            base_threshold = base_threshold - 5  # Weaken the threshold
            flagForMissing = True
        else:
            return base_threshold, canny_threshold
    print("Picture not good enough")  # Cannot give to the image good thresholds
    return -1


def get_x_by_rho_theta(_rho, _theta, theta):
    """
        This function calculate x by the parametric representation of line : xsin(theta)+ycos(theta)=rho.

        Parameters:
        _rho (float): The rho of the line (distance from center to the line).
        _theta (float): The angle of the line (between rho and axis).
        theta (float): The angle we want the lines to be on (approximately)..

        Returns:
        Number which is the result of x.

    """
    if theta == 0:
        return (_rho - 200 * np.sin(_theta)) / (np.cos(_theta))  # y=200
    elif theta == 60:
        return _rho / (np.cos(_theta) + np.sin(_theta))  # y=x
    else:
        return (_rho - 400 * np.sin(_theta)) / (np.cos(_theta) - np.sin(_theta))  # y=-x+400


def check_by_bins(lines, theta):
    """
        This function check whether there are correct number of bins (between 6 and 12).

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta (float): The angle we want the lines to be on (approximately).

        Returns:
        -1 if there are too low number of bins.
        0 if there are correct number of bins.
        1 if there are too high number of bins.

        """
    bins = 0
    for i in range(0, 400, 20):  # Dividing to 20 bins
        rhos = []
        for line in lines:
            _rho, _theta = line[0]  # Gets rho and theta to line
            x = get_x_by_rho_theta(_rho, _theta, theta)
            if i <= x <= i + 20:  # If x value is in the bin
                rhos.append(_rho)
        if len(rhos) > 0:  # If there are lines on bin
            bins = bins + 1
    if bins >= 13:  # Too much bins
        return 1
    elif bins <= 6:  # Too little bins
        return -1
    else:
        return 0


def find_thetas_possible(theta, thetas, min_bin, max_bin):
    """
        This function finds the maximum and minimum thetas that is in the side bins.

        Parameters:.
        theta (float): The angle we want the lines to be on (approximately).
        thetas (array): Array of all thetas of all lines.
        min_bin (int): The minimum bin.
        max_bin (int): The maximum bin.

        Returns:
        Minimum and maximum thetas.

    """
    min_theta_possible = 10000000000
    max_theta_possible = -1000000000
    if theta == 0:
        min_theta_possible = min(thetas[min_bin])  # minimum theta of all lines
        max_theta_possible = max(thetas[max_bin])  # maximum theta of all lines
    if theta == 60:
        min_theta_possible = min(thetas[max_bin])  # minimum theta of all lines
        max_theta_possible = max(thetas[min_bin])  # maximum theta of all lines
    if theta == 110:
        min_theta_possible = min(thetas[max_bin])  # minimum theta of all lines
        max_theta_possible = max(thetas[min_bin])  # maximum theta of all lines
    return min_theta_possible, max_theta_possible


def create_thetas_from_iterations(lines, theta):
    """
        This function creates a list of all thetas of all lines.

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta (float): The angle we want the lines to be on (approximately).

        Returns:
        thetas list.

    """
    thetas = []  # List of thetas of lines
    for i in range(0, 400, 20):  # Dividing to 20 bins
        theta_for_line = []  # List of thetas for one line
        for line in lines:
            _rho, _theta = line[0]  # Gets rho and theta to line
            x = get_x_by_rho_theta(_rho, _theta, theta)
            if i <= x <= i + 20:  # If x value is in the bin
                theta_for_line.append(_theta)
        thetas.append(theta_for_line)
    return thetas


def find_min_max_bins_with_thetas(thetas):
    """
        This function finds the minimum and maximum bins indexes with lines in them.

        Parameters:
        thetas (array): Array of all thetas of all lines.

        Returns:
        Minimum and maximum bins indexes.

    """
    min_bin_index_with_lines = 0  # Index for first bin with lines
    max_bin_index_with_lines = 0  # Index for last bin with lines
    for i in range(20):
        if (len(thetas[i]) > 0):
            min_bin_index_with_lines = i  # Find the first bin with lines
            break
    for i in range(19, -1, -1):
        if (len(thetas[i]) > 0):
            max_bin_index_with_lines = i  # Find the last bin with lines
            break
    return min_bin_index_with_lines, max_bin_index_with_lines


def filter_lines_unusual_thetas(lines, theta):
    """
        This function filters the lines which their thata is not between thatas on side bins.

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta (float): The angle we want the lines to be on (approximately).

        Returns:
        Lines with thatas not in the side bins.

    """
    thetas = create_thetas_from_iterations(lines, theta)  # List of thethas of lines
    min_bin_index_with_lines = find_min_max_bins_with_thetas(thetas)[0]  # Index for first bin with lines
    max_bin_index_with_lines = find_min_max_bins_with_thetas(thetas)[1]  # Index for last bin with lines
    min_theta_possible = min(find_thetas_possible(theta, thetas, min_bin_index_with_lines, max_bin_index_with_lines)[0],
                             find_thetas_possible(theta, thetas, min_bin_index_with_lines, max_bin_index_with_lines)[
                                 1])  #minimum theta of all lines
    max_theta_possible = max(find_thetas_possible(theta, thetas, min_bin_index_with_lines, max_bin_index_with_lines)[1],
                             find_thetas_possible(theta, thetas, min_bin_index_with_lines, max_bin_index_with_lines)[
                                 0])  #maximum theta of all lines
    lines_filtered = []
    for line in lines:
        _rho, _theta = line[0]
        if (
                _theta <= max_theta_possible and _theta >= min_theta_possible):  #if theta is between minimum and maximum theta
            lines_filtered.append(line)
    return np.array(lines_filtered)
