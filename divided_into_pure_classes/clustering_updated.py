
#import kociemba
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from detect_lines import *
import bisect

# I installed a package called kociemba
# to solve a rubix cube all we have to do is:
#     cube = 'DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD'
#     print(kociemba.solve(cube))
# The configuration string of the cube corresponds to the color of the stickers according to the following figure
#              |************|
#              |*U1**U2**U3*|
#              |************|
#              |*U4**U5**U6*|
#              |************|
#              |*U7**U8**U9*|
#              |************|
#  ************|************|************|************
#  *L1**L2**L3*|*F1**F2**F3*|*R1**R2**R3*|*B1**B2**B3*
#  ************|************|************|************
#  *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*
#  ************|************|************|************
#  *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*
#  ************|************|************|************
#              |************|
#              |*D1**D2**D3*|
#              |************|
#              |*D4**D5**D6*|
#              |************|
#              |*D7**D8**D9*|
#              |************|
# TODO: define constant variables such as frame size, vertical angle, sharp andle, dull angle...

def polar_to_cartesian(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def find_intersection_point_two_lines(line1, line2, frame_width, frame_height) -> (int, int):
    rho1, theta1 = line1
    rho2, theta2 = line2

    # Check if lines are parallel
    if np.isclose(np.sin(theta1 - theta2), 0):
        return -1, -1  # Lines are parallel, return a sentinel value

    # Calculate intersection point in Cartesian coordinates
    x_intersect = (rho1 * np.cos(theta2) - rho2 * np.cos(theta1)) / np.sin(theta1 - theta2)
    y_intersect = (rho1 * np.sin(theta2) - rho2 * np.sin(theta1)) / np.sin(theta2 - theta1)

    # Check if intersection point is within the frame size
    if 0 <= x_intersect <= frame_width and 0 <= y_intersect <= frame_height:
        # Round to integers and return as a tuple
        return int(round(y_intersect)), int(round(x_intersect))
    else:
        return -1, -1  # Intersection point is out of bounds, return a sentinel value


def find_intersection_point_neg_slope(line_rho, line_theta, im_size):
    # Convert polar line to Cartesian coordinates
    line_x, line_y = polar_to_cartesian(line_rho, line_theta)

    # Find intersection point with the line y = im_size - x
    x_intersect = (im_size - line_y + line_x) / 2
    y_intersect = im_size - x_intersect

    # Check if intersection point lies within the bounds of the frame
    if 0 <= x_intersect <= im_size and 0 <= y_intersect <= im_size:
        return x_intersect, y_intersect
    else:
        return None


def extract_pairs(array_of_arrays):
    res = []
    for sub_array in array_of_arrays:
        res.extend(sub_array)
    return res


def find_x_given_y(rho, theta, y):
    x, y_polar = polar_to_cartesian(rho, theta)
    # Calculate the x value when y = y_polar
    if y_polar == y:
        return x
    else:
        # The line is not parallel to the y-axis, so solve for x using the equation of the line
        x = (rho - y * np.sin(theta)) / np.cos(theta)
        # print("x value for ", rho, ", ", theta, " when y =", y, ":", x)
        return x


def filter_vertical_anomalies(rho_theta, im_size):
    n_bins = 20
    # Divide lines into Bins
    y_value = im_size / 2
    bins = [[] for _ in range(n_bins)]
    for rho, theta in rho_theta:
        x_val = int(find_x_given_y(rho, theta, y_value))
        index = x_val / (im_size / n_bins)
        # print(index)
        if 0 <= index < n_bins:
            bins[int(index)].append([rho, theta])
    print(bins)
    # filtered = filter_anomalies_theta(bins, 0.2)
    filtered = filter_closest_pair_to_average_theta(bins)
    return extract_pairs(filtered)


def filter_sharp_anomalies(rho_theta, im_size):
    n_bins = 80
    bins = [[] for _ in range(n_bins)]
    c_length = int(np.sqrt(2 * (im_size ** 2)))
    for rho, theta in rho_theta:
        x, y = polar_to_cartesian(rho, theta)
        dist_from_TL = int(np.sqrt(x ** 2 + x ** 2))  # TL = Top Left
        index = dist_from_TL / (c_length / n_bins)
        if 0 <= index < n_bins:
            bins[int(index)].append([rho, theta])
    print(bins)
    filtered = filter_closest_pair_to_average_rho(bins)
    return extract_pairs(filtered)


def filter_obtuse_anomalies(rho_theta, im_size):
    n_bins = 30
    bins = [[] for _ in range(n_bins)]
    c_length = int(np.sqrt(2 * (im_size ** 2)))
    for rho, theta in rho_theta:
        x, y = polar_to_cartesian(rho, theta)
        x_intersect = (im_size - y + x) / 2
        y_intersect = im_size - x_intersect
        if 0 <= x_intersect <= im_size and 0 <= y_intersect <= im_size:
            dist_from_TR = int(np.sqrt((x_intersect - im_size) ** 2 + y_intersect ** 2))
            index = dist_from_TR / (c_length / n_bins)
            if 0 <= index < n_bins:
                bins[int(index)].append([rho, theta])
    print(bins)
    filtered = filter_closest_pair_to_average_rho(bins)
    return extract_pairs(filtered)


def filter_anomalies_theta(bins, threshold_percent):
    filtered_bins = []
    for _bin in bins:
        if not _bin:  # Skip empty bins
            continue
        # Calculate the average of the second values
        second_values = [pair[1] for pair in _bin]
        avg_theta = sum(second_values) / len(second_values)
        # Filter pairs based on the deviation from the average
        filtered_pairs = []
        for pair in _bin:
            if abs(pair[1] - avg_theta) / avg_theta <= threshold_percent:
                filtered_pairs.append(pair)
        filtered_bins.append(filtered_pairs)
    return filtered_bins


def filter_closest_pair_to_average_theta(bins):
    filtered_bins = []
    for _bin in bins:
        if not _bin:  # Skip empty bins
            continue
        # Calculate the average of the second values
        second_values = [pair[1] for pair in _bin]
        avg_theta = sum(second_values) / len(second_values)
        # Find the pair whose second value is closest to the average
        closest_pair = min(_bin, key=lambda pair: abs(pair[1] - avg_theta))
        filtered_bins.append([closest_pair])
    return filtered_bins


def filter_closest_pair_to_average_rho(bins):
    filtered_bins = []
    for _bin in bins:
        if not _bin:  # Skip empty bins
            continue
        # Calculate the average of the second values
        first_values = [pair[0] for pair in _bin]
        avg_rho = sum(first_values) / len(first_values)
        # Find the pair whose second value is closest to the average
        closest_pair = min(_bin, key=lambda pair: abs(pair[1] - avg_rho))
        filtered_bins.append([closest_pair])
    return filtered_bins

def find_closest_pair_to_avg_theta(rho_theta):    #NOT USED YET
    avg_theta = sum(second for _, second in rho_theta) / len(rho_theta)
    closest_pair = min(rho_theta, key=lambda x: abs(x[1] - avg_theta))
    return closest_pair


def perform_k_means(array,rho_theta):
    """
        This function perform k-means clustering on a set of rhos.

        Parameters:
        rhos (array): Array of rho values.
        rho_theta (array): Array of lines representing the bins.

        Returns:
        The rhos representing lines chosen by k-means.

    """
    rho_2d = np.array(array).reshape(-1, 1)
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(rho_2d)
    k_means = [item for sublist in kmeans.cluster_centers_ for item in sublist]  # flatten the cluster centers array
    print("k_means ", k_means)
    return find_closest_rho(rho_theta, k_means)

def create_rho_theta_for_not_vertical(lines,theta_all):
    """
        This function filters not vertical lines according to their closeness and take the max rho from every filtered line.

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta_all : The direction angle of the lines.

        Returns:
        Array of lines : rhos and thetas.

     """
    rho_theta = []
    for index in range(len(lines)-1,-1,-1):
        rho=lines[index][0]
        theta=lines[index][1]
        is_intersect = False
        for _rho, _theta in rho_theta:
            x, y = find_intersection_point_two_lines([rho, theta], [_rho, _theta], 400, 400)
            if x != -1 and y != -1:
                is_intersect = True
                break
        if not is_intersect:  # There are not intersected lines
            rho_theta.append([rho,theta])  # Taking the line with min rho
    return rho_theta


def create_rho_theta_for_vertical(lines,theta_all):
    """
        This function filters vertical lines according to their closeness and take the max rho from every filtered line.

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta_all : The direction angle of the lines.

        Returns:
        Array of lines : rhos and thetas.

    """
    rho_theta = []
    for rho, theta in lines:
        is_intersect = False
        # Check if 2 lines are intersecting in the image
        for _rho, _theta in rho_theta:
            x, y = find_intersection_point_two_lines([rho, theta], [_rho, _theta], 400, 400)
            if x != -1 and y != -1:
                is_intersect = True
                break
        if not is_intersect:    #There are not intersected lines
            rho_theta.append([rho, theta])       #Taking the line with min rho
    return rho_theta

def sides_are_in(rho_theta,side_rho_1,side_rho_2):
    """
        This function checks if sides are in rho_theta.

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta_all : The direction angle of the lines.

        Returns:
        True if is and False otherwise.

    """
    rho_theta.sort(key=lambda x: x[0])
    if side_rho_1!=rho_theta[0][0] or side_rho_2!=rho_theta[len(rho_theta)-1][0]:
        return False
    else :
        return True

def remove_duplicates(list):
    """
        This function removes duplicates from a list and sorts it.

        Parameters:
        list (list): A list to remove duplicates and sorts.

        Returns:
        The list without duplicates and sorted.

    """
    list_without_duplicates = []
    [list_without_duplicates.append(x) for x in list if x not in list_without_duplicates]  #remove duplicates
    list_without_duplicates.sort(key=lambda x: x[0])  #sort the list according rho
    return list_without_duplicates


def filter_lines_distance(rho_theta,theta_all):
    """
        This function filters the list from too close lines.

        Parameters:
        rho_theta (array): Array of lines.
        theta_all (float): The direction angle of the lines.

        Returns:
        The list without too close lines.

    """
    rho_theta_new=[]
    for i in range(0,len(rho_theta)-1):
        r1=rho_theta[i][0]
        r2=rho_theta[i+1][0]
        t1=rho_theta[i][1]
        t2=rho_theta[i+1][1]
        distance=get_distance_between_points(r1,r2,t1,t2,theta_all)
        if ((theta_all==0 and distance >= 30) or (theta_all!=0 and distance >= 20)):     #distance is not too close
            rho_theta_new.append(rho_theta[i])
            rho_theta_new.append(rho_theta[i+1])
    rho_theta_new.append(rho_theta[len(rho_theta) - 1])
    rho_theta_new=remove_duplicates(rho_theta_new)
    if (len(rho_theta_new)==7):
        return rho_theta_new
    else:
        return filter_closest_lines(rho_theta_new,theta_all,30,15)

def filter_closest_lines(rho_theta,theta_all,dist_to_zero,dist_to_not_zero):
    rho_theta_new = []
    rho_theta_new.append(rho_theta[0])
    for i in range(1, len(rho_theta)):
        r1 = rho_theta[i-1][0]
        r2 = rho_theta[i][0]
        t1 = rho_theta[i-1][1]
        t2 = rho_theta[i][1]
        distance = get_distance_between_points(r1, r2, t1, t2, theta_all)
        if theta_all == 0 and distance >= dist_to_zero:
            rho_theta_new.append(rho_theta[i])
        if theta_all!=0 and distance >= dist_to_not_zero:  # distance is not too close
            rho_theta_new.append(rho_theta[i])
    return rho_theta_new

def check_dist_too_far(rho_theta,theta_all):
    for i in range(1, len(rho_theta)):
        r1 = rho_theta[i - 1][0]
        r2 = rho_theta[i][0]
        t1 = rho_theta[i - 1][1]
        t2 = rho_theta[i][1]
        distance = get_distance_between_points(r1, r2, t1, t2, theta_all)
        if (distance >80):
            return True
    return False


def get_distance_between_points(r1,r2,t1,t2,theta_all):
    """
        This function calculates the distance between two points according the theat_all.

        Parameters:
        r1 (float): The first rho.
        r2 (float): The second rho.
        t1 (float): The first theta.
        t2 (float): The second theta.
        theta_all (float): The direction angle of the lines.

        Returns:
        Distance between two points.

    """
    x1 = get_x_by_rho_theta(r1, t1, theta_all)
    x2 = get_x_by_rho_theta(r2, t2, theta_all)
    distance = 0
    if (theta_all == 0):     #case for theta 0
        distance = np.sqrt((x2 - x1) ** 2)
    elif (theta_all == 60):       #case for theta 60
        distance = np.sqrt((x2 - x1) ** 2 + (x2 - x1) ** 2)
    else:            #case for theta 110
        distance = np.sqrt((x2 - x1) ** 2 + (x1 - x2) ** 2)
    return distance

def filter_with_increase_decrease_thetas(lines):
    new_lines=[]
    for i in range(1,len(lines)):
        if lines[i-1][1]<lines[i][1]:
            new_lines.append(lines[i-1])
    new_lines.append(lines[len(lines)-1])
    return new_lines

def create_rho_theta(lines,theta_all):
    rho_theta=[]
    if (theta_all != 0):
        rho_theta = create_rho_theta_for_not_vertical(lines, np.deg2rad(theta_all))
    else:
        rho_theta = create_rho_theta_for_vertical(lines, theta_all)
    rho_theta = remove_duplicates(rho_theta)
    rho_theta = filter_lines_distance(rho_theta, theta_all)
    return rho_theta

def remove_min_dist_line(rho_theta,theta_all):
    distances=[]
    for i in range(0,len(rho_theta)-1):
        r1=rho_theta[i][0]
        r2=rho_theta[i+1][0]
        t1=rho_theta[i][1]
        t2=rho_theta[i+1][1]
        distance=get_distance_between_points(r1,r2,t1,t2,theta_all)
        distances.append(distance)
    min_dist=min(distances)
    for i in range(0,len(rho_theta)-1):
        r1=rho_theta[i][0]
        r2=rho_theta[i+1][0]
        t1=rho_theta[i][1]
        t2=rho_theta[i+1][1]
        distance=get_distance_between_points(r1,r2,t1,t2,theta_all)
        if (distance==min_dist):
            rho_theta.remove(rho_theta[i+1])
            return rho_theta
    return rho_theta

def check_dist_small(lines,rho_theta,theta_all,flag):
    if (flag == 0):
        r1 = rho_theta[0][0]
        r2 = lines[0][0]
        t1 = rho_theta[0][1]
        t2 = lines[0][1]
        distance = get_distance_between_points(r1, r2, t1, t2, theta_all)
        if (distance <=6):
            return True
    if (flag == 1):
        r1 = rho_theta[len(rho_theta) - 1][0]
        r2 = lines[len(lines) - 1][0]
        t1 = rho_theta[len(rho_theta) - 1][1]
        t2 = lines[len(lines) - 1][1]
        distance = get_distance_between_points(r1, r2, t1, t2, theta_all)
        if (distance <=6):
            return True
    return False


def k_means_for_lines(image,lines,theta_all):
    """
        This function creates lines that will represent the lines of the cube.

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta_all : The direction angle of the lines.

        Returns:
        New list of lines representing the lines of the cube.

    """
    rho_theta=[]
    rhos = []
    thetas = []
    if(theta_all!=0):
        rho_theta = create_rho_theta(lines,theta_all)
    else:
        rho_theta_1 = filter_with_increase_decrease_thetas(lines)
        rho_theta_1=create_rho_theta(rho_theta_1,theta_all)
        rho_theta = create_rho_theta(lines, theta_all)
    if(theta_all==0 and check_dist_too_far(rho_theta_1,theta_all)==False):
        rho_theta=rho_theta_1
    a=35
    b=20
    while(len(rho_theta)>7):
        rho_theta=filter_closest_lines(rho_theta,theta_all,a,b)
        a=a+1
        b=b+1
    if (len(rho_theta)<7 and lines[0] not in rho_theta and check_dist_small(lines,rho_theta,theta_all,0)==False):
        rho_theta.append(lines[0])
    elif (len(rho_theta)<7 and lines[len(lines) - 1] not in rho_theta and check_dist_small(lines,rho_theta,theta_all,1)==False):
        rho_theta.append(lines[len(lines) - 1])
    elif (len(rho_theta)==7 and ((lines[len(lines) - 1] not in rho_theta) and check_dist_small(lines,rho_theta,theta_all,1)==False)):
        rho_theta=remove_min_dist_line(rho_theta,theta_all)
        rho_theta.append(lines[len(lines) - 1])
    elif (len(rho_theta)==7 and ((lines[0] not in rho_theta) and check_dist_small(lines,rho_theta,theta_all,0)==False)):
        rho_theta=remove_min_dist_line(rho_theta,theta_all)
        rho_theta.append(lines[0])


    #If the side line is not in rho_theta or there is line close enough to side line already
   # if(lines[0] not in rho_theta and get_distance_between_points(rho_theta[0][0],lines[0][0],rho_theta[0][1],lines[0][1],theta_all)>5):
   #     rho_theta.append(lines[0])
   # if (lines[len(lines)-1] not in rho_theta and
   #         get_distance_between_points(rho_theta[len(rho_theta)-1][0],lines[len(lines)-1][0],rho_theta[len(rho_theta)-1][1],lines[len(lines)-1][1],theta_all)>5):
   #     rho_theta.append(lines[len(lines)-1])

    for rho, theta in rho_theta:
        rhos.append(rho)
        thetas.append(theta)

    print("rho_theta ", rho_theta)
    draw_lines_by_polar(image=image, rho_theta=rho_theta, color=(0, 0, 0),thickness=1)
    display_image(image, "after filtering")

    rhos.sort()
    thetas.sort()
    unique_thetas = list(filter(lambda x: thetas.count(x) == 1, thetas))
    if (len(rho_theta) < 7):   #Choosing was not possible
        print("Picture not good enough")
        return
    elif(len(unique_thetas) == 7):    #We can take lines according thetas
        unique_thetas = list(filter(lambda x: thetas.count(x) == 1, thetas))
        return find_closest_theta(rho_theta, unique_thetas)   #Adapting to the lines we have
    else:
        return rho_theta

def find_closest_rho(rho_theta, arr):
    rho_theta.sort(key=lambda x: x[0])  # Sort the larger array based on the first values
    closest_pairs = []
    for y in arr:
        # Use bisect to find the insertion point of x in the sorted Y array
        index = bisect.bisect_left([pair[0] for pair in rho_theta], y)

        # Check if r is greater than all numbers in K or if r is less than all numbers in K
        if index == len(rho_theta):
            closest_pairs.append(rho_theta[-1])
        elif index == 0:
            closest_pairs.append(rho_theta[0])
        else:
            # Get the closest pair in rho_theta to y based on the first value in each pair
            closest_pair = rho_theta[index] if rho_theta[index][0] - y < y - rho_theta[index-1][0] else rho_theta[index-1]
            closest_pairs.append(closest_pair)
    print("Closest_pairs ",closest_pairs)
    return closest_pairs

def find_closest_theta(rho_theta, arr):
    rho_theta.sort(key=lambda x: x[1])  # Sort the larger array based on the first values
    closest_pairs = []
    for y in arr:
        # Use bisect to find the insertion point of x in the sorted Y array
        index = bisect.bisect_left([pair[1] for pair in rho_theta], y)

        # Check if r is greater than all numbers in K or if r is less than all numbers in K
        if index == len(rho_theta):
            closest_pairs.append(rho_theta[-1])
        elif index == 0:
            closest_pairs.append(rho_theta[0])
        else:
            # Get the closest pair in rho_theta to y based on the first value in each pair
            closest_pair = rho_theta[index] if rho_theta[index][1] - y < y - rho_theta[index-1][1] else rho_theta[index-1]
            closest_pairs.append(closest_pair)
    print("Closest_pairs ",closest_pairs)
    return closest_pairs


def distinct_take_lines(lines):  # DOESN'T WORK
    rhos = []
    avg_thetas = 0
    for line in lines:
        rho, theta = line[0]
        is_distinct = True
        # Check if rho is distinct from every other rho by more than 3
        for existing_rho in rhos:
            if abs(rho - existing_rho) <= 20:
                is_distinct = False
                break
        if is_distinct:
            rhos.append(rho)
            avg_thetas += theta
    avg_thetas = avg_thetas / (len(rhos))
    print(avg_thetas)
    sorted_rhos = np.sort(np.array(rhos))
    differences = np.diff(sorted_rhos)
    indices = np.argsort(differences)[-7:]
    most_distinct = sorted_rhos[indices]
    print(most_distinct)
    return most_distinct, avg_thetas

def find_grid_for_theta(image, theta: int):
    print(np.deg2rad(theta))
    headline = "Vertical lines"
    if theta == 60:
        headline = "Sharp theta"
    if theta == 110:
        headline = "Obtuse theta"
    thresholds=get_hough_params(image, theta)
    edges = display_canny(image,thresholds[1])
    lines = hough_lines_for_theta(image=image, edges=edges, theta=theta, headline=headline,hough_threshold=thresholds[0])
    #print("lines before filter ",len(lines))
    after_filter_lines=filter_lines_unusual_thetas(lines,theta)
    #print("lines after filter ",len(after_filter_lines))
    if(len(after_filter_lines)-len(lines)!=0):
        lines=after_filter_lines
    if(len(after_filter_lines)==0 or len(lines)==0):
        print("Picture not good enough")
        return
    if theta == 0:
        lines = filter_vertical_anomalies(lines[:, 0], im_size=image.shape[1])
    elif theta == 60:
        lines = filter_sharp_anomalies(lines[:, 0], image.shape[1])
    elif theta == 110:
        lines = filter_obtuse_anomalies(lines[:, 0], image.shape[0])
    draw_lines_by_polar(image=image, rho_theta=lines, color=(0, 255, 0))
    display_image(image, "after filtering")
    rho_theta = k_means_for_lines(image,lines,theta)
    draw_lines_by_polar(image=image, rho_theta=rho_theta, color=(255, 0, 0), thickness=2)
    display_image(image, headline)
    return

def sort_lines_left_to_right(lines):
    sorted_lines = sorted(lines, key=lambda x: x[1])  # Sort based on the angle (theta)
    return sorted_lines


def sort_lines_top_to_bottom(lines):
    sorted_lines = sorted(lines, key=lambda x: x[0])
    return sorted_lines


def main(name):
    # Example string and solution for a cube:
    cube = 'DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD'
    #print(kociemba.solve(cube))
    # Actual program
    image_path = "C:\\Users\\alex\\PycharmProjects\\pythonProject\\.venv\\cube_13.jpg"
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (400, 400))
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=60, headline="Sharp Angle Lines")
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=120, headline="Obtuse Angle Lines")
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=0, headline="Vertical Lines")
    find_grid_for_theta(resized_image, theta=110)


    #print(get_hough_params(resized_image,0))
    #edges = display_canny(resized_image,50)
    #lines = hough_lines_for_theta(image=resized_image, edges=edges, theta=60, headline="Vertical Lines",hough_threshold=115)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')

