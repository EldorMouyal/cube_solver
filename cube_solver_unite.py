
#import kociemba
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
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


def display_image(image, title):
    cv2.imshow("title", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def display_canny(image,low_threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    display_image(edges, "Canny edges")
    return edges


def polar_to_cartesian(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def find_intersection_point_two_lines(line1, line2, frame_width, frame_height):
    rho1, theta1 = line1
    rho2, theta2 = line2
    # Convert polar lines to Cartesian coordinates
    x1, y1 = polar_to_cartesian(rho1, theta1)
    x2, y2 = polar_to_cartesian(rho2, theta2)
    # Find intersection point in Cartesian coordinates
    if np.sin(theta1 - theta2) == 0:
        return -1, -1  # Lines are parallel
    x_intersect = (rho1 * np.cos(theta2) - rho2 * np.cos(theta1)) / np.sin(theta1 - theta2)
    y_intersect = (rho1 * np.sin(theta2) - rho2 * np.sin(theta1)) / np.sin(theta2 - theta1)
    # Check if intersection point is within the frame size
    if 0 <= x_intersect <= frame_width and 0 <= y_intersect <= frame_height:
        return x_intersect, y_intersect
    else:
        return -1, -1


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
    n_bins = 10  #may be change to 10 15
    # Divide lines into Bins
    y_value = im_size/2
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
    n_bins = 30
    bins = [[] for _ in range(n_bins)]
    c_length = int(np.sqrt(2 * (im_size**2)))
    for rho, theta in rho_theta:
        x, y = polar_to_cartesian(rho, theta)
        dist_from_TL = int(np.sqrt(x**2 + x**2))  # TL = Top Left
        index = dist_from_TL / (c_length / n_bins)
        if 0 <= index < n_bins:
            bins[int(index)].append([rho, theta])
    print(bins)
    filtered = filter_closest_pair_to_average_rho(bins)
    return extract_pairs(filtered)


def filter_obtuse_anomalies(rho_theta, im_size):
    n_bins = 30
    bins = [[] for _ in range(n_bins)]
    c_length = int(np.sqrt(2 * (im_size**2)))
    for rho, theta in rho_theta:
        x, y = polar_to_cartesian(rho, theta)
        x_intersect = (im_size - y + x)/2
        y_intersect = im_size - x_intersect
        if 0 <= x_intersect <= im_size and 0 <= y_intersect <= im_size:
            dist_from_TR = int(np.sqrt((x_intersect - im_size)**2 + y_intersect**2))
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


def k_means_for_lines(lines):
    rho_theta = []
    rhos = []
    for rho, theta in lines:
        is_intersect = False
        # Check that if a line is chosen it does not intersect inside the picture
        for _rho, _theta in rho_theta:
            x, y = find_intersection_point_two_lines([rho, theta], [_rho, _theta], 400, 400)
            if x != -1 and y != -1:
                is_intersect = True
                break
        if not is_intersect:
            rho_theta.append([rho, theta])
            rhos.append(rho)
    print(rho_theta)
    if len(rho_theta) < 7:
        print("Picture not good enough")
        return
    rho_2d = np.array(rhos).reshape(-1, 1)
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(rho_2d)
    k_means = [item for sublist in kmeans.cluster_centers_ for item in sublist]  # flatten the cluster centers array
    print(k_means)
    return find_closest_rho(rho_theta, k_means)


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
    print(closest_pairs)
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


def hough_lines_for_theta(image, edges, theta: int, headline: str,hough_threshold: int):
    # Convert the angle range to radians
    theta_rad = np.deg2rad(theta)

    # Define the angle range (in radians)
    min_angle = np.deg2rad(theta - 20)  # Convert to radians
    max_angle = np.deg2rad(theta + 20)  # Convert to radians

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, rho=2, theta=np.deg2rad(1), threshold=hough_threshold, min_theta=min_angle, max_theta=max_angle)
    # lines = cv2.HoughLines(edges, 2,  np.pi / 180,  100)
    # Draw the detected lines on the original image
    draw_lines_by_polar(image, lines[:, 0], thickness=2)
    # Display the image with detected lines
    display_image(image, headline)
    return lines


def find_grid_for_theta(image, theta: int):
    print(np.deg2rad(theta))
    headline = "Vertical lines"
    if theta == 60:
        headline = "Sharp theta"
    if theta == 110:
        headline = "Obtuse theta"
    thresholds=get_hough_params(image, theta)
    edges = display_canny(image,thresholds[1])
    before_filter_lines = hough_lines_for_theta(image=image, edges=edges, theta=theta, headline=headline,hough_threshold=thresholds[0])
    print("lines before filter ",len(before_filter_lines))
    after_filter_lines=filter_lines_unusual_thetas(before_filter_lines,theta)
    print("lines after filter ",len(after_filter_lines))
    if(len(after_filter_lines)==0):
        lines=before_filter_lines
    else: lines=after_filter_lines
    if theta == 0:
        lines = filter_vertical_anomalies(lines[:, 0], im_size=image.shape[1])
    elif theta == 60:
        lines = filter_sharp_anomalies(lines[:, 0], image.shape[1])
    elif theta == 110:
        lines = filter_obtuse_anomalies(lines[:, 0], image.shape[0])
    draw_lines_by_polar(image=image, rho_theta=lines, color=(0, 255, 0))
    display_image(image, "after filtering")
    # rhos, avg_theta = k_means_for_lines(lines)
    # rhos, avg_theta = distinct_take_lines(lines)
    rho_theta = k_means_for_lines(lines)
    draw_lines_by_polar(image=image, rho_theta=rho_theta, color=(255, 0, 0), thickness=2)
    display_image(image, headline)
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

#Functions of determinating thresholds
def check_threshold_if_load_lines(lines,theta):
    """
        This function check whether the detected lines are loaded in small amount of bins.

        Parameters:
        lines (array): The lines from the Hough Transform.
        theta (float): The angle we want the lines to be on (approximately).

        Returns:
        True if there is a bin with more than 20 lines on it, means too many lines in one bin
        and False if there isn't.

    """
    bins_size=[]
    for i in range(0, 400, 20):   #Dividing to 20 bins
        rhos = []
        for line in lines:
            _rho, _theta = line[0]    #Gets rho and theta to line
            x = get_x_by_rho_theta(_rho, _theta, theta)
            if (x >= i and x <= i + 20):   #If x value is in the bin
                rhos.append(_rho)
        bins_size.append(len(rhos))
    check = False
    for i in range(0,len(bins_size)-1,1):
        if(bins_size[i] > 20):    #If there are too much lines in one bin
            check = True
            break
    return check


def fix_threshold_if_load_lines(image,theta,hough_params):
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
    threshold=hough_params[0]
    min_angle = np.deg2rad(theta - 20)
    max_angle = np.deg2rad(theta + 20)
    #Calculate Canny with Canny's threshold (hough_params[1])
    edges = display_canny(image, hough_params[1])
    #Calculating the hough lines with the Hough Transform's threshold parameter (hough_params[0])
    lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=hough_params[0], min_theta=min_angle,max_theta=max_angle)
    iterations=0    #In case we won't be in the loop
    # We have too much lines and number of bins is correct
    while(check_threshold_if_load_lines(lines,theta)==True and check_by_bins(lines,theta)==0):
        threshold=threshold+1
        # Calculating the hough lines with the new threshold
        lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=threshold, min_theta=min_angle,
                               max_theta=max_angle)
        iterations+=1
    if iterations==0:   #We don't get into the loop
        return hough_params
    else:
        return threshold-1,hough_params[1]


def get_hough_params(image,theta):
    """
        This function calculate the best thresholds for Canny and Hough Transform.

        Parameters:
        image (matrix): The matrix of the image.
        theta (float): The angle we want the lines to be on (approximately).

        Returns:
        Canny and Hough Transform thresholds.

    """
    params_thres_min = get_hough_params_by_base_threshold(image,theta,50)
    params_thres_max = get_hough_params_by_base_threshold(image, theta, 200)
    params_res = 0  #The result by the base thresholds
    # If the Cannys' threshold is bigger in the smaller base_threshold
    if((params_thres_min[1] > params_thres_max[1])):
        params_res=params_thres_max    #Choose the smaller Canny threshold
        # If the Cannys' threshold is bigger in the bigger base_threshold
    elif ((params_thres_min[1] < params_thres_max[1])):
        params_res = params_thres_min    #Choose the smaller Canny threshold
    elif (params_thres_max[0]==200):   #If the initial base_threshold is the best threshold
        #The result will be a weighted average of the bigger base_threshold and the smallest , with
        #more weight to the smallest
        params_res=int((0.58 * params_thres_min[0] + 0.42 * params_thres_max[0])), params_thres_min[1]
    elif(params_thres_min[0]==50):    #If the initial base_threshold is the best threshold
        # The result will be a weighted average of the bigger base_threshold and the smallest , with
        # more weight to the biggest
        params_res=int((0.42 * params_thres_min[0] + 0.58 * params_thres_max[0])), params_thres_max[1]
    else:
        # The result will be a weighted average of the bigger base_threshold and the smallest , with
        # more weight to the smallest (more lines prefered from missing lines)
        params_res=int((0.55 * params_thres_min[0] + 0.45 * params_thres_max[0])),params_thres_min[1]
    #Result of thresholds after decreasing some lines, if needed
    params_if_load_lines=fix_threshold_if_load_lines(image,theta,params_res)
    #If the difference between the original params and the new params is low (less than 5)
    if (abs(params_if_load_lines[0]-params_res[0])<=5):
        return params_res    #Choose the original params (more lines prefered from missing lines)
    else:
        # The result will be a weighted average of the original params and the new ones
        return int(0.5 * params_if_load_lines[0] + 0.5 * params_res[0]), params_res[1]


def get_hough_params_by_base_threshold(image,theta,base_threshold):
    """
        This function calculate the best thresholds for Canny and Hough Transform according to
        inital threshold of Hough Transform (Canny's initial threshold is 50).

        Parameters:
        image (matrix): The matrix of the image.
        theta (float): The angle we want the lines to be on (approximately).
        base_threshold (float): The initial threshold we want the Hough Transform to be.

        Returns:
        Canny and Hough Transform thresholds.

    """
    original_base_threshold=base_threshold
    edges = display_canny(image)
    min_angle = np.deg2rad(theta - 20)  # Convert to radians
    max_angle = np.deg2rad(theta + 20)
    canny_threshold = 50
    flagForMissing = False   #If we decrease the threshold
    flagForMany = False      #If we increase the threshold
    while(base_threshold > 0):
        if (canny_threshold <=0):
            break
        if (flagForMany and flagForMissing):     #The correct threshold is between 2 thresholds with difference of 5
            canny_threshold=canny_threshold-5
            # Caluclating Canny with the canny_threshold
            edges = display_canny(image,canny_threshold)
            flagForMissing = False       #Back to initial data
            flagForMany = False
            base_threshold=original_base_threshold
        #Caluclating the hough lines with base_threshold
        lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=base_threshold, min_theta=min_angle,max_theta=max_angle)
        if(lines is None):
            base_threshold=base_threshold-1
            continue
        res=check_by_bins(lines,theta)    #Check if there are correct number of bins
        if(check_by_bins(lines,theta)==1):    #Too many lines
            base_threshold=base_threshold+5    #Strengthen the threshold
            flagForMany = True
        elif(check_by_bins(lines,theta)==-1):    #Too littile lined
            base_threshold = base_threshold - 5     #Weaken the threshold
            flagForMissing = True
        else:
            return base_threshold,canny_threshold
    print("Please provide me another image")     #Cannot give to the image good thresholds
    return -1

def get_x_by_rho_theta(_rho, _theta,theta):
    """
        This function calculate x by the parametric representation of line : xsin(theta)+ycos(theta)=rho.

        Parameters:
        _rho (float): The rho of the line (distance from center to the line).
        _theta (float): The angle of the line (between rho and axis).
        theta (float): The angle we want the lines to be on (approximately)..

        Returns:
        Number which is the result of x.

    """
    if(theta == 0):
        return (_rho - 200 * np.sin(_theta)) / (np.cos(_theta))     #y=200
    elif(theta == 60):
        return _rho/(np.cos(_theta) + np.sin(_theta))    #y=x
    else:
        return (_rho-400*np.sin(_theta))/(np.cos(_theta)-np.sin(_theta))    #y=-x+400

def check_by_bins(lines,theta):
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
    for i in range(0, 400, 20):     #Dividing to 20 bins
        rhos=[]
        for line in lines:
            _rho, _theta = line[0]      #Gets rho and theta to line
            x = get_x_by_rho_theta(_rho, _theta, theta)
            if (x >= i and x <= i + 20):    #If x value is in the bin
                rhos.append(_rho)
        if (len(rhos) > 0):   #If there are lines on bin
            bins = bins + 1
    if (bins >= 13):    #Too much bins
        return 1
    elif (bins <= 6):   #Too little bins
        return -1
    else:
        return 0

def filter_lines_unusual_thetas(lines,theta):
    thetas = []   #List of thethas of lines
    for i in range(0, 400, 20):     #Dividing to 20 bins
        thetas_for_line=[]      #List of thetas for one line
        for line in lines:
            _rho, _theta = line[0]      #Gets rho and theta to line
            x = get_x_by_rho_theta(_rho, _theta, theta)
            if (x >= i and x <= i + 20):    #If x value is in the bin
                thetas_for_line.append(_theta)
        thetas.append(thetas_for_line)
    min_bin_index_with_lines = 0     #Index for first bin with lines
    max_bin_index_with_lines = 0     #Index for last bin with lines
    for i in range(20):
        if(len(thetas[i])>0):
            min_bin_index_with_lines = i    #Find the first bin with lines
            break
    for i in range(19,-1,-1):
        if(len(thetas[i])>0):
            max_bin_index_with_lines = i     #Find the last bin with lines
            break
    min_theta_possible = min(thetas[max_bin_index_with_lines])     #minimum theta of all lines
    max_theta_possible = max(thetas[min_bin_index_with_lines])     #maximum theta of all lines
    lines_filtered=[]
    for line in lines:
        _rho, _theta = line[0]
        if(_theta<=max_theta_possible and _theta>=min_theta_possible):   #if theta is between minimum and maximum theta
            lines_filtered.append(line)
    return np.array(lines_filtered)



def fill_face_top(image, h_lines, v_lines, face):
    return None


def fill_face_right(image, h_lines, v_lines, face):
    return None


def fill_face_left(image, h_lines, v_lines, face):
    return None


def main(name):
    # Example string and solution for a cube:
    cube = 'DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD'
    #print(kociemba.solve(cube))
    # Actual program
    image_path = "C:\\Users\\alex\\PycharmProjects\\pythonProject\\.venv\\cube_14.jpg"
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (400, 400))
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=60, headline="Sharp Angle Lines")
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=120, headline="Obtuse Angle Lines")
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=0, headline="Vertical Lines")
    find_grid_for_theta(resized_image, theta=110)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')

