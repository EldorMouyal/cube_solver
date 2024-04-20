
import kociemba
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


def display_canny(image):
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
    n_bins = 10
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


def hough_lines_for_theta(image, edges, theta: int, headline: str):
    # Convert the angle range to radians
    theta_rad = np.deg2rad(theta)

    # Define the angle range (in radians)
    min_angle = np.deg2rad(theta - 20)  # Convert to radians
    max_angle = np.deg2rad(theta + 20)  # Convert to radians

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, rho=2, theta=np.deg2rad(1), threshold=90, min_theta=min_angle, max_theta=max_angle)
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
    edges = display_canny(image)
    lines = hough_lines_for_theta(image=image, edges=edges, theta=theta, headline=headline)
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


def fill_face_top(image, h_lines, v_lines, face):
    return None


def fill_face_right(image, h_lines, v_lines, face):
    return None


def fill_face_left(image, h_lines, v_lines, face):
    return None


def main(name):
    # Example string and solution for a cube:
    cube = 'DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD'
    print(kociemba.solve(cube))
    # Actual program
    image_path = "rubix1.jpg"
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

