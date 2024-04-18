
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


def display_image(image, title):
    cv2.imshow("title", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    display_image(edges, "Canny edges")
    return edges


def k_means_for_lines(lines):
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
    avg_thetas = avg_thetas/(len(rhos))
    print(rhos)
    print(avg_thetas)
    rho_2d = np.array(rhos).reshape(-1, 1)
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(rho_2d)
    # cluster_centers = sorted(kmeans.cluster_centers_)
    print(kmeans.cluster_centers_)
    return kmeans.cluster_centers_, avg_thetas


def k_means_for_lines2(lines):
    rho_theta = []
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
    print(rhos)
    print(avg_thetas)
    rho_2d = np.array(rhos).reshape(-1, 1)
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(rho_2d)
    # cluster_centers = sorted(kmeans.cluster_centers_)
    print(kmeans.cluster_centers_)
    return kmeans.cluster_centers_, avg_thetas


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
    draw_lines_from_hough(image, lines[:, 0], thickness=2)
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
    rhos, avg_theta = k_means_for_lines(lines)
    # rhos, avg_theta = distinct_take_lines(lines)
    rho_theta = [(r, avg_theta) for r in rhos]
    draw_lines_from_hough(image=image, rho_theta=rho_theta, color=(255, 0, 0), thickness=2)
    display_image(image, headline)
    return


def draw_lines_from_hough(image, rho_theta, color=(0, 0, 255), thickness=1):
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
    image_path = "rubix2.jpg"
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (400, 400))
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=60, headline="Sharp Angle Lines")
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=120, headline="Obtuse Angle Lines")
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=0, headline="Vertical Lines")
    find_grid_for_theta(resized_image, theta=60)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')

