import numpy as np
import cv2


# Define the hue ranges for different colors in the HSV (Hue, Saturation, Value) domain
color_ranges = {
    'red': ([0, 100, 150], [2, 255, 255]),  # red
    'red2': ([160, 100, 100], [180, 255, 255]),
    'orange': ([4, 100, 100], [20, 255, 255]),  # orange
    'yellow': ([20, 150, 150], [38, 255, 255]),  # yellow
    'green': ([40, 100, 100], [80, 255, 255]),  # green
    'blue': ([95, 100, 100], [130, 255, 255]),  # blue
    'white': ([20, 0, 150], [180, 50, 255])  # white
}


def square_center(point1, point2, point3, point4) -> (int, int):
    """ this method returns the center of the square created by four points """
    y_values = [point[0] for point in [point1, point2, point3, point4]]
    x_values = [point[1] for point in [point1, point2, point3, point4]]
    center_y = int(sum(y_values) / 4)
    center_x = int(sum(x_values) / 4)
    return center_y, center_x


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


def get_ROI(image, points):
    points_np = np.array(points, dtype=np.int32)
    # Extract the ROI by creating an empty image and filling only the polygon created by the points with white.
    mask = np.zeros_like(image[:, :, 0])
    pts = points_np.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    roi = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("roi", roi)
    return roi


def most_common_color(image, points):
    global color_ranges
    roi = get_ROI(image, points)
    # Convert the ROI to the HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Initialize counters for each color
    color_counts = {color: 0 for color in color_ranges}

    # Count pixels in each color range
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        color_counts[color_name] = cv2.countNonZero(mask)
        print(color_name, color_counts[color_name])
    color_counts["red"] += color_counts["red2"]
    # Get the color with the maximum count
    most_common = max(color_counts, key=color_counts.get)
    return most_common


def fill_face(image, horizontal, vertical):
    colors = [""] * 9
    color_index = 0
    for i in range(3):
        for j in range(3):
            p1 = find_intersection_point_two_lines(line1=horizontal[i], line2=vertical[j],
                                                   frame_height=image.shape[0],
                                                   frame_width=image.shape[1])
            p2 = find_intersection_point_two_lines(line1=horizontal[i], line2=vertical[j + 1],
                                                   frame_height=image.shape[0],
                                                   frame_width=image.shape[1])
            p3 = find_intersection_point_two_lines(line1=horizontal[i + 1], line2=vertical[j],
                                                   frame_height=image.shape[0],
                                                   frame_width=image.shape[1])
            p4 = find_intersection_point_two_lines(line1=horizontal[i + 1], line2=vertical[j + 1],
                                                   frame_height=image.shape[0],
                                                   frame_width=image.shape[1])
            p = square_center(p1, p2, p3, p4)
            cv2.circle(image, p1, 3, (0, 0, 0), -1)
            cv2.circle(image, p2, 3, (0, 0, 0), -1)
            cv2.circle(image, p3, 3, (0, 0, 0), -1)
            cv2.circle(image, p4, 3, (0, 0, 0), -1)
            cv2.circle(image, p, 3, (0, 100, 0), -1)
            color = most_common_color(image=image, points=[p1, p2, p3, p4])
            print("the most dominant color is: ", color, "\n")
            colors[color_index] = color
            color_index += 1
            cv2.imshow("dots", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    print(colors)
    return colors


def sort_lines_by_theta(lines):
    sorted_lines = sorted(lines, key=lambda x: x[1])  # Sort based on the angle (theta)
    return sorted_lines


def sort_lines_top_by_rho(lines):
    sorted_lines = sorted(lines, key=lambda x: x[0])
    return sorted_lines


class RubiksCubeTriplet:
    def __init__(self, image, vertical_lines, sharp_lines, obtuse_lines):
        self.image = image
        self.vertical = sort_lines_top_by_rho(vertical_lines)
        self.sharp = sort_lines_top_by_rho(sharp_lines)
        self.obtuse = sort_lines_top_by_rho(obtuse_lines)
        self.center_color = ""
        # self.colors = [[[0, 0, 0] for _ in range(3)] for _ in range(3)]
        self.top_colors = [""] * 9
        self.left_colors = [""] * 9
        self.right_colors = [""] * 9
        self._fill_top_colors()
        self._fill_left_colors()
        self._fill_right_colors()

    def _fill_top_colors(self):
        vertical = list(reversed(self.obtuse[0:4]))
        horizontal = self.sharp[0:4]
        self.top_colors = fill_face(self.image.copy(), vertical, horizontal)

    def _fill_left_colors(self):
        vertical = self.vertical[0:4]
        horizontal = self.obtuse[3:7]
        self.left_colors = fill_face(self.image.copy(), vertical, horizontal)

    def _fill_right_colors(self):
        vertical = self.vertical[3:7]
        horizontal = self.sharp[3:7]
        self.right_colors = fill_face(self.image.copy(), vertical, horizontal)




