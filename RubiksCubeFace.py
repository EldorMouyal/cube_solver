import numpy as np
import cv2


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


def most_common_color(image, points) -> [int, int, int]:
    points_np = np.array(points, dtype=np.int32)
    # Determine the bounding box
    x, y, w, h = cv2.boundingRect(points_np)

    # Extract the region of interest (ROI) from the image
    roi = image[y:y + h, x:x + w]

    # Convert the ROI to the HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define the hue ranges for different colors in the HSV (Hue, Saturation, Value) domain
    color_ranges = {
        'red': ([-10, 170, 50], [5, 255, 255]),  # red
        'orange': ([5, 150, 100], [20, 255, 255]),  # orange
        'yellow': ([20, 100, 100], [30, 255, 255]),  # yellow
        'green': ([40, 100, 100], [80, 255, 255]),  # green
        'blue': ([90, 100, 100], [130, 255, 255]),  # blue
        'white': ([0, 0, 200], [180, 50, 255])  # white
    }

    # Initialize counters for each color
    color_counts = {color: 0 for color in color_ranges}

    # Count pixels in each color range
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        color_counts[color_name] = cv2.countNonZero(mask)

    # Get the color with the maximum count
    most_common = max(color_counts, key=color_counts.get)

    return most_common


class RubiksCubeFace:
    def __init__(self, image, lines1, lines2):
        self.image = image
        self.lines1 = lines1
        self.lines2 = lines2
        self.center_color = ""
        # self.colors = [[[0, 0, 0] for _ in range(3)] for _ in range(3)]
        self.colors = [["" for _ in range(3)] for _ in range(3)]

    def __str__(self):
        """
        Convert the Rubik's Cube face to a string representation.
        """
        # return "".join(["".join(row) for row in self.colors])
        return "Not Implemented yet"

    def fill_colors_top(self):
        p1 = None
        p2 = None
        p3 = None
        p4 = None

        for i in range(3):
            for j in range(3):
                p1 = find_intersection_point_two_lines(line1=self.lines1[i], line2=self.lines2[j], frame_height=self.image.shape[0],
                                                       frame_width=self.image.shape[1])
                p2 = find_intersection_point_two_lines(line1=self.lines1[i], line2=self.lines2[j + 1], frame_height=self.image.shape[0],
                                                       frame_width=self.image.shape[1])
                p3 = find_intersection_point_two_lines(line1=self.lines1[i + 1], line2=self.lines2[j], frame_height=self.image.shape[0],
                                                       frame_width=self.image.shape[1])
                p4 = find_intersection_point_two_lines(line1=self.lines1[i + 1], line2=self.lines2[j + 1], frame_height=self.image.shape[0],
                                                       frame_width=self.image.shape[1])
                p = square_center(p1, p2, p3, p4)
                cv2.circle(self.image, p1, 3, (0, 0, 0), -1)
                cv2.circle(self.image, p2, 3, (0, 0, 0), -1)
                cv2.circle(self.image, p3, 3, (0, 0, 0), -1)
                cv2.circle(self.image, p4, 3, (0, 0, 0), -1)
                color = most_common_color(image=self.image, points=[p1, p2, p3, p4])
                cv2.circle(self.image, p, 3, (0, 100, 0), -1)
                self.colors[j][i] = color
                # print(color[2], color[1], color[0])
                # print(color)
        cv2.imshow("dots", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(self.colors)

