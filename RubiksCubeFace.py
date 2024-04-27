import numpy as np
import cv2


# Define the hue ranges for different colors in the HSV (Hue, Saturation, Value) domain
color_ranges = {
    'red': ([0, 200, 175], [5, 255, 255]),  # red
    'red2': ([170, 150, 255], [180, 255, 255]),
    'orange': ([4, 100, 150], [20, 255, 255]),  # orange
    'yellow': ([20, 150, 150], [38, 255, 255]),  # yellow
    'green': ([40, 150, 120], [75, 255, 255]),  # green
    'blue': ([75, 100, 150], [130, 255, 255]),  # blue
    'white': ([20, 0, 200], [100, 50, 255])  # white
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

    # Get the color with the maximum count
    most_common = max(color_counts, key=color_counts.get)

    return most_common


def get_color(r, g, b):  # compare rgb values and return color
    if (118 <= r <= 230) and (60 <= g <= 174) and (15 < b < 130):
        return 'blue'
    elif (148 <= r <= 250) and (140 <= g < 250) and (140 <= b < 250):
        return 'white'
    elif (21 <= r <= 118) and (130 < g < 255) and (150 < b < 255):
        return 'yellow'
    elif (0 < r <= 75) and (79 <= g <= 130) and (125 < b < 255):
        return 'orange'
    elif (10 <= r <= 70) and (20 <= g < 79) and (90 <= b < 255):
        return 'red'
    elif (40 <= r <= 116) and (130 < g <= 235) and (80 < b <= 170):
        return 'green'
    else:
        pass


def most_common_color2(image, points):
    roi = get_ROI(image, points)
    b, g, r = cv2.split(roi)
    r_avg = int(cv2.mean(r)[0])
    g_avg = int(cv2.mean(g)[0])
    b_avg = int(cv2.mean(b)[0])
    print(r_avg, g_avg, b_avg)
    res = get_color(r_avg, g_avg, b_avg)
    return res


class RubiksCubeFace:
    def __init__(self, image, vertical, horizontal):
        self.image = image
        self.vertical = vertical
        self.horizontal = horizontal
        self.center_color = ""
        # self.colors = [[[0, 0, 0] for _ in range(3)] for _ in range(3)]
        self.colors = [""] * 9

    def __str__(self):
        """
        Convert the Rubik's Cube face to a string representation.
        """
        # return "".join(["".join(row) for row in self.colors])
        return "Not Implemented yet"

    def fill_side(self, side="side"):
        p1 = None
        p2 = None
        p3 = None
        p4 = None
        # self.horizontal = list(reversed(self.horizontal))
        if side.lower() == "top":
            self.vertical = list(reversed(self.vertical))
        color_index = 0
        for i in range(3):
            for j in range(3):
                p1 = find_intersection_point_two_lines(line1=self.horizontal[i], line2=self.vertical[j],
                                                       frame_height=self.image.shape[0],
                                                       frame_width=self.image.shape[1])
                p2 = find_intersection_point_two_lines(line1=self.horizontal[i], line2=self.vertical[j + 1],
                                                       frame_height=self.image.shape[0],
                                                       frame_width=self.image.shape[1])
                p3 = find_intersection_point_two_lines(line1=self.horizontal[i + 1], line2=self.vertical[j],
                                                       frame_height=self.image.shape[0],
                                                       frame_width=self.image.shape[1])
                p4 = find_intersection_point_two_lines(line1=self.horizontal[i + 1], line2=self.vertical[j + 1],
                                                       frame_height=self.image.shape[0],
                                                       frame_width=self.image.shape[1])
                p = square_center(p1, p2, p3, p4)
                print(p1, p2, p3, p4)
                cv2.circle(self.image, p1, 3, (0, 0, 0), -1)
                cv2.circle(self.image, p2, 3, (0, 0, 0), -1)
                cv2.circle(self.image, p3, 3, (0, 0, 0), -1)
                cv2.circle(self.image, p4, 3, (0, 0, 0), -1)
                color = most_common_color(image=self.image, points=[p1, p2, p3, p4])
                print(color)
                cv2.circle(self.image, p, 3, (0, 100, 0), -1)
                self.colors[color_index] = color
                color_index += 1
                # print(color[2], color[1], color[0])
                cv2.imshow("dots", self.image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        print(self.colors)


