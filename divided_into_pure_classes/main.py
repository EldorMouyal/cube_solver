import cv2

from detect_lines import *
from clustering import *
from Rubiks_cube import *
from RubiksCubeTriplet import *


def grid_for_image(image):
    vertical = find_grid_for_theta(image.copy(), 0)
    sharp = find_grid_for_theta(image.copy(), 60)
    obtuse = find_grid_for_theta(image.copy(), 110)
    return vertical, sharp, obtuse


def _build_cube_by_paths(path_top, path_bottom):
    top_image = cv2.imread(path_top)
    if top_image is None:
        print("bad path for image (top)")
        return
    bottom_image = cv2.imread(path_bottom)
    if bottom_image is None:
        print("bad path for image (bottom)")
        return
    top_image = cv2.resize(top_image, (400, 400))
    bottom_image = cv2.resize(bottom_image, (400, 400))
    cube = rubiks_cube()

    top_vertical, top_sharp, top_obtuse = grid_for_image(top_image)
    # draw_lines_by_polar(image=resized_top_copy, rho_theta=top_obtuse, color=(255, 0, 0))
    # draw_lines_by_polar(image=resized_top_copy, rho_theta=top_vertical, color=(255, 0, 0))
    # draw_lines_by_polar(image=resized_top_copy, rho_theta=top_sharp, color=(255, 0, 0))
    # display_image(resized_top_copy, "cube")
    bottom_vertical, bottom_sharp, bottom_obtuse = grid_for_image(bottom_image)
    faces_URF = RubiksCubeTriplet(image=top_image, vertical_lines=top_vertical, sharp_lines=top_sharp,
                                  obtuse_lines=top_obtuse)
    faces_DLB = RubiksCubeTriplet(image=bottom_image, vertical_lines=bottom_vertical, sharp_lines=bottom_sharp,
                                  obtuse_lines=bottom_obtuse)
    cube.set_URF(up_colors=faces_URF.top_colors, front_colors=faces_URF.left_colors,
                 right_colors=faces_URF.right_colors)
    cube.set_DLB(down_colors=faces_DLB.top_colors, left_colors=faces_DLB.left_colors,
                 back_colors=faces_DLB.right_colors)
    return cube


def main(name):
    top_path = "../cube_17.jpg"
    bottom_path = "../cube_18.jpg"
    # top_path = input("Enter first path: ")
    # bottom_path = input("Enter second path: ")
    cube = _build_cube_by_paths(top_path, bottom_path)
    print("cube string:", cube.get_cube_string())
    print("solution: ", cube.get_cube_solution())
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
