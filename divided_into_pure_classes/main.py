from detect_lines import *
from clustering import *
from Rubiks_cube import *
from RubiksCubeTriplet import *

def main(name):
    # Example string and solution for a cube:
    cube = 'DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD'
    # print(kociemba.solve(cube))
    # Actual program
    cube = rubiks_cube()
    image_path_top = "../cube_17.jpg"
    top_image = cv2.imread(image_path_top)
    resized_top = cv2.resize(top_image, (400, 400))
    resized_top_copy = resized_top.copy()
    top_vertical = find_grid_for_theta(resized_top.copy(), theta=0)
    top_sharp = find_grid_for_theta(resized_top.copy(), theta=60)
    top_obtuse = find_grid_for_theta(resized_top.copy(), theta=110)
    draw_lines_by_polar(image=resized_top_copy, rho_theta=top_obtuse, color=(255, 0, 0))
    draw_lines_by_polar(image=resized_top_copy, rho_theta=top_vertical, color=(255, 0, 0))
    draw_lines_by_polar(image=resized_top_copy, rho_theta=top_sharp, color=(255, 0, 0))
    display_image(resized_top_copy, "cube")
    faces_URF = RubiksCubeTriplet(image=resized_top, vertical_lines=top_vertical, sharp_lines=top_sharp, obtuse_lines=top_obtuse)
    cube.set_URF(up_colors=faces_URF.top_colors, front_colors=faces_URF.left_colors, right_colors=faces_URF.right_colors)

    # for v in vertical:
    #     draw_lines_by_polar(resized_image, [v], color=(0, 255, 0))
    #     display_image(image=resized_image, title="sorted")

    image_path_bottom = "../cube_18.jpg"
    bottom_image = cv2.imread(image_path_bottom)
    resized_bottom = cv2.resize(bottom_image, (400, 400))
    resized_bottom_copy = resized_bottom.copy()
    bottom_vertical = find_grid_for_theta(resized_bottom.copy(), theta=0)
    bottom_sharp = find_grid_for_theta(resized_bottom.copy(), theta=60)
    bottom_obtuse = find_grid_for_theta(resized_bottom.copy(), theta=110)
    draw_lines_by_polar(image=resized_bottom_copy, rho_theta=bottom_obtuse, color=(255, 0, 0))
    draw_lines_by_polar(image=resized_bottom_copy, rho_theta=bottom_vertical, color=(255, 0, 0))
    draw_lines_by_polar(image=resized_bottom_copy, rho_theta=bottom_sharp, color=(255, 0, 0))
    display_image(resized_bottom_copy, "cube")
    faces_DLB = RubiksCubeTriplet(image=resized_bottom, vertical_lines=bottom_vertical, sharp_lines=bottom_sharp, obtuse_lines=bottom_obtuse)
    cube.set_DLB(down_colors=faces_DLB.top_colors, left_colors=faces_DLB.left_colors, back_colors=faces_DLB.right_colors)
    cube_string = cube.get_cube_string()
    print("cube string:", cube_string)
    sol = cube.get_cube_solution()
    print("solution: ", sol)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')