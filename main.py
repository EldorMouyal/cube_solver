import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
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
    top_copy = top_image.copy()
    bottom_image = cv2.resize(bottom_image, (400, 400))
    bottom_copy = bottom_image.copy()
    cube = rubiks_cube()

    top_vertical, top_sharp, top_obtuse = grid_for_image(top_image)
    draw_lines_by_polar(image=top_copy, rho_theta=top_obtuse, color=(255, 0, 0))
    draw_lines_by_polar(image=top_copy, rho_theta=top_vertical, color=(255, 0, 0))
    draw_lines_by_polar(image=top_copy, rho_theta=top_sharp, color=(255, 0, 0))
    display_image(top_copy, "cube")
    bottom_vertical, bottom_sharp, bottom_obtuse = grid_for_image(bottom_image)
    draw_lines_by_polar(image=bottom_copy, rho_theta=bottom_obtuse, color=(255, 0, 0))
    draw_lines_by_polar(image=bottom_copy, rho_theta=bottom_vertical, color=(255, 0, 0))
    draw_lines_by_polar(image=bottom_copy, rho_theta=bottom_sharp, color=(255, 0, 0))
    display_image(bottom_copy, "cube")
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
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # top_path = "../cube_17.jpg"
    # bottom_path = "../cube_18.jpg"
    # top_path = input("Enter first path: ")
    # bottom_path = input("Enter second path: ")
    messagebox.showinfo("Cube solver",
                        "Welcome to Cube Solver, please make sure you have two pictures of the cube before proceeding")
    messagebox.showinfo("Cube solver", "Select image for Up Front Right faces")
    top_path = filedialog.askopenfilename(title="Select top image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    messagebox.showinfo("Cube solver", "Select image for Down Left Back faces")
    bottom_path = filedialog.askopenfilename(title="Select bottom image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if top_path and bottom_path:  # check paths are selected
        cube = _build_cube_by_paths(top_path, bottom_path)
        # Create a message containing both cube string and solution
        try:
            message = f"Cube string: {cube.get_cube_string()}\n\nCube Solution:\n{cube.get_cube_solution()}"
            messagebox.showinfo("Error", message)
            return
        except Exception as e:
            messagebox.showerror("Error", "Please make sure both your pictures are valid")
        return
    else:
        print("No files selected.")
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
