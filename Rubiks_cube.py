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
#  cube = 'DRLU-U-BFBR BLUR-R-LRUB LRDD-F-DLFU FUFF-D-BRDU BRUF-L-LFDD BFLU-B-LRBD'
#  *our cube = URBF-U-RBDD RUUB-R-FLUU LDDU-F-BULL FRFL-D-DFRF RBBD-L-LLUD DFRF-B-LRBB
#  solved = UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB
#  =>U->R->F->D->L->B
import kociemba


def opposite_color(color):
    color = color.lower()
    opposite_colors = {
        'white': 'yellow',
        'yellow': 'white',
        'orange': 'red',
        'red': 'orange',
        'green': 'blue',
        'blue': 'green'
    }
    return opposite_colors.get(color, None)


class rubiks_cube:
    def __init__(self):
        self.color_values = {
            'white': None,
            'yellow': None,
            'red': None,
            'orange': None,
            'blue': None,
            'green': None}
        self.cube_string = ""

    def set_URF(self, up_colors, right_colors, front_colors):
        self.color_values[up_colors[4]] = 'U'
        self.color_values[opposite_color(up_colors[4])] = 'D'
        self.color_values[right_colors[4]] = 'R'
        self.color_values[opposite_color(right_colors[4])] = 'L'
        self.color_values[front_colors[4]] = 'F'
        self.color_values[opposite_color(front_colors[4])] = 'B'
        for color in up_colors:
            self.cube_string += self.color_values.get(color)
        for color in right_colors:
            self.cube_string += self.color_values.get(color)
        for color in front_colors:
            self.cube_string += self.color_values.get(color)

    def set_DLB(self, down_colors, left_colors, back_colors):
        if self.cube_string == "":
            print("other side needs to be leaded first")
            return None
        # if (self.color_values[down_colors[4]] != 'D' or
        #         self.color_values[left_colors[4]] != 'L' or
        #         self.color_values[back_colors[4]] != 'B'):
        #     print("wrong colors in photo")
        #     return None
        for color in down_colors:
            self.cube_string += self.color_values.get(color)
        for color in left_colors:
            self.cube_string += self.color_values.get(color)
        for color in back_colors:
            self.cube_string += self.color_values.get(color)

    def get_cube_string(self):
        if len(self.cube_string) < 54:
            print("all faces must be loaded first")
            return None
        return self.cube_string

    def get_cube_solution(self):
        if len(self.cube_string) < 54:
            print("all faces must be loaded first")
            return None
        return kociemba.solve(self.cube_string)
