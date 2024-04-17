class RubiksCubeFace:
    def __init__(self, colors):
        """
        Initialize a Rubik's Cube face with a 2D array representing the colors of the squares.
        :param colors: A 2D array representing the colors of the squares on the face.
        """
        self.colors = colors

    def __str__(self):
        """
        Convert the Rubik's Cube face to a string representation.
        """
        return "".join(["".join(row) for row in self.colors])