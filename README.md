# Cube Solver

## Description

The Cube Solver program utilizes Kociemba and OpenCV libraries to swiftly provide a solution for a Rubik's Cube.

## Usage Instructions

1. **Capture Images**: Take two pictures of your Rubik's Cube, ensuring that in each image, three faces of the cube are clearly visible. For optimal results, ensure that your cube has a black base and that the background is a single color. You may take pictures on a blank paper page and crop the images to display only the cube.

2. **Run the Program**: Execute the program from the `Cube_Solver.exe` file. You will be prompted to select the two photos you captured.

3. **View Solution**: The solution will be printed to your console.

## Program Functionality

The program employs various image processing techniques to analyze the cube images:

1. **Canny Edge Detection**: The initial step involves applying the Canny edge detection algorithm to the images.

2. **Hough Transform**: The program utilizes the Hough transform to detect lines in the images, specifically looking for lines with specific angles associated with cube faces.

3. **Thresholding**: Filtering methods are employed to determine appropriate threshold values for both the Canny edge detection and Hough transform.

4. **Line Filtering**: The detected lines are further filtered to retain only relevant lines associated with the cube's edges.

5. **Intersection Calculation**: With seven lines for each angle, the program calculates the intersections between lines to determine the exact area for each cubicle of the cube.

6. **Color Extraction**: Once the cubicle areas are determined, the program extracts the most dominant colors from common Rubik's cube colors within each area.

7. **Face Mapping**: The extracted colors are used to create three cube-face-like arrays containing the colors of each face. This process is repeated for both images, resulting in all six faces of the cube being mapped to arrays.

8. **Solution Generation**: Using the mapped cube faces, the program creates the string format expected by Kociemba and prints the solution to the console.

## Dependencies

- Python 3.x
- OpenCV
- Kociemba

## Notes
- Ensure your images contain only the cube with no background shades or noise
- Ensure that your images are clear and well-lit for optimal results.
- Experiment with different backgrounds and lighting conditions to improve the accuracy of color extraction.

## Example Images

Examples of properly captured images can be found in the provided `pictures.zip` file.

## Contributing

- Adding shade and noise robustness to the Canny filter phase.
- Providing more accurate color detection algorithm

## License
- This project is private. you may contact me for any explanations and further discussion.
