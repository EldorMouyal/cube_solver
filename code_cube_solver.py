#import kociemba
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

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
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def display_canny(image,t=50):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, t, 3*t)
    #cv2.imshow("Edges", edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
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

    lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=87, min_theta=min_angle, max_theta=max_angle)
    # Draw the detected lines on the original image
    draw_lines_from_hough(image, lines[:, 0], thickness=2)
    # Display the image with detected lines
    output_image_path = 'output_image.jpg'
    cv2.imwrite(output_image_path, image)
    print(f"Image with detected lines saved to {output_image_path}")
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
    cv2.imshow(headline, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

def check_threshold_if_load_lines(lines,theta):
    bins_size=[]
    for i in range(0, 400, 20):
        rhos = []
        for line in lines:
            _rho, _theta = line[0]
            x = get_x_by_rho_theta(_rho, _theta, theta)
            if (x >= i and x <= i + 20):
                rhos.append(_rho)
        bins_size.append(len(rhos))
    #for i in range(len(bins_size)):
    #    print("size of bins: ", i," is ", bins_size[i])
    check = False
    for i in range(0,len(bins_size)-1,1):
        if(bins_size[i] > 20):
            check = True
            break
    return check


def fix_threshold_if_load_lines(image,theta,hough_params):
    threshold=hough_params[0]
    min_angle = np.deg2rad(theta - 20)
    max_angle = np.deg2rad(theta + 20)
    edges = display_canny(image, hough_params[1])
    lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=hough_params[0], min_theta=min_angle,max_theta=max_angle)
    iterations=0
    while(check_threshold_if_load_lines(lines,theta)==True and check_by_bins(lines,theta)==0):
        threshold=threshold+1
        lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=threshold, min_theta=min_angle,
                               max_theta=max_angle)
        iterations+=1
    if iterations==0:
        return hough_params
    else:
        return threshold-1,hough_params[1]


def get_hough_params(image,theta):
    params_thres_min = check_threshold_correctness(image,theta,50)
    params_thres_max = check_threshold_correctness(image, theta, 200)
    params_res = 0
    if((params_thres_min[1] > params_thres_max[1]) ):
        params_res=params_thres_max
    elif (params_thres_max[0]==200):
        params_res=int((0.58 * params_thres_min[0] + 0.42 * params_thres_max[0])), params_thres_min[1]
    elif ((params_thres_min[1] < params_thres_max[1])):
        params_res=params_thres_min
    elif(params_thres_min[0]==50):
        params_res=int((0.42 * params_thres_min[0] + 0.58 * params_thres_max[0])), params_thres_max[1]
    else:
        params_res=int((0.55 * params_thres_min[0] + 0.45 * params_thres_max[0])),params_thres_min[1]
    params_if_load_lines=fix_threshold_if_load_lines(image,theta,params_res)
    return int(0.5*params_if_load_lines[0]+0.5*params_res[0]),params_res[1]

def check_threshold_correctness(image,theta,base_threshold):
    original_base_threshold=base_threshold
    edges = display_canny(image)
    min_angle = np.deg2rad(theta - 20)  # Convert to radians
    max_angle = np.deg2rad(theta + 20)
    canny_threshold = 50
    flagForMissing = False
    flagForMany = False
    while(base_threshold > 0):
        if (canny_threshold <=0):
            break
        if (flagForMany and flagForMissing):
            canny_threshold=canny_threshold-5
            edges = display_canny(image,canny_threshold)
            flagForMissing = False
            flagForMany = False
            base_threshold=original_base_threshold
        lines = cv2.HoughLines(edges, rho=2, theta=np.pi / 180, threshold=base_threshold, min_theta=min_angle,max_theta=max_angle)
        if(lines is None):
            base_threshold=base_threshold-1
            continue
        res=check_by_bins(lines,theta)
        if(check_by_bins(lines,theta)==1):
            base_threshold=base_threshold+5
            flagForMany = True
        elif(check_by_bins(lines,theta)==-1):
            base_threshold = base_threshold - 5
            flagForMissing = True
        else:
            return base_threshold,canny_threshold
    print("Please provide me another image")
    return -1

def get_x_by_rho_theta(_rho, _theta,theta):
    if(theta == 0):
        return (_rho - 200 * np.sin(_theta)) / (np.cos(_theta))
    elif(theta == 60):
        return _rho/(np.cos(_theta) + np.sin(_theta))
    else:
        return (_rho-400*np.sin(_theta))/(np.cos(_theta)-np.sin(_theta))

def check_by_bins(lines,theta):
    bins = 0
    for i in range(0, 400, 20):
        rhos=[]
        for line in lines:
            _rho, _theta = line[0]
            x = get_x_by_rho_theta(_rho, _theta, theta)
            if (x >= i and x <= i + 20):
                rhos.append(_rho)
        if (len(rhos) > 0):
            bins = bins + 1
    #print(bins)
    if (bins >= 13):
        # Too much lines
        return 1
    elif (bins <= 6):
        # Missing lines
        return -1
    else:
        return 0

def find_hough_params(edges,min_angle,max_angle): #DOESN'T WORK
    rhos = []
    thetas = []
    base_threshold=170
    pass_thetas=True
    pass_rhos = True
    while(base_threshold >= 50):
        #HoughLine with this threshold
        lines = cv2.HoughLines(edges, rho=1.5, theta=np.pi / 180, threshold=base_threshold, min_theta=min_angle,max_theta=max_angle)
        #Checking rhos
        #Checking thetas
        for line in lines:
            rho, theta = line[0]
            is_close = True
            # Check if rho is distinct from every other rho by more than 3
            for existing_theta in thetas:
                theta_p=abs(theta)
                existing_theta_p=abs(existing_theta)
                if abs(theta_p - existing_theta_p) > 35:
                    is_close = False
                    break
            if is_close:
                thetas.append(theta)
            else:
                pass_thetas=False
                base_threshold =base_threshold-1
                break
    return base_threshold+1


def main(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    cube = 'DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD'
    #print(kociemba.solve(cube))
    image_path = "C:\\Users\\alex\\PycharmProjects\\pythonProject\\.venv\\cube_6.jpg"
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (400, 400))
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=60, headline="Sharp Angle Lines")
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=120, headline="Obtuse Angle Lines")
    # lines = hough_lines_for_theta(image=image, edges=edges, theta=0, headline="Vertical Lines")
    #find_grid_for_theta(resized_image, theta=110)

    edges = display_canny(resized_image)
    lines = hough_lines_for_theta(image=resized_image, edges=edges, theta=60, headline="Vertical Lines")
    print(check_by_bins(lines,60))
    #print(get_hough_params(resized_image,60))
    #params=get_hough_params(resized_image,60)
    #print(check_threshold_if_load_lines(lines, 110))
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
