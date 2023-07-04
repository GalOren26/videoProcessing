import time
import numpy as np
import cv2
from scipy.stats import gaussian_kde

import scipy.ndimage as nd
import itertools as it
import numba
def draw_percent_image_Matting_1(frame, n_frame):
    image = np.ones((100, 600)) / 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'frame ' + str(frame) + ' of ' + str(n_frame) + ' -- ' + str(int((frame / n_frame) * 100)) + '% Done'
    cv2.putText(image, text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow('Matting percent', image)
    cv2.waitKey(2)

    

def probability_map_calc_1(frame, background_mask, foreground_mask):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_chanel_density_fg, v_chanel_density_bg = pdf_estimation_1(hsv_frame, foreground_mask, background_mask)

    prob_given_fg = v_chanel_density_fg[hsv_frame[:, :, 1]]
    prob_given_bg = v_chanel_density_bg[hsv_frame[:, :, 1]]

    prob_fg = prob_given_fg / (prob_given_fg + prob_given_bg +  np.finfo(float).eps)
    prob_bg = 1 - prob_fg
    
    grad_bg=nd.gaussian_gradient_magnitude(prob_bg, 1.5)
    grad_fg=nd.gaussian_gradient_magnitude(prob_fg, 1.5)
    # grad_bg = cv2.Sobel(prob_bg, cv2.CV_64F, 1, 1, ksize=5)
    # grad_fg = cv2.Sobel(prob_fg, cv2.CV_64F, 1, 1, ksize=5)

    return grad_bg, grad_fg, prob_fg, prob_bg


def pdf_estimation_1(image, mask_fg, mask_bg):
    image = cv2.resize(image, (image.shape[0] // 4, image.shape[1] // 4))
    mask_fg = cv2.resize(mask_fg, (mask_fg.shape[0] // 4, mask_fg.shape[1] // 4))
    mask_bg = cv2.resize(mask_bg, (mask_bg.shape[0] // 4, mask_bg.shape[1] // 4))

    # limit the size of the rectangular around the center
    y_max =np.round(np.max(np.where(mask_fg==255)[0])*1.4).astype(np.uint32)
    y_min =np.min(np.where(mask_fg==255)[0])
    x_max =np.round(np.max(np.where(mask_fg==255)[1])*1.4).astype(np.uint32)
    x_min=np.min(np.where(mask_fg==255)[1])
    
    y_max=np.min([y_max,mask_fg.shape[0]-1])
    y_min=np.max([y_min,0])
    x_max=np.min([x_max,mask_fg.shape[1]-1])
    x_min=np.max([x_min,0])
    
  
    # crop the rectangular around the center
    
    image_crop = image[y_min:y_max, x_min:x_max,:]
    mask_crop_fg = mask_fg[y_min:y_max, x_min:x_max]
    mask_crop_bg = mask_bg[y_min:y_max, x_min:x_max]

    # calculate the kde of bg and fg
    color_level_grid = np.linspace(0, 255, 256)
    h, s, v = cv2.split(image_crop)
    rows, cols = np.where(mask_crop_fg == 255)
    info = s[rows, cols]
    pdf_fg = kde_scipy_1(info, color_level_grid)

    rows, cols = np.where(mask_crop_bg == 255)
    info = s[rows, cols]
    pdf_bg = kde_scipy_1(info, color_level_grid)
    return pdf_fg, pdf_bg


def GDT(fg_mask, grad_fg, max_iter_n=80, max_diff=0.1):
    C = 1.0+grad_fg*200
    A = np.zeros_like(C)
    A[:] = 1e6  
    A[fg_mask == 255] = 0  
    sweeps = [A, A[:,::-1], A[::-1], A[::-1,::-1]]
    costs = [C, C[:,::-1], C[::-1], C[::-1,::-1]]
    for i, (a, c) in enumerate(it.cycle(zip(sweeps, costs))):
        r = sweep(a, c)
        if r < max_diff or i >= max_iter_n:
            break
    return A

@numba.jit
def sweep(A, Cost):
    max_diff = 0.0
    for i in range(1, A.shape[0]):
        for j in range(1, A.shape[1]):
            t1, t2 = A[i, j-1], A[i-1, j]
            C = Cost[i, j]
            if abs(t1-t2) > C:
                t0 = min(t1, t2) + C  # handle degenerate case
            else:    
                t0 = 0.5*(t1 + t2 + np.sqrt(2*C*C - (t1-t2)**2))
            max_diff = max(max_diff, A[i, j] - t0)
            A[i, j] = min(A[i, j], t0)
    return max_diff






def kde_scipy_1(x, x_grid, bandwidth=0.2):
    """Kernel Density Estimation with Scipy"""
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1))
    return kde.evaluate(x_grid)


def matting(files,variables):
    """    main Function    """
    background_name = files["input"]["background_name"]
    stable_name = files[ "input"]["stabilize_name"]
    binary_name =files[ "input"]["binary_name"]
    alpha_name = files[ "output"]["alpha_name"]
    trimap_name = files[ "output"]["trimap_name"]
    matted_name = files[ "output"]["matted_name"]
    r = variables["r_matting"]
    rho = variables["rho_matting"]

    # set the parameters for the input and outputs videos
    image_vid = cv2.VideoCapture(stable_name)
    binary_vid = cv2.VideoCapture(binary_name)
    bg = cv2.imread(background_name)

    n_frames = int(image_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(image_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(image_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = image_vid.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out1 = cv2.VideoWriter(alpha_name, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(trimap_name, fourcc, fps, (width, height))
    out3 = cv2.VideoWriter(matted_name, fourcc, fps, (width, height))

 

    i_f = 0
    while image_vid.isOpened():
        draw_percent_image_Matting_1(i_f, n_frames)
        i_f += 1

        # read one frame
        ret1, img_frame = image_vid.read()
        ret2, binary_frame = binary_vid.read()
        
        if (ret1==0 or  ret2 == 0):
            break

        binary_gray = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
        th,binary_gray = cv2.threshold(binary_gray,127, 255, cv2.THRESH_BINARY)
        fg_mask = np.zeros_like(binary_gray)
        bg_mask = np.zeros_like(binary_gray)
        trimap = np.zeros_like(binary_gray)
        alpha = np.zeros_like(binary_gray)
        binary_gray = cv2.GaussianBlur(binary_gray, (11,11), 0)
        # calculate the fg and the bg based on the binary image
        fg_mask[binary_gray >= 240] = 255
        bg_mask[binary_gray <= 10] = 255
        # calculate the initial probability map and the geodesic distance
        grad_bg, grad_fg, prob_fg, prob_bg = probability_map_calc_1(img_frame, bg_mask, fg_mask)

        fg_gdf=GDT(fg_mask, grad_fg)
        bg_gdf=GDT(bg_mask, grad_bg)
        

        trimap[(fg_gdf - bg_gdf) > rho] = 0
        trimap[(bg_gdf - fg_gdf) > rho] = 255
        trimap[abs(bg_gdf - fg_gdf) <= rho] = 0.5 * 256

        # calculate the weight based on the probability map and the geodesic distance
        W_fg = (fg_gdf + np.finfo(float).eps) ** (-r) * prob_fg
        W_bg = (bg_gdf + np.finfo(float).eps) ** (-r) * prob_bg

        # calc alpha map
        alpha = W_fg / (W_fg + W_bg)
        alpha[trimap == 255] = 1
        alpha[trimap == 0] = 0

        # Wrap the new background image to the frame
        frame = np.zeros_like(img_frame)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_frame_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        frame_hsv[:, :, 0] = alpha * img_frame_hsv[:, :, 0] + (1 - alpha) * bg_hsv[:, :, 0]
        frame_hsv[:, :, 1] = alpha * img_frame_hsv[:, :, 1] + (1 - alpha) * bg_hsv[:, :, 1]
        frame_hsv[:, :, 2] = alpha * img_frame_hsv[:, :, 2] + (1 - alpha) * bg_hsv[:, :, 2]

        # Save the alpha, trimap and matted frame
        alpha = (alpha * 255).astype(np.uint8)

        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        trimap = cv2.cvtColor(trimap, cv2.COLOR_GRAY2BGR)
        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

        out1.write(np.uint8(alpha))
        out2.write(np.uint8(trimap))
        out3.write(np.uint8(frame))

    # release all Videos
    out1.release()
    out2.release()
    out3.release()
    image_vid.release()
    binary_vid.release()
    cv2.destroyAllWindows()

    
# """ matting Variables """
# r_matting = 1.2
# rho_matting = 10
# object_half_size_in_y = 600
# object_half_size_in_x = 200
# selected_matting = 0
# background_name = "../Input/background.jpg"
# matting(background_name, "../CODE/stabilize.avi", "../CODE/binary.avi", "../Outputs/alpha.avi",
#                        "../Outputs/trimap.avi", "../Outputs/matted.avi", r=r_matting, rho=rho_matting,
#                        y_width=object_half_size_in_y, x_width=object_half_size_in_x)

