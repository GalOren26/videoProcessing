import cv2
import numpy as np
from  scipy.ndimage import gaussian_filter



def medianBackgroundFrame(cap,n_frames,percentage_bg_random_frames_seed):
    # Choose randomly 50 frames and Calculate the median in time
    num_frames=int(percentage_bg_random_frames_seed*n_frames/100)
    frameIds = n_frames * np.random.uniform(size=num_frames)
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        _, frame = cap.read()
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frames.append(frame)
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    return medianFrame

def draw_starting_image():
    image = np.ones((100, 600)) / 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Starting...'
    cv2.putText(image, text, (150, 70), font, 2, (0, 255, 255), 2, cv2.LINE_4)
    # Display the resulting frame
    cv2.imshow('Binary percent', image)
    cv2.waitKey(200)

def draw_percent_image_binary(frame, n_frame):
    image = np.ones((100, 600)) / 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'frame ' + str(frame) + ' of ' + str(n_frame) + ' -- ' + str(int((frame / n_frame) * 100)) + '% Done'
    cv2.putText(image, text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    # Display the resulting frame
    cv2.imshow('Binary percent', image)
    cv2.waitKey(2)
    
    


def build_hist(image,mask=np.array([])):
    if mask.size!=0:
        image = image[mask==255]
    # Build 64 bin histogram (divide the image values by 4)
    histogram = np.zeros((64, 64, 64))
    image = image // 4

    # Flatten the frame array to simplify indexing
    flattened_frame = image.reshape(-1, 3)
    unique_values, counts = np.unique(flattened_frame, axis=0, return_counts=True)

    # Update histogram using vectorized indexing
    histogram[unique_values[:, 0], unique_values[:, 1], unique_values[:, 2]] = counts
    histogram=gaussian_filter(histogram,sigma=1 ,radius=3, order=0)
    return histogram

def foreground(cap,background_gray,n_frames,percentage_fg_random_frames_seed): 
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_jump=n_frames//percentage_fg_random_frames_seed
    histograms =[]
    for fid in range(0,n_frames,frame_jump):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            # Read the next frame
            ret, frame = cap.read()
            # Convert the frame to grayscale
            curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Compute the absolute difference between the current frame and the previous frame
            frame_diff = cv2.absdiff(curr_frame_gray, background_gray)
            _, mask = cv2.threshold(frame_diff,0, 255, cv2.THRESH_BINARY|+ cv2.THRESH_OTSU)
            #remove noise and close gaps
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            _, mask = cv2.threshold(mask,220, 255, cv2.THRESH_BINARY)
            ####   process frame    ####
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # remove noise - opening morphology
            mask = cv2.erode(mask, kernel, iterations=1)            
            mask = cv2.dilate(mask, kernel, iterations=2)
            ####   histogram for frame  ########
            rows,cols=np.where(mask==255)
            frame_masked=np.zeros(frame.shape)
            frame_masked[rows,cols,:]=frame[rows,cols,:]
            frame_masked=frame_masked.astype(np.uint8)
            #for efficency 
            frame_masked = cv2.resize(frame_masked, (0, 0), fx=0.5, fy=0.5)
            mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
            
            histogram_fg = build_hist(frame_masked,mask)   
            histograms.append(histogram_fg)
  
    #######################################################  BEST HISTOGRAM  ################################################################
    # Get the shape of the histograms
    hist_shape = histograms[0].shape
    
    # Initialize the final histogram
    final_histogram = np.zeros(hist_shape, dtype=np.uint8)

    # Iterate over each bin/index
    final_histogram=np.amax(histograms, axis=0)

    cv2.destroyAllWindows()
    return final_histogram 

def background_sub(files,variables):

    """Main function"""
    draw_starting_image()

    # define the input and the output Videos
    input_name=files["input"]["input_name"]
    binary_name=files["output"]["binary_name"]
    extracted_name=files["output"]["extracted_name"]
    percentage_fg_random_frames_seed=variables["percentage_fg_random_frames_seed"]
    percentage_bg_random_frames_seed=variables["percentage_bg_random_frames_seed"]
       
    cap = cv2.VideoCapture(input_name)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(binary_name, fourcc, fps, (width, height))
    out2 = cv2.VideoWriter(extracted_name, fourcc, fps, (width, height))
    medianFrame =  medianBackgroundFrame(cap,n_frames,percentage_bg_random_frames_seed)
    background_gray= cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    histogram_foreground = foreground(cap,background_gray,n_frames,percentage_fg_random_frames_seed)
    histogram_background=  build_hist(medianFrame)

    normalized_histogram_background = histogram_background / np.sum(histogram_background)
    normalized_histogram_foreground = histogram_foreground/np.sum(histogram_foreground)


    #probability maps
    fg_prop = np.zeros((height // 2, width // 2))
    bg_prop = np.zeros((height // 2, width // 2))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
    idx = 0
    
    while cap.isOpened():
        draw_percent_image_binary(idx, n_frames)

        idx += 1
        ret, frame_org = cap.read()
        if (ret == False):
            break

        # divide by 4 due to 64 Bin histogram
        frame = frame_org // 4  # 64 Bin
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        

        # calculate the probability map
        for i in range(0, height // 2):
            for j in range(0, width // 2):
                fg_prop[i, j] = normalized_histogram_foreground[frame[i, j, 0], frame[i, j, 1], frame[i, j, 2]]
                bg_prop[i, j] = normalized_histogram_background[frame[i, j, 0], frame[i, j, 1], frame[i, j, 2]]

        fg = (fg_prop + np.finfo(float).eps) / (fg_prop + bg_prop + np.finfo(float).eps)
        th, fg = cv2.threshold(fg, 0.95, 255, cv2.THRESH_BINARY)
        fg = fg.astype(np.uint8)

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
                                  key=lambda x: x[1])
        fg_label = np.zeros((height // 2, width // 2))
        fg_label[output == max_label] = 1
        fg = fg_label*255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.dilate(fg, kernel, iterations=4)
        fg = cv2.erode(fg, kernel, iterations=2)
   

        fg = fg.astype(np.uint8)
        #return to original size
        fg = cv2.resize(fg, (0, 0), fx=2, fy=2)
        #save binary and extracted
        th,fg = cv2.threshold(fg,127, 255, cv2.THRESH_BINARY)
        binary = cv2.cvtColor(fg, cv2.COLOR_GRAY2RGB)
        # extracted the object from the frame
        extract = np.zeros_like(frame_org)
        fg[fg==255]=1
        extract[:, :, 0] = fg * frame_org[:, :, 0]
        extract[:, :, 1] = fg * frame_org[:, :, 1]
        extract[:, :, 2] = fg * frame_org[:, :, 2]
        fg[fg==1]=255

        out.write(binary)
        out2.write(extract)

    out.release()
    out2.release()
    cap.release()
    cv2.destroyAllWindows()

# input_name ='stabilize.avi'
# binary_name = 'binary.avi'
# extracted_name = 'extracted.avi')

