import numpy as np
import cv2



def draw_precent_image_stabilize(frame, n_frame):
    image = np.ones((100, 600)) / 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'frame ' + str(frame) + ' of ' + str(n_frame) + ' -- ' + str(int((frame / n_frame) * 100)) + '% Done'
    cv2.putText(image, text, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    # Display the resulting frame
    cv2.imshow('Stabilize percent', image)
    cv2.waitKey(2)



def fixHomographyBorders(frame,input_frame,H):
    height ,width=frame.shape[0],frame.shape[1]
    # Define the corners of the image and transform them using the homography matrix
    corners = np.array([[1, 1], [width-1, 1], [width-1, height-1], [1, height-1]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
    mask = np.zeros_like(frame, dtype=np.uint8)
    # create mask for the transformed image acoording to the transformed corners
    cv2.fillConvexPoly(mask, transformed_corners.astype(np.int32), (255,255,255))
    mask =mask/255
    output=(mask*frame+input_frame*(1-mask)).astype(np.uint8)

    return output



def stabilize_vid(files_names, variables):
    #parse params 
    maxCorners=variables["maxCorners"]
    qualityLevel=variables["qualityLevel"]
    minDistance=variables["minDistance"]
    blockSize=variables["blockSize"]
    vid_name=files_names[ "input"]["input_name"]
    output_name=files_names["output"]["output_stabilize"]
    
    # Read the video
    cap = cv2.VideoCapture(vid_name)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Output fourcc and Video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
    # read the first frame convert it to gray scale
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(30,30), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS |cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    # Read the first frame
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # # Find feature points in the first frame
  

    i=0
    while True:
        draw_precent_image_stabilize(i + 1, n_frames)
        i+=1
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=maxCorners, qualityLevel=qualityLevel,
                                           minDistance=minDistance, blockSize=blockSize)
        # Calculate optical flow
        prev_points, status, _ = cv2.calcOpticalFlowPyrLK( gray , prev_gray, points, None, **lk_params)
        
        # choose only the points that exist in the current frame
        idx = np.where(status == 1)[0]
        prev_points = prev_points[idx]
        points = points[idx]
        assert points.shape == prev_points.shape
         # Calculate Homography using RANSAC
        M, status = cv2.findHomography(points, prev_points, cv2.RANSAC,2.0)

        # # Warp source image to destination based on homography
        frame= cv2.warpPerspective(frame, M,(width, height))
        frame_fixed=fixHomographyBorders(frame,prev_frame,M)

        out.write(frame_fixed)
    
        # Update previous frame and points
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_points = points.reshape(-1, 1, 2)
        # becouse of change in brightness we want to update the prev frame (that we project to its plan )every 1/3 of the video
        if(i%(n_frames//3)==0):     
            prev_frame=frame_fixed

    # Release video capture and writer
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()
        
# stabilize_vid('../Input/INPUT.avi', "../Outputs/stabilize.avi")
    
    
    