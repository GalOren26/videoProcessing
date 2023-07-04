import json
import time
import cv2
import numpy as np

def  Tracking(files):
    matted_name = files["input"]["matted_name"]
    binary_name = files["input"]["binary_name"]
    tracked_name = files["output"]["output_name"]
    json_name = files["output"]["json_name"]
    
    binary_vid = cv2.VideoCapture(binary_name)
    matted_name = cv2.VideoCapture(matted_name)
    n_frames = int(binary_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(binary_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(binary_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = binary_vid.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    i = 0
    out1 = cv2.VideoWriter(tracked_name, fourcc, fps, (width, height))
    boundaryOverTime={}
    while binary_vid.isOpened():

        i += 1
        ret1, frame_org = binary_vid.read()
        ret2, frame_mat = matted_name.read()
        if (ret1*ret2 == False):
            break
        frame_g = cv2.cvtColor(frame_org, cv2.COLOR_BGR2GRAY)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(frame_g, connectivity=8)
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Get the centroid coordinates of the largest connected component
        row = int(centroids[largest_component_index][0])
        col = int(centroids[largest_component_index][1])

        width = int(stats[largest_component_index, cv2.CC_STAT_WIDTH]*1.3)
        height = stats[largest_component_index, cv2.CC_STAT_HEIGHT]
        
        image_with_rectangle = cv2.rectangle(frame_mat, (row-(width//2), col-(height//2)), (row+(width//2), col+(height//2)), (0, 255, 0), 2)

        boundaryOverTime[f"{i}"] = [row,col, width, height]
        out1.write(image_with_rectangle)
        boundaryOverTime = {key: [int(val) for val in value] for key, value in boundaryOverTime.items()}

        with open(json_name, 'w') as file:
            # Write the JSON data to the file
            json.dump(boundaryOverTime, file)
    out1.release()
    binary_vid.release()
    matted_name.release()
    cv2.destroyAllWindows()
# tracked_name = 'tracked_name.avi'
# start = time.time()
# Tracking( "matted.avi","binary.avi" , tracked_name)
# end = time.time()
# print(end - start)

