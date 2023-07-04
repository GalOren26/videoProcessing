import json
import cv2
import logging
import time

#modulus 
from Stabilize import *
from matting import *
from Tracking import *
import config as Config
from backgound_substraction import *
#todo create jsons
#todo save as gray 


timing={"time_to_stabilize":0
        ,"time_to_binary":0
        ,"time_to_alpha":0
        ,"time_to_matted":0
        ,"time_to_output":0}

def main():
    

    # stabilization 
 
    logging.warning('Started the Stabilization Process')
    InputVidName = Config.stabilize["variables"]
    start = time.time()
    stabilize_vid(Config.stabilize["files"],Config.stabilize["variables"])
    end = time.time()
    logging.warning('Done the Stabilization Process')
    logging.warning("\t \t Total stabilization Run Time: %f sec which equal %f min", end - start, (end - start) / 60)
    timing["time_to_stabilize"]= end - start


    # background substraction 

    logging.warning('Started the Background Subtraction Process')
    start = time.time()
    background_sub(Config.bgSub["files"],Config.bgSub["variables"])
    end = time.time()
    logging.warning('Done the Background Subtraction Process')
    logging.warning("\t \t Total Background Subtraction Run Time: %f sec which equal %f min", end - start,
                    (end - start) / 60)
    timing["time_to_binary"]= end - start

    # matting+ alpha 
    
    logging.warning('Started the Matting Process')
    start = time.time()
    matting(Config.matting["files"],Config.matting["variables"])
    end = time.time()
    logging.warning('Done the Matting Process')
    logging.warning("\t \t Total Alpha and Matting Run Time: %f sec which equal %f min", end - start,
                    (end - start) / 60)
    timing["time_to_alpha"]= end - start
    timing["time_to_matted"]= end - start
  

  

  #tracking 
    logging.warning('Started the tracking Process')
    start = time.time()
    start = time.time()
    Tracking(Config.tracking["files"] )
    end = time.time()
    logging.warning('Done the tracking Process')
    logging.warning("\t \t Total tracking Run Time: %f sec which equal %f min", end - start,
                    (end - start) / 60)
    timing["time_to_output"]= end - start
    
    
    # save timing to json
    filename = "../Outputs/timing.json"
    with open(filename, 'w') as file:
    # Write the JSON data to the file
       json.dump(timing, file)

main()