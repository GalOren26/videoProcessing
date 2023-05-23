import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "206232506"
ID2 = "206454092"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:   
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior
    """ DELETE THE LINE ABOVE AND:
    INSERT YOUR CODE HERE."""
    
    
    state_drifted[0,:] = state_drifted[0,:] + s_prior[4,:] + np.random.normal(0, 3, np.shape(s_prior[0,:]))    
    state_drifted[1, :] = state_drifted[1, :] + s_prior[5, :]+np.random.normal(0, 3, np.shape(s_prior[1,:]))
    state_drifted[4,:]=s_prior[4,:]+np.random.normal(0, 1, np.shape(s_prior[4,:]))
    state_drifted[5,:]=s_prior[5,:]+np.random.normal(0, 1, np.shape(s_prior[5,:]))
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """

    histogram = np.zeros((1, 16 * 16 * 16))
   
    x_start = int(max(state[0] - state[2], 0))
    x_end = int(min(state[0] + state[2], image.shape[1] - 3))
    y_start = int(max(state[1] - state[3], 0))
    y_end = int(min(state[1] + state[3], image.shape[0] - 3))
    T= image[ y_start:y_end,x_start:x_end, :]
    histogram = np.zeros((16, 16, 16))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            r, g, b = T[i, j]
            histogram[int(np.floor(r/16)),int(np.floor(g/16)), int(np.floor(b/16))] += 1
            
    histogram_vector = histogram.reshape(-1, 1)
    normalized_histogram_vector = histogram_vector / np.sum(histogram_vector)
    
    return normalized_histogram_vector
 
def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = np.zeros(previous_state.shape)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""

    for i in range(np.size(previous_state,1)):
        r = np.random.uniform()
        smallest_J = np.size(cdf)-1
        
        for j in range(np.size(cdf)):
            if cdf[j]>=r:
                smallest_J = j
                break
        S_next[:,i] = previous_state[:,smallest_J]

    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    sqrt_of_mult = np.sqrt(p*q)
    sum_of_sqrt =np.sum(sqrt_of_mult)
    distance = np.exp(20*sum_of_sqrt)
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    maxParticle = np.argmax(W)
    #if np.size(maxParticle) > 1: 
    #    maxpParticle = maxParticle[0]

    plt.imshow(image)
    plt.title(ID + " - Frame number = " + str(frame_index))
    W_matrix = np.matlib.repmat(W, 6, 1)
    S_average = np.sum(W_matrix*state, 1)

    # Avg particle box
    (x_avg, y_avg, w_avg, h_avg) = (S_average[0], S_average[1], S_average[2], S_average[3])

    rect = patches.Rectangle((x_avg-w_avg, y_avg-h_avg), w_avg*2, h_avg*2, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max) = (state[0,maxParticle] , state[1,maxParticle],state[2,maxParticle],state[3,maxParticle])
    
    rect = patches.Rectangle((x_max-w_avg, y_max-h_avg), w_max*2, h_max*2, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state



def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    # print(state_at_first_frame)
    S = predict_particles(state_at_first_frame)
    #print(S)
    

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))
    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)
    
    W=np.zeros(np.size(S,1))
    C=np.zeros(np.size(S,1))
    idx=0 
    for n in range (np.size(S,1)):
        p=compute_normalized_histogram(image,S[:,n])
        W[n]= bhattacharyya_distance(p,q)
        
    W=W/np.sum(W)  
    for j in range(np.size(S,1)):
        if (j!=0):
            C[j]=C[j-1]+W[j]
        else:
            C[j]=W[j]
    images_processed = 1
    
    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}

    for image_name in image_name_list[1:]:
        S_prev = S
        
        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""
        W = np.zeros(np.size(S,1))
        C = np.zeros(np.size(S,1))
        for n in range (np.size(S,1)):

            p = compute_normalized_histogram(current_image,S[:,n])
            #print(W)
            W[n]= bhattacharyya_distance(p,q)


        W=W/np.sum(W)  
        
        for j in range(np.size(S,1)):
            if (j!=0):
                C[j]=C[j-1]+W[j]
            else:
                C[j]=W[j]

        # CREATE DETECTOR PLOTS
        images_processed += 1
        print(images_processed)
        show_particles(current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
