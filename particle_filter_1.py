import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "123456789"
ID2 = "987654321"

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
    
    state_drifted[:,0] = state_drifted[:,0] + s_prior[:,4]
    state_drifted[:, 1] = state_drifted[:, 1] + s_prior[:, 5]
    vector = [0,1,4,5]
    for i in vector:
        noise = np.round(np.random.normal(0, 1, np.shape(s_prior[:,i]))).astype('int32')
        state_drifted[:,i] = np.add(state_drifted[:,i], noise)
    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    '''
    S=state
    I=image
    histogram = np.zeros((16,16,16))
    for i in range(int(S[0]-S[2]),int(S[0]+S[2])):
        for j in range(int(S[1]-S[3]),int(S[1]+S[3])):
            j = abs(j)
            i = abs(i)
            if j>=np.size(I,0):
                j = 2*np.size(I,0)-j-1
            if i>=np.size(I,1):
                i = 2*np.size(I,1)-i-1
            print(np.size(I,0),np.size(I,1))
            print(j)
            print(i)
            print(I[j,i,0]/16)
            histogram[int(np.floor(I[j,i,0]/16)),int(np.floor(I[j,i,1]/16)),int(np.floor(I[j,i,2]/16))]+=1
    histogram = np.reshape(histogram, (4096,1))
    histogram = histogram/(4*S[2]*S[3])
    return histogram
    '''
    #state = np.floor(state)
    #state = state.astype(int)
    
    histogram = np.zeros((1, 16 * 16 * 16))
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    #x_start=int(state[0]-state[2])
    #x_end=int(state[0]+state[2])
   
    print(state)
    #y_start=int(state[1]-state[3])
    #y_end=int(state[1]+state[3])
    x_start = int(max(state[0] - state[2], 0))
    x_end = int(min(state[0] + state[2], image.shape[1] - 3))
    y_start = int(max(state[1] - state[3], 0))
    y_end = int(min(state[1] + state[3], image.shape[0] - 3))

    
    #T= np.zeros((int( x_end -x_start ),int(y_start - y_end),3))
    print(x_start,x_end, y_start,y_end,'x_start,x_end, y_start,y_end')
    #print(image.shape)
    T= image[ y_start:y_end,x_start:x_end, :]
    print(T.shape, 'size')
    
    # Compute the histogram for the combinations of RGB values
    histogram = np.zeros((16, 16, 16))
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            r, g, b = T[i, j]
            histogram[int(np.floor(r/16)),int(np.floor(g/16)), int(np.floor(b/16))] += 1

    # Reshape the histogram vector into a shape of (4096, 1)
    histogram_vector = histogram.reshape(-1, 1)

    # Normalize the histogram vector
    normalized_histogram_vector = histogram_vector / np.sum(histogram_vector)

    # Print the normalized histogram vector
    
    return normalized_histogram_vector
 
'''
   histogram, _ = np.histogramdd(T.reshape(-1, 3), bins=16)
    #print(histogram)

    histogram = np.reshape(histogram, (4096, 1))
    # normalize
    print(histogram)
    print(np.sum(histogram))

    histogram = histogram /sum(histogram)
'''
    


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
            if cdf[j]>r:
                smallest_J = j
                break
        S_next[:,i] = previous_state[:,smallest_J]
        print(S_next[:,i].shape,'S_next[:,i]',S_next.shape)

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
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    '''
    I = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(W)
    maxParticle = np.argmax(W)
    if np.size(maxParticle) > 1:
        max_particle = maxParticle[0]
    first_border_max= int(state[maxParticle,0]-state[maxParticle,2]),int(state[maxParticle,1]-state[maxParticle,3])
    second_border_max= int(state[maxParticle,0]+state[maxParticle,2]),int(state[maxParticle,1]+state[maxParticle,3])
    #draw red rectangle
    I = cv2.rectangle(I,first_border_max,second_border_max,(255,0,0),2)
    #print(W)
    W_matrix = np.matlib.repmat(W, 100, 1).transpose()
    #print(W_matrix)
    print(state)
    S_average = np.sum(W_matrix*state, 0)
    print(S_average)
    first_border_average = int(S_average[0]-S_average[2]),int(S_average[1]-S_average[3])
    second_border_average = int(S_average[0]+S_average[2]),int(S_average[1]+S_average[3])
    #draw green rectangle
    I = cv2.rectangle(I, first_border_average , second_border_average, (0,255,0),2)
    cv2.imwrite("/content/gdrive/Shared drives/VideoProcessing/Project/Sime-final/Tracking/track"+str(frame_index)+".jpg", I) 
    plot_name = ID+'-'+str(frame_index)+'.png'
    fig = plt.figure()
    plt.savefig(plot_name)
    plt.imshow(I)
    plt.title(ID+'- Frame number = '+str(frame_index))
    plt.show(block=False)

    '''
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    maxParticle = np.argmax(W)
    #if np.size(maxParticle) > 1:
    #    maxpParticle = maxParticle[0]

    #plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))
    #print(W.shape)
    W_matrix = np.matlib.repmat(W, 6, 1)
    print(W_matrix.shape,'W_matrix')
    print(state.shape,'state')
    S_average = np.sum(W_matrix*state, 1)

    # Avg particle box
    (x_avg, y_avg, w_avg, h_avg) = (S_average[0], S_average[1], S_average[2], S_average[3])
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""


    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max) = (state[0,maxParticle] , state[1,maxParticle],state[2,maxParticle],state[3,maxParticle])
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""

    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    print(state_at_first_frame)
    S = predict_particles(state_at_first_frame)
    #print(S)
    

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)
    
    print('done')
    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""
    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
# YOU NEED TO FILL THIS PART WITH CODE:
# ........
    print(np.size(S,1),'np.size(S,1)')
    W=np.zeros(np.size(S,1))
    C=np.zeros(np.size(S,1))
    idx=0 
    for n in range (np.size(S,1)):
        idx+=1
        print(idx,'idx')
        print(image.shape)
        print(S[:,n].shape)

        p=compute_normalized_histogram(image,S[:,n])

        print(p,'p')
        print(p)
        W[n]= bhattacharyya_distance(p,q)
        print(W)
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
    #q = np.matlib.repmat(q, N, 1).T
    for image_name in image_name_list[1:]:
        S_average = np.sum(W*S, 1)
        q = compute_normalized_histogram(image, S_average)

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

            #q[n] = p[n]
            #print(W)
        W=W/np.sum(W)  
        #print(W)
        #print(np.sum(W))
        for j in range(np.size(S,0)):
            if (j!=0):
                C[j]=C[j-1]+W[j]
            else:
                C[j]=W[j]

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if images_processed ==5:
            break
        #if 0 == images_processed%10:
        #frame_index_to_avg_state, frame_index_to_max_state = 
        show_particles(current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
