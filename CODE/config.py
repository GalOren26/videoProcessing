
ID1=206232506
ID2=206454092

#### stabilize#####

""" stabilize Variables """

""" variables for goodFeatureTOTrack"""

stabilize={
"files":{
      "input":{
      "input_name" :  "../Input/INPUT.avi",
    },  
      "output":{
        "output_stabilize" : f"../Outputs/stabilize_{ID1}_{ID2}.avi",
    }
},
"variables":{
    "maxCorners":500,
    "qualityLevel":0.005,
    "minDistance":3,
    "blockSize":3
    }
}

bgSub={
    
"files":{
    "input":{
        "input_name" :  f"../Outputs/stabilize_{ID1}_{ID2}.avi",
    },
    "output":{
        "binary_name" : f'../Outputs/binary_{ID1}_{ID2}.avi',
        "extracted_name" : f'../Outputs/extracted_{ID1}_{ID2}.avi'
    }
},
"variables":{
        "percentage_fg_random_frames_seed":10,
        "percentage_bg_random_frames_seed":20}
}

##### matting  #####

matting={
"files":{
     "input":{
            "binary_name" : f"../Outputs/binary_{ID1}_{ID2}.avi",
            "stabilize_name" :f"../Outputs/stabilize_{ID1}_{ID2}.avi",
            "background_name" : "../Input/background.jpg",
    },
         "output":{
            "alpha_name" :  f"../Outputs/alpha_{ID1}_{ID2}.avi",
            "matted_name" : f"../Outputs/matted_{ID1}_{ID2}.avi",
            "trimap_name" : f"../Outputs/trimap_{ID1}_{ID2}.avi",
    }
},
"variables":{
    "r_matting" : 1.2,
    "rho_matting" : 10,
    }
}

""" Tracking Variables """
tracking={
"files":{
      "input":{
      "binary_name" : f"../Outputs/binary_{ID1}_{ID2}.avi",
      "matted_name" : f"../Outputs/matted_{ID1}_{ID2}.avi"
    },  
      "output":{
        "output_name" : f"../Outputs/output_{ID1}_{ID2}.avi",
        "json_name" : "../Outputs/tracking.json"
    }
},
"variables":{
    }
}
