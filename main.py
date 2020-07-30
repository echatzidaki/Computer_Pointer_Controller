#!/usr/bin/env python
import os
import cv2
import sys

import time

import numpy as np
import logging as log

from argparse import ArgumentParser
from src.mouse_controller import MouseController
from src.input_feeder import InputFeeder

from src.graphics import DetectionGraphics

from src.face_detection import FaceDetector
from src.head_pose_estimation import HeadPoseEstimator
from src.facial_landmarks_detection import FacialLandmarksDetector
from src.gaze_estimation import GazeEstimator

# DEF_FD_MPATH = "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
DEF_FD_MPATH = "models/intel/face-detection-adas-0001/FP16/face-detection-adas-0001"
DEF_HPE_MPATH = "models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001"
DEF_FLD_MPATH = "models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009"
DEF_GE_MPATH = "models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002"
# DEF_HPE_MPATH = "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"
# DEF_FLD_MPATH = "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009"
# DEF_GE_MPATH = "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"


DEVICE_TARGETS = ['CPU', 'GPU', 'MYRIAD', 'MULTI:CPU,MYRIAD', 'MULTI:GPU,MYRIAD', 'MULTI:CPU,GPU,MYRIAD', 'MULTI:HDDL,GPU', 'HETERO:MYRIAD,CPU', 'HETERO:GPU,CPU', 'HETERO:FPGA,GPU,CPU', 'HDDL']


DEF_INPUT_STREAM = "bin/demo.mp4"

DEF_OUTPUT_PATH = "bin/"
DEF_STATS_PATH = "bin/stats/stats.txt"


def build_argparser():

    parser = ArgumentParser()
        
    parser.add_argument("-fd", "--facedetectionmodel", required=False, type=str, default=DEF_FD_MPATH,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-fld", "--faciallandmarksdetectionmodel", required=False, type=str, default=DEF_FLD_MPATH,
                        help="Specify Path to .xml file of Facial Landmarks Detection model.")
    parser.add_argument("-hpe", "--headposeestimationmodel", required=False, type=str, default=DEF_HPE_MPATH,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeestimationmodel", required=False, type=str, default=DEF_GE_MPATH,
                        help="Specify Path to .xml file of Gaze Estimation model.")                
                        
                        
    parser.add_argument("-i", "--input", required=False, type=str, default=DEF_INPUT_STREAM,
                        help="Specify Path to video file or enter cam for webcam.")
                        
    # Target Devices                         
    parser.add_argument("-ext", "--extensions", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                            "Absolute path to a shared library with the kernels impl.")
                          
    parser.add_argument("-graphics", "--show_graphics", nargs="+", default=[],
                        help="Specify the models you want to show from fd, fld, hpe, ge, all, stats"
                             "like --show_graphics fd hpe fld (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hpe for Head Pose Estimation, ge for Gaze Estimation,"
                             "nogfor output without graphics"
                             "stats to show inference time.")
                             
    parser.add_argument("-prob", "--prob_threshold", type=float, default=0.5,
                        help="(Optional) Probability threshold for detection filtering (0.5 by default)")
                      
    parser.add_argument("-d", "--device", type=str, default='CPU',
                        help="(Optional) Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Default device is CPU.")
                             
    # Target Devices  for models    
    parser.add_argument('-devfd',  '--device_fd', default='CPU', choices=DEVICE_TARGETS,
                        help="(Optional) Specify the target device to infer on" \
                            "for the Face Detection model (CPU by default).")
    # parser.add_argument('-devfld', default='CPU', choices=DEVICE_TARGETS,
                        # help="(Optional) Specify the target device to infer on" \
                            # "for the Facial Landmark Detection model (CPU by default).")
    # parser.add_argument('-devhpe', default='CPU', choices=DEVICE_TARGETS,
                        # help="(Optional) Specify the target device to infer on" \
                            # "for the Head Pose Estimation model (CPU by default).")
    # parser.add_argument('-devge', default='CPU', choices=DEVICE_TARGETS,
                        # help="(Optional) Specify the target device to infer on" \
                                      # "for the Gaze Estimation model (CPU by default).")
                             
    parser.add_argument('-async', "--async_mode", action='store_true', required=False, default=True,
                        help="(Optional) Perform sync  or async inference..")
    
    parser.add_argument("-o_stats", "--output_stats", type=str, default=DEF_STATS_PATH,  required=False,
                        help="Save performance stats in given path.")
                        
    return parser


def main():

    global out_graphics
    font = cv2.FONT_HERSHEY_COMPLEX
     
    args = build_argparser().parse_args()
    
    log.basicConfig(filename='app.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=log.INFO)

    log.info("Start Application")
    
    mouse_precision = "medium"
    mouse_speed = "fast"

    # Initialize mouse controller
    mouse = MouseController(precision=mouse_precision, speed=mouse_speed)
    out = cv2.VideoWriter("test.mp4", 0x00000021, 30,  (450, 450), True)


    # Function

    stream_graphics = False

    if args.input.lower() == 'cam':
        feeder = InputFeeder(input_type='cam')
        stream_graphics = True
    elif (args.input.lower().endswith('.mp4')) and os.path.isfile(args.input):
        feeder = InputFeeder('video', args.input)
    else:
        log.error("Wrong input path. Try again")
        sys.exit(1)
    
    feeder.load_data()

    pathmodels_dict = {'FD': args.facedetectionmodel,
                                    'FLD': args.faciallandmarksdetectionmodel,
                                    'HPE': args.headposeestimationmodel,
                                    'GE': args.gazeestimationmodel}
              
             
     #Function         
              
    for model in pathmodels_dict.keys():
        if not os.path.isfile(pathmodels_dict[model] + '.xml'):
            log.error("Unable to find specified '" + pathmodels_dict[model].split('/')[-1] + "' model")
            sys.exit(1)

    start_total_load_time = time.time()
    log.info("Models start loading")


    facedetect = FaceDetector(model_name=pathmodels_dict['FD'],
                            prob_threshold=args.prob_threshold,
                            device=args.device,
                            extensions=args.extensions,
                            async_infer=args.async_mode)
                            
    facedetect.load_model()

    faciallandmarksdetect = FacialLandmarksDetector(model_name=pathmodels_dict['FLD'],
                                 # prob_threshold=args.prob_threshold, 
                                 device=args.device,
                                 extensions=args.extensions,
                                 async_infer=args.async_mode)
                                 
    faciallandmarksdetect.load_model()
    
    headposeestimate = HeadPoseEstimator(model_name=pathmodels_dict['HPE'],
                                 # prob_threshold=args.prob_threshold, 
                                 device=args.device,
                                 extensions=args.extensions,
                                 async_infer=args.async_mode)
                                 
    headposeestimate.load_model()
    
    gazeestimate = GazeEstimator(model_name=pathmodels_dict['GE'],
                             # prob_threshold=args.prob_threshold, 
                             device=args.device,
                             extensions=args.extensions,
                             async_infer=args.async_mode)

    
    gazeestimate.load_model()

    total_models_load_time = time.time() - start_total_load_time
    log.info("All models loaded.")

    if len(args.show_graphics) != 0 :
        for option in args.show_graphics:
            if not option in ['fld', 'hpe', 'fd', 'ge', 'nog', 'stats']:
                log.error("Invalid flag or oprtions are not separated with spaces. Try again.")
                sys.exit(1)

    # log save out file

    counter = 0
    start_total_inference_time = time.time()
    # log start inference
    infer_time_ge = 0.0
    infer_time_fd = 0.0
    infer_time_fld = 0.0
    infer_time_hpe = 0.0
    

    try:
        # add ret
        for frame in feeder.next_batch():
            if frame is None :
                break
            key_pressed = cv2.waitKey(60)

            counter += 1
            
            start_infer_time = time.time()

            if len(args.show_graphics) != 0:
                out_graphics = DetectionGraphics(frame, stream_graphics)
                log.info("Initiate Graphics Class")
                
            start_infer_time_fd = time.time()    
            copy_frame = frame.copy()
            face_coords, crop_face = facedetect.predict(copy_frame)
            infer_time_fd = infer_time_fd + time.time()  - start_infer_time_fd
            
            # Nobody detected
            if len(face_coords) == 0:
                log.error("Nobody detected!")
            else:
                log.info("A person detected")
                if 'fd' in args.show_graphics:
                    out_graphics.face_detection(face_coords)
                
                cropped_face_copy = crop_face.copy()
                
                start_infer_time_fld = time.time()    
                eyes_coord, crop_left, crop_right = faciallandmarksdetect.predict(cropped_face_copy)
                infer_time_fld = infer_time_fld + time.time()  - start_infer_time_fld
                
                if len(eyes_coord) != 0 and 'fld' in args.show_graphics:
                    out_graphics.eyes_detection(eyes_coord, face_coords, crop_face.shape[1])

                crop_face_copy = crop_face.copy()

                start_infer_time_hpe = time.time()    
                head_pose_angles = headposeestimate.predict(crop_face_copy)
                infer_time_hpe = infer_time_hpe +  time.time()  - start_infer_time_hpe
                
                if len(head_pose_angles) != 0 and 'hpe' in args.show_graphics:
                    out_graphics.head_pose_estimation(head_pose_angles,face_coords,crop_face)

                if len(crop_left) != 0 and len(crop_right) != 0:
                    head_pose_angles_copy = head_pose_angles.copy()
                    crop_right_copy = crop_right.copy()
                    crop_left_copy = crop_left.copy()
                    
                    start_infer_time_ge = time.time() 
                    gaze = gazeestimate.predict(crop_left_copy,crop_right_copy, head_pose_angles_copy )
                    infer_time_ge = infer_time_ge + time.time()  - start_infer_time_ge
                    
                    if len(gaze) != 0 and 'ge' in args.show_graphics:
                        out_graphics.eyes_gaze_estimation(eyes_coord, face_coords, gaze, crop_face )

                    # Move the mouse n screen
                    if len(gaze) != 0:
                        # Camera
                        if stream_graphics: 
                            mouse.move(-gaze[0], gaze[1])
                        else:
                            mouse.move(gaze[0], gaze[1])

            inference_time = time.time() - start_infer_time

            # out.write(out_graphics)
            if len(args.show_graphics) != 0:
                frame = cv2.resize(frame, (450, 450))
                color_text = (150, 0, 255)
                if stream_graphics:
                    frame = cv2.stream_graphics(frame, 1)
                if len(face_coords) == 0:
                    text = "Nobody detected"
                    cv2.putText(frame, text, (20, 40), font, 0.45,color_text, 2)
                    log.error(text)

                cv2.imshow("out_graphics", frame)
                cv2.moveWindow("out_graphics", 70, 70)

            if key_pressed == 27:
                break

        total_inference_time = time.time() - start_total_inference_time
        fps = counter / total_inference_time

        log.warning("VideoStream ended...")
        cv2.destroyAllWindows()
        feeder.close()

        # Save  stats
        if args.output_stats:
            dir_path = args.output_stats.rsplit("/", 1)[0]
            if os.path.exists(dir_path) is not True:
                os.makedirs(dir_path)
            with open(args.output_stats, 'w') as f:
                f.write(str("models total loading time (s): ") + str(total_models_load_time) + '\n')
                f.write(str("FD inference time per frame (s): ") + str(infer_time_fd) + '\n')
                f.write(str("FLD inference time  per frame (s): ") + str(infer_time_fld) + '\n')
                f.write(str("HPE inference time per frame (s): ") + str(infer_time_hpe) + '\n')
                f.write(str("GE inference time per frame (s): ") + str(infer_time_ge) + '\n')
                f.write(str("Total inference time (s): ") + str(total_inference_time) + '\n')
                f.write(str("Frames: ") + str(counter) + '\n')
                f.write(str("FPS: ") + str(fps) + '\n')

    except KeyboardInterrupt:
        log.error("Application interrupted by user.")


if __name__ == '__main__':
    main()
