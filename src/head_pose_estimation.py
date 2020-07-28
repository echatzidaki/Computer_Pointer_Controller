import cv2
import sys
import logging as log
import numpy as np
from openvino.inference_engine import IECore, IENetwork

class HeadPoseEstimator:

    def __init__(self, model_name, device='CPU', extensions=None, async_infer=True):
  
        self.net = None
        self.infer_request_handle = None

        self.model_name = model_name
        self.device = device
        
        self.async_infer = async_infer
        
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

        self.w_image = 0
        self.h_image = 0

        try:
            # try:
            self.core=IECore()
            self.model=self.core.read_network(model=self.model_structure, weights=self.model_weights)
            # except AttributeError:
                # self.model=IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception:
            raise ValueError("Could not initialise the network. Have you entered the correct model path?")

        
        self.input_blobs=next(iter(self.model.inputs))
        # Gets the input shape of the network
        self.input_shape=self.model.inputs[self.input_blobs].shape
        self.output_blobs=next(iter(self.model.outputs))
        # Gets the output shape of the network
        self.output_shape=self.model.outputs[self.output_blobs].shape



    def load_model(self):
        '''
        Load the network 
        '''
        num_req=1
        
        self.net=self.core.load_network(network=self.model, device_name=self.device, num_requests=num_req)
    
        
        return

    def predict(self, image):

        request_id = 0
        preprocess_image = self.preprocess_input(image)
        input_dict = {self.input_blobs:preprocess_image}

        if self.async_infer:
            self.infer_request_handle = self.net.start_async(request_id=request_id, inputs=input_dict)
            if self.net.requests[request_id].wait(-1) == 0:
                out = self.infer_request_handle.outputs
                pose_angles = self.preprocess_output(out)
        else:
            self.infer_request_handle = self.net.infer(inputs=input_dict)
            out = self.infer_request_handle
            pose_angles = self.preprocess_output(out)

        return pose_angles


    def preprocess_input(self, image):
        '''
        Change image-data layout from HWC to CHW 
        '''
        h_img = self.input_shape[2]
        w_img = self.input_shape[3]
        
        img = cv2.resize(image, (w_img, h_img))
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, *img.shape)
        
        return img

    def preprocess_output(self, outputs):

        try:
            yaw = outputs["angle_y_fc"][0][0]
            pitch = outputs["angle_p_fc"][0][0]
            roll = outputs["angle_r_fc"][0][0]
            head_pose_angles = np.array([yaw, pitch, roll])
        except Exception:
            log.error("Error in Head Pose Estimation Model - process output function" )
        
        return head_pose_angles

