import cv2
import sys
import logging as log
import numpy as np
from openvino.inference_engine import IECore, IENetwork


class GazeEstimator:

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
            log.error ("Could not initialise the network. Have you entered the correct model path?")
            raise ValueError("Could not initialise the network. Have you entered the correct model path?")

        self.input_blob = [key for key in self.model.inputs.keys()]
        self.input_shape = []
        for input in self.input_blob:
            self.input_shape.append(self.model.inputs[input].shape)
        self.output_blob=next(iter(self.model.outputs))
        # Gets the output shape of the network
        self.output_shape=self.model.outputs[self.output_blob].shape

    def load_model(self):
        '''
        Load the network 
        '''
        num_req=1
        
        self.net=self.core.load_network(network=self.model, device_name=self.device, num_requests=num_req)
    
        
        return

    def predict(self, left_eye, right_eye, angles):

        request_id = 0
        angles = np.array(angles)
        angles = angles.reshape(1, *angles.shape)

        if np.all(np.array(left_eye.shape)) and np.all(np.array(right_eye.shape)):
            input_dict = {self.input_blob[0]: angles,
                         self.input_blob[1]: self.preprocess_input(left_eye),
                         self.input_blob[2]: self.preprocess_input(right_eye)}

            if self.async_infer:
                self.infer_request_handle = self.net.start_async(request_id=request_id, inputs=input_dict)

                if self.net.requests[request_id].wait(-1) == 0:
                    out = self.infer_request_handle.outputs[self.output_blob]
                    gaze = self.preprocess_output(out)

            else:
                self.infer_request_handle = self.net.infer(inputs=input_dict)
                out = self.infer_request_handle[self.output_blob]
                gaze = self.preprocess_output(out)

        else:
            gaze = []
            log.error("Nothing found")

        return gaze


    def preprocess_input(self, image):
        '''
        Change image-data layout from HWC to CHW 
        '''
        h_img = self.input_shape[1][2]
        w_img = self.input_shape[1][3]
        
        img = cv2.resize(image, (w_img, h_img))
        img = img.transpose((2, 0, 1))
        img = img.reshape(1, *img.shape)
        
        return img
    def preprocess_output(self, outputs):

        try:
            gaze_out = outputs[0]
        except Exception as e:
            log.error("Error in Gaze Estimation Model - process output function" )

        return gaze_out
