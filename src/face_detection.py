import cv2
import sys
import logging as log
from openvino.inference_engine import IECore, IENetwork


class FaceDetector:


    def __init__(self, model_name, prob_threshold=0.60, device='CPU', extensions=None, async_infer=True):

        self.threshold=prob_threshold
        self.extensions = extensions
        
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
            sys.exit(1)
        
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
                out = self.infer_request_handle.outputs[self.output_blobs]
                face_coords, crop_face = self.preprocess_output(out, image)
        else:
            self.infer_request_handle = self.net.infer(inputs=input_dict)
            out = self.infer_request_handle[self.output_blobs]
            face_coords, crop_face = self.preprocess_output(out, image)


        return face_coords, crop_face

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
        
    def preprocess_output(self, outputs, image):

        coords = []
        width, height = int(image.shape[1]), int(image.shape[0])

        for output in outputs[0][0]:
            confidence = output[2]
            if confidence > self.threshold:
                coords.append(output[3:])

        if len(coords) != 0:
            # Chose the first detected face to control mouse. Ignore people around
            coords = coords[0]
            xmin = int(coords[0] * width) - 10
            ymin = int(coords[1] * height) - 10 
            xmax = int(coords[2] * width) + 10
            ymax = int(coords[3] * height) + 10

            face_coords = [xmin, ymin, xmax, ymax]
            crop_face = image[ymin:ymax, xmin:xmax]
        else:
            face_coords = []
            crop_face = []
            
        return face_coords, crop_face