'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import sys

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.device = device
        self.extensions = extensions
        self.core = None
        self.model = None
        self.net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.model=IENetwork(self.model_structure, self.model_weights)


        supported_layers = self.core.query_network(self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0 and self.device=='CPU':
            print("Unsupported layers found: {}".format(unsupported_layers))
            if not self.extensions==None:
                self.core.add_extension(self.extensions, self.device)
                supported_layers = self.core.query_network(network = self.model, device_name=self.device)
                unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!= 0:
                    print("Even adding the extension there are unsupported layers found")
                    exit(1)
            else:
                print("Check whether extensions are available to add to IECore and provide the path.")
                exit(1)

        self.net = self.core.load_network(network=self.model, device_name='CPU', num_requests=1)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_frame = self.preprocess_input(image)
        outputs = self.net.infer({self.input_name: p_frame})
        coords = self.preprocess_output(outputs, image, prob_threshold)

        if len(coords) == 0:
            return 0, 0
        face_detected_coords = coords[0]
        cropped_face = image[face_detected_coords[1]:face_detected_coords[3],
        face_detected_coords[0]:face_detected_coords[2]]

        return cropped_face, face_detected_coords

    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs, image, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords=[]
        outs = outputs[self.output_name][0][0]
        for box in outs:
            conf_t = box[2]
            if conf_t > prob_threshold:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                coords.append([xmin, ymin, xmax, ymax])

        return coords
