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
        self.model_structure = model_name+'xml'
        self.model_weights = model_name+'bin'
        self.device = device

        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you \
            enterred the correct model path?")
        raise NotImplementedError

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        net = self.core.load_network(network=model, device_name='CPU', num_requests=1)


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_frame = self.preprocess_input(image)
        outputs = self.net.infer({self.input_name: p_frame})
        coords = self.preprocess_outputs(outputs[self.output_name], image)

        if len(corrds) == 0:
            print("Face was not detected")
            return 0, 0
        face_detected_coords = coords[0]
        cropped_face = image[face_detected_coords[1]:face_detected_coords[3],
        face_detected_coords[0]:face_detected_coords[2]]

        return face_detected_coords, cropped_face

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

    def preprocess_output(self, outputs, image):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        coords=[]
        for box in outputs[self.output_name][0][0]:
            conf_t = box[2]
            if conf_t > self.threshold:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                coords.append((xmin, ymin, xmax, ymax))

        return coords
