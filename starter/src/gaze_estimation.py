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
import math

class GazeEstimation:
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

        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        self.input_name = [i for i in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.model.outputs.keys()]

    def predict(self, left_eye_img, right_eye_img, head_pose_angle):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye_img_proc, right_eye_img_proc = self.preprocess_input(left_eye_img, right_eye_img)
        results = self.net.infer(
            {'head_pose_angles': head_pose_angle, 'left_eye_image': left_eye_img_proc,
            'right_eye_image': right_eye_img_proc})
        new_mouse_coord, gaze_vector = self.preprocess_output(results, head_pose_angle)

        return new_mouse_coord, gaze_vector

    def check_model(self):
        pass

    def preprocess_input(self,  left_eye_img, right_eye_img):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_eye_img_proc = cv2.resize(left_eye_img, (self.input_shape[3], self.input_shape[2]))
        left_eye_img_proc = np.transpose(np.expand_dims(left_eye_img_proc,axis=0), (0,3,1,2))

        right_eye_img_proc = cv2.resize(right_eye_img, (self.input_shape[3], self.input_shape[2]))
        right_eye_img_proc = np.transpose(np.expand_dims(right_eye_img_proc,axis=0), (0,3,1,2))

        return left_eye_img_proc, right_eye_img_proc

    def preprocess_output(self, outputs, head_pose_estimation):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[self.output_names[0]].tolist()[0]
        roll_angle = head_pose_estimation[2]
        cos_angle = math.cos(roll_angle * math.pi / 180.0)
        sin_angle = math.sin(roll_angle * math.pi / 180.0)

        x_coord = gaze_vector[0] * cos_angle + gaze_vector[1] * sin_angle
        y_coord = gaze_vector[1] * cos_angle - gaze_vector[0] * sin_angle

        return (x_coord, y_coord), gaze_vector
