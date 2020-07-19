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

class GazeEstimation:
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


    def predict(self, left_eye_image, right_eye_image, head_pose_angle):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye_img_proc, right_eye_img_proc = self.preprocess_input(left_eye_img, right_eye_img)
        results = self.net.infer(
            {'head_pose_angles': head_pose_angle, 'left_eye_image': left_eye_img_proc,
            'right_eye_image': right_eye_img_proc})
        updated_mouse_coord, gaze_vector = self.preprocess_output(results, head_pose_angle)

        return updated_mouse_coord, gaze_vector

    def check_model(self):


    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        left_eye_img_proc = cv2.resize(image, self.input_shape[3], self.input_shape[2])
        left_eye_img_proc = left_eye_img_proc.transpose((2,0,1))
        left_eye_img_proc = left_eye_img_proc.resize(1, *left_eye_img_proc.shape)

        right_eye_img_proc = cv2.resize(image, self.input_shape[3], self.input_shape[2])
        right_eye_img_proc = right_eye_img_proc.transpose((2,0,1))
        right_eye_img_proc = right_eye_img_proc.resize(1, *right_eye_img_proc.shape)

        return left_eye_img_proc, right_eye_img_proc

    def preprocess_output(self, outputs, head_pose_estimation):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        gaze_vector = outputs[self.output_name][0]
        roll_angle = head_pose_estimation[2]
        cos_angle = math.cos(roll_angle * math.pi / 180.0)
        sin_angle = math.sin(roll_angle * math.pi / 180.0)

        x_coord = outputs[0] * cos_angle + outputs[1] * sin_angle
        y_coord = outputs[1] * cos_angle - outputs[0] * sin_angle

        return (x_coord, y_coord), gaze_vector
