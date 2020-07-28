import cv2
import os
import logging
import numpy as np
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarks
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():
    '''
    Parse command line arguments.
    :return: command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-f", "--face_detection_model", required=True, type=str,
                        help="Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str,
                        help="Path to .xml file of Facial Landmarks Detection model.")
    parser.add_argument("-hp", "--head_pose_estimation_model", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--models_outputs_flags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge typing --flags fd hp fld"
                             "this allows to see the outputs for the inferenced models,"
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="CPU Extension for custom layers")
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for detection filtering.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA, or MYRIAD"
                             "CPU is the default device")

    return parser



def main():

    # Grab command line args
    args = build_argparser().parse_args()
    flags = args.models_outputs_flags

    logger = logging.getLogger()
    input_file_path = args.input
    input_feeder = None
    if input_file_path.lower() == "cam":
            input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file_path):
            logger.error("Unable to find specified video file")
            exit(1)
        input_feeder = InputFeeder("video", input_file_path)

    model_path_dict = {'FaceDetection':args.face_detection_model, 'FacialLandmarks':args.facial_landmarks_model,
    'GazeEstimation':args.gaze_estimation_model, 'HeadPoseEstimation':args.head_pose_estimation_model}

    for file_name_key in model_path_dict.keys():
        if not os.path.isfile(model_path_dict[file_name_key]):
            logger.error("Unable to find specified " + file_name_key + " xml file")
            exit(1)

    fdm = FaceDetection(model_path_dict['FaceDetection'], args.device, args.cpu_extension)
    flm = FacialLandmarks(model_path_dict['FacialLandmarks'], args.device, args.cpu_extension)
    gem = GazeEstimation(model_path_dict['GazeEstimation'], args.device, args.cpu_extension)
    hpem = HeadPoseEstimation(model_path_dict['HeadPoseEstimation'], args.device, args.cpu_extension)

    mc = MouseController('medium','fast')

    input_feeder.load_data()
    fdm.load_model()
    flm.load_model()
    hpem.load_model()
    gem.load_model()

    frame_count = 0
    for ret, frame in input_feeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video', cv2.resize(frame, (500,500)))

        key = cv2.waitKey(60)
        cropped_face, face_coords = fdm.predict(frame, args.prob_threshold)
        if type(cropped_face)==int:
            logger.error("Unable to detect any face.")
            if key==27:
                break
            continue

        hp_output = hpem.predict(cropped_face)

        left_eye_img, right_eye_img, eye_coords = flm.predict(cropped_face)

        new_mouse_coord, gaze_vector = gem.predict(left_eye_img, right_eye_img, hp_output)

        if (not len(flags)==0):
            preview_frame = frame
            if 'fd' in flags:
                preview_frame = cropped_face
            if 'fld' in flags:
                cv2.rectangle(cropped_face, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                cv2.rectangle(cropped_face, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)

            if 'hp' in flags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_output[0], hp_output[1], hp_output[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            if 'ge' in flags:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                left_eye = cv2.line(left_eye_img, (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(left_eye, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                right_eye = cv2.line(right_eye_img, (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(right_eye, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                cropped_face[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = left_eye
                cropped_face[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = right_eye

            cv2.imshow("Visualization", cv2.resize(preview_frame,(500,500)))

        if frame_count%5==0:
            mc.move(new_mouse_coord[0], new_mouse_coord[1])
        if key==27:
                break
    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    input_feeder.close()



if __name__ == '__main__':
    main()
