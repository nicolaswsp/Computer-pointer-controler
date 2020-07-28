# Computer Pointer Controller
This project is a computer vision application developed with Intel Distribution of OpenVINO. This application was developed to control the computer pointer controler through the gaze estimation of the eyes. This program can be feed with a video file or a webcam. It was used four models to run Face Detection, Facial Landmarks Detection, Head Pose Estimation and Gaze Estimation all provided by OpenVINO.

## Project Set Up and Installation
The requirements for this project are included in the requeriments.txt. The instalation can be done with the following command:

pip install -r requirements.txt

To obtain the python scrips used in this project you can clone my repository:
https://github.com/nicolaswsp/Computer-pointer-controler.git

It is required install the OpenVINO Toolkit to run the project. You can to https://docs.openvinotoolkit.org/ and following the installation steps to your operating system. 
After the instalation, it is necessary to download the models face-detection-adas-binary-0001, landmarks-regression-retail-0009, head-pose-estimation-adas-0001, gaze-estimation-adas-0002 through the model downloader provided by OpenVino in the directory C:<OpenVINO-Path>\IntelSWTools\openvino_2020.1.033\deployment_tools\tools\model_downloader using the downloader.py file. The following commands can be run in the shell to download the models:
 
Face Detection Model - python downloader.py --name face-detection-adas-binary-0001

Facial Landmarks Detection Model - python downloader.py --name landmarks-regression-retail-0009

Head Pose Estimation Model - python downloader.py --name head-pose-estimation-adas-0001

Gaze Estimation Model - python downloader.py --name gaze-estimation-adas-0002 
 
## Demo
First, it is necessary to set the OpenVINO environment. The followind command can be typed:

cd <OpenVINO-Path>\IntelSWTools\openvino\bin\
setupvars.bat
 
Secondly, open the repository folder, for example

cd <Project-Repo-Path>\src
  
Lastly, type the following command to run the application:

python main.py -f face-detection-adas-binary-0001.xml -fl landmarks-regression-retail-0009.xml -hp head-pose-estimation-adas-0001.xml -g gaze-estimation-adas-0002.xml -i "video file" or "CAM"

You can also check ou other inputs with --help.
For default the device to run the inference is CPU, but you can also use GPU, MYRIAD and FPGA.


## Documentation
The pipeline project can be understood as the following. First, it is obtained the cropped face of the person detected with the Face Detection Model. After this, the cropped face is input in the Facial Landamarks Detection model and in the Head Pose Estimation model. Finaly, boths outputs are feed in the Gaze Estimation model and then we obtain a gaze vector that will move the computer mouse controler. 

You can check the models documatation in the following links:
https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html
https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

For the command line the following are the inputs to main.py:

-f "Path to an .xml file with Face Detection model"
-fl "Path to an .xml file with Facial Landmark Detection model"
-hp "Path to an .xml file with Head Pose Estimation model".
-g "Path to an .xml file with Gaze Estimation model"
-i "Path to video file or enter cam for webcam"
-flags(optional) "Visualize the outputs for the inferenced models"
-l(optional) "CPU Extension for custom layers"
-pt(optional) "Probability threshold for detection filtering"
-d(optional) "Target device"

## Benchmarks
I compared the inference Time, model loading Time, and frames per second model for the precisions INT8, FP16, FP32 in IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE (CPU) and IEI Tank 870-Q170 edge node with an Intel® Core™ i5-6500TE (CPU + Integrated Intel® HD Graphics 530 card GPU). 

<b>INT8<b>
CPU - inference time: 79s - loading time: 1.3s - FPS 8
GPU - inference time: 74s - loading time: 52.4s - FPS 9
<b>FP16<b>
CPU - inference time: 77s - loading time: 1.3s - FPS 8
GPU - inference time: 75s - loading time: 52.4s - FPS 9
<b>FP32<b>
CPU - inference time: 68s - loading time: 1.5s - FPS 9
GPU - inference time: 69s - loading time: 55s - FPS 9
 
## Results
Comparing the results above, we can se that when reducing the precision the inference time was higher although the loading time was slightly lower. The lower precisions models have lower accuracy when compared to the FP32. Comparing the CPU against GPU, the GPU's model loading time was much higher than the CPU's as expected. However, the FPS were very close between them.
