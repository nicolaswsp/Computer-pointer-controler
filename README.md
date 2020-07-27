# Computer Pointer Controller
This project is a computer vision application developed with Intel Distribution of OpenVINO. This application was developed to control the computer pointer controler through the gaze estimation of the eyes. This program can be input as a video or a webcam. It was used four models to run Face Detection, Facial Landmarks Detection, Head Pose Estimation and Gaze Estimation all provided by OpenVINO.

## Project Set Up and Installation
The requirements for this project are included in the requiriments.txt. The instalation can be done with the following command:
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
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.
The pipeline project can be understood easily with the following image:


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

