# Computer Pointer Controller

*TODO:* Write a short introduction to your project
This project is a computer vision application developed with Intel Distribution of OpenVINO. This application was developed to control the computer pointer controler through the gaze estimation of the eyes. This program can be input as a video or a webcam. It was used four models to run Face Detection, Facial Landmarks Detection, Head Pose Estimation and Gaze Estimation all provided by OpenVINO.
## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.
It is required install the OpenVINO Toolkit to run the project. After the instalation, it is necessary to download the models face-detection-adas-binary-0001, landmarks-regression-retail-0009, head-pose-estimation-adas-0001, gaze-estimation-adas-0002 through the model downloader provided by OpenVino in the directory C:\Program Files (x86)\IntelSWTools\openvino_2020.1.033\deployment_tools\tools\model_downloader using the downloader.py file. The
## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
