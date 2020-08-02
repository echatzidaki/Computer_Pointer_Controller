# Computer Pointer Controller

Using a gaze detection model it is possible to control the mouse pointer of your computer, by estimating the gaze of the user's eyes and change the mouse pointer position accordingly. This project can run multiple models in the same machine and coordinate the flow of data between those models.

[![Udacity](https://video.udacity-data.com/topher/2020/April/5e923081_pipeline/pipeline.png)](https://video.udacity-data.com/topher/2020/April/5e923081_pipeline/pipeline.png)

## Project Overview

1. Proposed a possible hardware solution
2. Built out the person detection application and tested its performance on the DevCloud using multiple hardware types
3. Compared the performance to see which hardware performed best
4. Revised the primal proposal based on the test results

## Project Set Up and Installation

1. Install the intel openVINO toolkit for your system from the official website

     [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html)

2. Clone or download the repository into your local machine.

3. Navigate to the folder src in the project's root directory

4. Create and activate a virtual environment 
    Linux

        sudo apt-get install python3-pip
        pip3 install virtualenv
        virtualenv vino
        virtualenv -p /usr/bin/python3 vino
        source vino/bin/activate
    
5. Install Rrequired libraries

        pip3 install -r requirements.txt

6. Initialize OpenVINO environment

        source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6

7. Download the models using the script 'model_downloader.sh'.
        
        python3 ./model_downloader.sh

8. Run 'main.py' with '-h' to see the input options.
        
        python3 main.py -h


### Run a Demo
9.    python3 main.py -graphics fd fld hpe ge

#### Command Line Arguments 
By default the app run with the webcam on a CPU with Intel OpenVINO models in FP16 except Face Detection which uses FP32-INT1, however the user can change it: 

<pre>usage: main.py [-h] [-fd FACEDETECTIONMODEL]
               [-fld FACIALLANDMARKSDETECTIONMODEL]
               [-hpe HEADPOSEESTIMATIONMODEL] [-ge GAZEESTIMATIONMODEL]
               [-i INPUT] [-ext EXTENSIONS]
               [-graphics SHOW_GRAPHICS [SHOW_GRAPHICS ...]]
               [-prob PROB_THRESHOLD] [-d DEVICE]
               [-devfd {CPU,GPU,MYRIAD,MULTI:CPU,MYRIAD,MULTI:GPU,MYRIAD,MULTI:CPU,GPU,MYRIAD,MULTI:HDDL,GPU,HETERO:MYRIAD,CPU,HETERO:GPU,CPU,HETERO:FPGA,GPU,CPU,HDDL}]
               [-async] [-o_stats OUTPUT_STATS]
</pre>


| arguments | explanation |
| --------- | ----------- |
|  -fd|Specify Path to .xml file of Face Detection model.|
|  -fld |Specify Path to .xml file of Facial Landmarks Detection model.|
|  -hpe |Specify Path to .xml file of Head Pose Estimation model.|
|  -ge | Specify Path to .xml file of Gaze Estimation model.|
|  -i |Specify Path to video file or enter cam for webcam.|
|  -ext |MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl.|
|  -graphics | Specify the models you want to show from fd, fld, hpe, ge, all, stats like --show_graphics fd hpe fld (Seperate each flag by space)for see the visualization of different model outputs of each frame,fd for Face Detection, fld for Facial Landmark Detectionhpe for Head Pose Estimation, ge for Gaze Estimation,nog for output without graphicsstats to show inference time.|
|  -prob |(Optional) Probability threshold for detection filtering (0.5 by default) |
|  -d |(Optional) Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. Default device is CPU. 
|  -devfd | (Optional) Specify the target device to infer on for the Face Detection model (CPU by default) {CPU,GPU,MYRIAD,MULTI:CPU,MYRIAD,MULTI:GPU,MYRIAD,MULTI:CPU,GPU,MYRIAD,MULTI:HDDL,GPU,HETERO:MYRIAD,CPU,HETERO:GPU,CPU,HETERO:FPGA,GPU,CPU,HDDL}.|
|  -async |  (Optional) Perform sync or async inference..|
|  -o_stats |Save performance stats in given path.|

### Directory Structure
```
├── bin
│   ├── semo.mp4
│   ├── stats
│         └── stats.txt
├── models
│   └── intel
│       ├── face-detection-adas-binary-0001
│       │   └── FP32-INT1
│       │       ├── face-detection-adas-binary-0001.bin
│       │       └── face-detection-adas-binary-0001.xml
│       ├── gaze-estimation-adas-0002
│       │   ├── FP16
│       │   │   ├── gaze-estimation-adas-0002.bin
│       │   │   └── gaze-estimation-adas-0002.xml
│       │   ├── FP16-INT8
│       │   │   ├── gaze-estimation-adas-0002.bin
│       │   │   └── gaze-estimation-adas-0002.xml
│       │   └── FP32
│       │       ├── gaze-estimation-adas-0002.bin
│       │       └── gaze-estimation-adas-0002.xml
│       ├── head-pose-estimation-adas-0001
│       │   ├── FP16
│       │   │   ├── head-pose-estimation-adas-0001.bin
│       │   │   └── head-pose-estimation-adas-0001.xml
│       │   ├── FP16-INT8
│       │   │   ├── head-pose-estimation-adas-0001.bin
│       │   │   └── head-pose-estimation-adas-0001.xml
│       │   └── FP32
│       │       ├── head-pose-estimation-adas-0001.bin
│       │       └── head-pose-estimation-adas-0001.xml
│       └── landmarks-regression-retail-0009
│           ├── FP16
│           │   ├── landmarks-regression-retail-0009.bin
│           │   └── landmarks-regression-retail-0009.xml
│           ├── FP16-INT8
│           │   ├── landmarks-regression-retail-0009.bin
│           │   └── landmarks-regression-retail-0009.xml
│           └── FP32
│               ├── landmarks-regression-retail-0009.bin
│               └── landmarks-regression-retail-0009.xml
├── README.md
├── requirements.txt
├──  main.py
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── graphics.py
    └── mouse_controller.py
```

## Benchmarks
The application build in a Linux VirtualBox vm. Virtual box does not support GPU connection. The system used only the CPU. It is recommended to use VMWare to connect and run on the GPU. The CPU that was used is an Intel Core i3-8145U @2.1-2.3GHz with 8 GB RAM. The VirtualBox used half of CPU cores (2) and 4GB RAM.

## Results

Inference time for FP32, FP16 and INT8 models.
FP32-INT1 is the binary edition of face detection pretrained model. 

| Async - Model  |FP32-INT1 + FP16|  FP16 + FP16 | FP32-INT1 + FP16-INT8|
| ----------- | ----------- | ----------- | ----------- | 
| Face Det. | 1.5081827640533447 | 1.9178917407989502 | **1.1236345767974854** | 
| Facial Landmarks Det. | 0.0929574966430664 | **0.044764041900634766** | 0.04594707489013672 | 
| Head Pose Est.| 0.1161191463470459 | 0.11187100410461426 | **0.07950186729431152** | 
| Gaze Est. | 0.14135289192199707 | 0.11991500854492188 | **0.08041238784790039** | 

Inference time of models (asynchronous) perform better at Face Detection FP32-INT1 and the others FP16-INT8 combination. 

| Sync - Model  | FP32-INT1 + FP16  |  FP16 + FP16 | FP32-INT1 + FP16-INT8|
| ----------- | ----------- | ----------- | ----------- | 
| Face Det.             | 1.1949219703674316 | 1.7288234233856201 | **1.228731393814087** | 
| Facial Landmarks Det. | **0.04770374298095703** | 0.057895660400390625| 0.053205013275146484 | 
| Head Pose Est.        | **0.10491204261779785** | 0.10558724403381348 | 0.10793638229370117 | 
| Gaze Est.             | 0.12326288223266602 | 0.11881518363952637 | **0.0959312915802002** | 

At synchronous mode best performance (inference time) varies. However, the FP32-INT1 seems to be the best choice.

| Inference Time - Mode  |   FP32-INT1 + FP16  |  FP16 + FP16 | FP32-INT1 + FP16-INT8|
| ----------- | ----------- | ----------- | ----------- | 
| Async - Total - All models    |  **82.15526676177979** | 85.76474475860596  | 82.78374147415161 | 
| Sync - Total - All models    | **82.7205445766449** | 83.52421069145203 | 84.85066962242126 |

The FP32-INT1 + FP16 combination has the best total inference time in both asynchronous and synchronous modes.

| Load Time - Mode  |   FP32-INT1 + FP16  |   FP16 + FP16 | FP32-INT1 + FP16-INT8|
| ----------- | ----------- | ----------- | ----------- | 
| Async - Total - All models    | **0.49913835525512695** | 0.637099027633667  | 0.7627449035644531 | 
| Sync - Total - All models    | **0.4821746349334717**  |0.5999157428741455| 0.7493643760681152 |

Also, the FP32-INT1 + FP16 combination has the best total load time in both asynchronous and synchronous modes.

| FPS - Mode  |   FP32-INT1 + FP16  |  FP16 + FP16 | FP32-INT1 + FP16-INT8|
| ----------- | ----------- | ----------- | ----------- | 
| Async - Total - All models    | **0.718152375684914** |  0.6879283575793504 | 0.7127003316034243 |
| Sync - Total - All models    | **0.7132448208840481** | 0.7063820120126933 | 0.6953392384826815|

Finally, the FP32-INT1 + FP16 combination has the best FPS in both asynchronous and synchronous modes.

## Stand Out Suggestions
### Async Inference
Using async inference, the application achieves better performance, due to  multithreading.

### Edge Cases

The application uses the first detected face and ignores anyone else.
