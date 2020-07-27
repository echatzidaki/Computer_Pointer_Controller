# Store input arguments: <output_directory> <device> <fp_precision> <input_file>
OUTPUT_FILE=$1
DEVICE=$2
FP_MODEL=$3
INPUT_FILE=$4
THRESHOLD=$5

# The default path for the job is the user's home directory,
#  change directory to where the files are.
cd $PBS_O_WORKDIR

# Make sure that the output directory exists.
mkdir -p $OUTPUT_FILE

FACE_MODEL_PATH=models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml
    
if [ "$FP_MODEL" = "FP32" ]; then
    # Set inference model IR files using specified precision
    HEAD_MODEL_PATH=../models/intel/head-pose-estimation-adas-0001/${FP_MODEL}/head-pose-estimation-adas-0001.xml
    GAZE_MODEL_PATH=../models/intel/gaze-estimation-adas-0002/${FP_MODEL}/gaze-estimation-adas-0002.xml
    LAND_MODEL_PATH=../models/intel/landmarks-regression-retail-0009/${FP_MODEL}/landmarks-regression-retail-0009.xml
      
elif [ "$FP_MODEL" = "FP16" ]; then    
    HEAD_MODEL_PATH=../models/intel/head-pose-estimation-adas-0001/${FP_MODEL}/head-pose-estimation-adas-0001.xml
    GAZE_MODEL_PATH=../models/intel/gaze-estimation-adas-0002/${FP_MODEL}/gaze-estimation-adas-0002.xml
    LAND_MODEL_PATH=../models/intel/landmarks-regression-retail-0009/${FP_MODEL}/landmarks-regression-retail-0009.xml

elif [ "$FP_MODEL" = "FP16-INT8" ]; then     
    HEAD_MODEL_PATH=../models/intel/head-pose-estimation-adas-0001/${FP_MODEL}/head-pose-estimation-adas-0001.xml
    GAZE_MODEL_PATH=../models/intel/gaze-estimation-adas-0002/${FP_MODEL}/gaze-estimation-adas-0002.xml
    LAND_MODEL_PATH=../models/intel/landmarks-regression-retail-0009/${FP_MODEL}/landmarks-regression-retail-0009.xml
else   
    HEAD_MODEL_PATH=../models/intel/head-pose-estimation-adas-0001/${FP_MODEL}/head-pose-estimation-adas-0001.xml
    GAZE_MODEL_PATH=../models/intel/gaze-estimation-adas-0002/${FP_MODEL}/gaze-estimation-adas-0002.xml
    LAND_MODEL_PATH=../models/intel/landmarks-regression-retail-0009/${FP_MODEL}/landmarks-regression-retail-0009.xml    
fi


# Check for special setup steps depending upon device to be used
if echo "$DEVICE" | grep -q "FPGA"; then
#if [ "$DEVICE" = "HETERO:FPGA,CPU" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs - Updated for OpenVINO 2020.3
    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2
    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-3_PL2_FP16_MobileNet_Clamp.aocx
    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

# Run the pointer control code
python3 pointer-control.py -fm $FACE_MODEL_PATH \
                                -hm $HEAD_MODEL_PATH \
                                -gm $GAZE_MODEL_PATH \
                                -lm $LAND_MODEL_PATH \
                                -i $INPUT_FILE \
                                -o $OUTPUT_FILE \
                                -d $DEVICE \
                                -th $THRESHOLD

cd /output

tar zcvf output.tgz *