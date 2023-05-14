#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import os
import serial
import threading
import time 
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log
import warnings

warnings.filterwarnings('ignore')

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 
start = time.time()
global flag 
global flag2
global poses
global arduino_squats
global pose_squats
arduino_squats = int(0)
pose_squats = int(0)
flag2 = int(0)
flag = int(0)
def arduino_result():
    global flag
    global flag2
    global arduino_squats
    global pose_squats
    #print("The results of Arduino")
    ser= serial.Serial('/dev/ttyACM0', 115200)
    ser.write(b'Start')
    line = ser.readline().decode()
    #print(line)
    #print(line[0])
    if(line[0] == 'N'):
        arduino_squats = 0
    else:
        arduino_squats = 1
    ser.close()
    flag = 0
    flag2 = 0
    if(arduino_squats == 0 and pose_squats == 0):
        arduino_squats = 0
        pose_squats = 0
        print("CORRECT SQUATS")
    else:
        arduino_squats = 0
        pose_squats = 0
        print("WRONG SQUATS")
        
    
def posenet_squat():
    global flag
    global poses
    for pose in poses:
        # find the keypoint index from the list of detected keypoints
        # you can find these keypoint names in the model's JSON file, 
        # or with net.GetKeypointName() / net.GetNumKeypoints()
        left_wrist_idx = pose.FindKeypoint('left_wrist')
        right_wrist_idx = pose.FindKeypoint('right_wrist')
        left_shoulder_idx = pose.FindKeypoint('left_shoulder')

        # if the keypoint index is < 0, it means it wasn't found in the image
        if left_wrist_idx < 0 or left_shoulder_idx < 0:
            continue
        
        left_wrist = pose.Keypoints[left_wrist_idx]
        right_wrist = pose.Keypoints[right_wrist_idx]
        left_shoulder = pose.Keypoints[left_shoulder_idx]

        # point_x = left_shoulder.x - left_wrist.x
        # point_y = left_shoulder.y - left_wrist.y

        #print(left_wrist.x,right_wrist.x)
        
        if (left_wrist.x < left_shoulder.x):
            print(f"SQUATS START POSITION INTITIATED")
            flag = 1

def visual_inspection():
    global poses
    global pose_squats
    for pose in poses:
        left_shoulder_idx = pose.FindKeypoint('left_shoulder')
        right_shoulder_idx = pose.FindKeypoint('right_shoulder')
        left_shoulder = pose.Keypoints[left_shoulder_idx]
        right_shoulder = pose.Keypoints[right_shoulder_idx]
        
        if(abs(left_shoulder.y-right_shoulder.y)>75):
            print("WRONG SHOULDER POSITION")
            pose_squats = 1
        
try:
    
	args = parser.parse_known_args()[0]
except:
	#print("")
	#parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
    curr = time.time()
    #if(curr - start >11):
    #    print("Firing interrupt")
    # start arduino thread
    #    start = curr
    #poses = net.Process(img, overlay=args.overlay)
    poses = net.Process(img)
    if(flag == 1):
        if(flag2 == 0):
            t1 = threading.Thread(target=arduino_result)
            #t2 = threading.Thread(target=visual_inspection)
            t1.start()
            #t2.start()
            flag2 = 1
        visual_inspection()
    #t1.join()
    
    # perform pose estimation (with overlay)
    
    if(flag == 0):
        posenet_squat()

    # print the pose results
    # print("detected {:d} objects in image".format(len(poses)))

    # for pose in poses:
    #     print(pose)
    #     print(pose.Keypoints)
    #     print('Links', pose.Links)
    
    # render the image
    output.Render(img)

    # update the title bar
    #output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    # net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break


    
