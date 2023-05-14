poses = net.Process(img)

for pose in poses:
    # find the keypoint index from the list of detected keypoints
    # you can find these keypoint names in the model's JSON file, 
    # or with net.GetKeypointName() / net.GetNumKeypoints()
    left_wrist_idx = pose.FindKeypoint('left_wrist')
    left_shoulder_idx = pose.FindKeypoint('left_shoulder')

    # if the keypoint index is < 0, it means it wasn't found in the image
    if left_wrist_idx < 0 or left_shoulder_idx < 0:
        continue
	
    left_wrist = pose.Keypoints[left_wrist_idx]
    left_shoulder = pose.Keypoints[left_shoulder_idx]

    point_x = left_shoulder.x - left_wrist.x
    point_y = left_shoulder.y - left_wrist.y

    print(f"person {pose.ID} is pointing towards ({point_x}, {point_y})")
