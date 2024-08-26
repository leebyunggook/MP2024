import os
import cv2
import csv
import math
import time
import platform
import numpy as np
import mediapipe as mp
from skimage import draw

keypoint_ids = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17)
]

fps = 30
width = 1920
height = 1080
stime = ""

radius = 3
thickness = 1
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

def cameraIndexes(n):
    # checks the first 10 indexes.
    arr = []
    index = 0
    while index < n:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Frame Width
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Frame Height
            fps = cap.get(cv2.CAP_PROP_FPS) 
            print('camera ' + str(index) + ' width: ' + str(width) + ', height: ' + str(height) + ', fps: ' + str(fps))
            arr.append(index)
            cap.release()
        index += 1
    return arr


def setCamera(camId, _width, _height):
    global fps, width, height
    width = _width
    height = _height
    cap = cv2.VideoCapture(camId) #, cv2.CAP_DSHOW
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps using camera id = %d  : %d" % (camId, fps))
    return cap, fps


def setSaveSkeleton(output_dir):
    global fps, width, height, stime
    stime = time.strftime('%Y%m%d%H%M', time.localtime())
    outputvideo = output_dir + "video_" + stime + ".mp4" 
    skeletonvideo = output_dir + "skeleton_" + stime + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outputvideo, fourcc, fps, (width, height))
    skl = cv2.VideoWriter(skeletonvideo, fourcc, fps, (width, height))
    print("%s" % (outputvideo))
    print("%s" % (skeletonvideo))
    return out, skl


def setSavePose(output_dir, mp_pose):
    global fps, width, height, stime
    outputcsv = output_dir + "skeleton_" + stime + ".csv"
    csvfile = open(outputcsv, "wt", newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    jointlist = []
    jointlist.extend(['frameCount', 'id'])
    for plm in mp_pose.PoseLandmark:
        jointlist.extend(['x '+str(plm), 'y '+str(plm), 'z '+str(plm), 'viz '+str(plm)])
    writer.writerow(jointlist)
    print("%s" % (outputcsv))
    return csvfile, writer


def setSaveHands(output_dir, mp_hands):
    global fps, width, height, stime
    outputcsv = output_dir + "skeleton_" + stime + ".csv"
    csvfile = open(outputcsv, "wt", newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    jointlist = []
    jointlist.extend(['frameCount', 'id'])
    for plm in mp_hands.HandLandmark:
        jointlist.extend(['x '+str(plm), 'y '+str(plm), 'z '+str(plm), 'viz '+str(plm)])
    writer.writerow(jointlist)
    print("%s" % (outputcsv))
    return csvfile, writer


def setSaveFaces(output_dir, length):
    global fps, width, height, stime
    outputcsv = output_dir + "skeleton_" + stime + ".csv"
    csvfile = open(outputcsv, "wt", newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    jointlist = []
    jointlist.extend(['frameCount', 'id'])
    for i in range(length):
        jointlist.extend([str(i) + '_x', str(i) + '_y', str(i) + '_z'])
    writer.writerow(jointlist)
    print("%s" % (outputcsv))
    return csvfile, writer


def closeSkeleton(out, skl, csv):
    csv.close()
    out.release()    
    skl.release()    


def setCVWindows(title, display_scale, margin):
    global width, height
    dwidth = int(width/display_scale+margin)
    dheight = int(height/display_scale)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)        
    cv2.resizeWindow(title, dwidth, dheight)  
    return

def debugFrameMessage(frame, font, frameCount, start, count):
    seconds = time.time() - start
    fps  = frameCount / seconds
    msg = "fps {:.2f}".format(fps) + " - {0}".format(frameCount)+ "/{:.2f} sec".format(seconds) + " - {0}".format(count)
    cv2.putText(frame, msg, (10, 30), font, 0.6, (255,0,0), 1, cv2.LINE_AA)
    return


def writePoseLandmark(writer, mp_pose, results, frameCount):
    global width, height
    landmark = results.pose_landmarks.landmark
    if not landmark:
        return
    jointlist = []
    jointlist.extend([frameCount, 0])
    for plm in mp_pose.PoseLandmark:
        jointlist.extend([landmark[plm.value].x, landmark[plm.value].y, landmark[plm.value].z, landmark[plm.value].visibility])
    writer.writerow(jointlist)


def printPoseLandmark(frame, mp_pose, results, width, height):
    if mp_pose == None or results == None:
        return
    landmark = results.pose_landmarks.landmark
    if not landmark:
        return
    # for plm in mp_pose.PoseLandmark:
    #     print("%2d - (%7.2f, %7.2f, %7.2f) - %4.2f " % (plm.value, landmark[plm.value].x * width, 
    #         landmark[plm.value].y * height, landmark[plm.value].z, landmark[plm.value].visibility), plm)  
    id = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    title = ["shoulder", "elbow", "wrist", "pinky", "index", "thumb", "hip", "knee", "ankle", "heel", "foot index"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg = "left coordinate"
    cv2.putText(frame, msg, (width+100, 20), font, .4, (255,255,255), 1, cv2.LINE_AA)
    msg = "right coordinate"
    cv2.putText(frame, msg, (width+250, 20), font, .4, (255,255,255), 1, cv2.LINE_AA)

    for i in range(11) :
        cv2.putText(frame, title[i], (width+10, 50+i*20), font, .4, (255,255,255), 1, cv2.LINE_AA)

    for i in range(11) :
        center_coordinates = (int(landmark[id[i]].x * width), int(landmark[id[i]].y * height))
        cv2.circle(frame, center_coordinates, 6, (0,0,255), 3)
        msg = "{0}".format(center_coordinates) + " - {:.2f}".format(landmark[id[i]].visibility)
        cv2.putText(frame, msg, (width+100, 50+i*20), font, .4, (255,255,255), 1, cv2.LINE_AA)

    for i in range(11) :
        center_coordinates = (int(landmark[id[i]+1].x * width), int(landmark[id[i]+1].y * height))
        cv2.circle(frame, center_coordinates, 6, (0,0,255), 3)
        msg = "{0}".format(center_coordinates) + " - {:.2f}".format(landmark[id[i]+1].visibility)
        cv2.putText(frame, msg, (width+250, 50+i*20), font, .4, (255,255,255), 1, cv2.LINE_AA)


def writeHandsLandmark(writer, mp_hands, results, frameCount):
    global width, height
    index = 0
    for hand_landmarks in results.multi_hand_landmarks:
        jointlist = []
        jointlist.extend([frameCount, index])
        for plm in mp_hands.HandLandmark:
            jointlist.extend([hand_landmarks.landmark[plm.value].x, hand_landmarks.landmark[plm.value].y, 
                hand_landmarks.landmark[plm.value].z, hand_landmarks.landmark[plm.value].visibility])
        writer.writerow(jointlist)
        index += 1


def printHandsLandmark(mp_hands, results):
    global width, height
    for hand_landmarks in results.multi_hand_landmarks:
        for plm in mp_hands.HandLandmark:
            print("(%7.2f, %7.2f, %7.2f) - %4.2f " % (hand_landmarks.landmark[plm.value].x * width, 
                hand_landmarks.landmark[plm.value].y * height, hand_landmarks.landmark[plm.value].z, 
                hand_landmarks.landmark[plm.value].visibility), plm)  


def writeFacesLandmark(writer, results, frameCount):
    global width, height
    index = 0
    jointlists = []
    for face in results.multi_face_landmarks:
        jointlist = []
        jointlist.extend([frameCount, index])
        for landmark in face.landmark:
            jointlist.extend([landmark.x, landmark.y, landmark.z])
        writer.writerow(jointlist)
        jointlists.append(jointlist)
        index += 1
    return jointlists    


def printFacesLandmark(results):
    global width, height
    i = 0
    for face in results.multi_face_landmarks:
        j = 0
        print("face - %d" % i) 
        for landmark in face.landmark:
            print("%3d - (%7.2f, %7.2f, %7.2f)" % (j, landmark.x * width, 
                landmark.y * height, landmark.z))
            j += 1      
        i += 1        


def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle


def get_valid_keypoints(keypoint_ids, skeleton, confidence_threshold):
    keypoints = [
        (tuple(map(int, skeleton.joints[i])), tuple(map(int, skeleton.joints[v])))
        for (i, v) in keypoint_ids
        if skeleton.confidences[i] >= confidence_threshold
        and skeleton.confidences[v] >= confidence_threshold
    ]
    valid_keypoints = [
        keypoint
        for keypoint in keypoints
        if keypoint[0][0] >= 0 and keypoint[0][1] >= 0 and keypoint[1][0] >= 0 and keypoint[1][1] >= 0
    ]
    return valid_keypoints


def render_result(skeletons, img, confidence_threshold):
    skeleton_color = (100, 254, 213)
    for index, skeleton in enumerate(skeletons):
        keypoints = get_valid_keypoints(keypoint_ids, skeleton, confidence_threshold)
        for keypoint in keypoints:
            cv2.line(img, keypoint[0], keypoint[1], skeleton_color, thickness=2, lineType=cv2.LINE_AA)


def print_result(skeletons, img, confidence_threshold):
    for skeleton in skeletons:
        for joint in skeleton.joints:
            x, y = tuple(map(int, joint))
            if x < 0 or y < 0:
                continue
            print("(%3d,%3d)-%3d" % (x, y, skeleton.id))
    for index, skeleton in enumerate(skeletons):
        keypoints = get_valid_keypoints(keypoint_ids, skeleton, confidence_threshold)
        for keypoint in keypoints:
            print("(%3d,%3d)-(%3d,%3d)" % (keypoint[0][0], keypoint[0][1], keypoint[1][0], keypoint[1][1]))


def write_skeleton(writer, skeletons, frameCount):
    for skeleton in skeletons:
        jointlist = []
        jointlist.extend([frameCount, skeleton.id])
        for i in range(18):
            c = skeleton.confidences[i]
            joint = skeleton.joints[i]
            x, y = tuple(map(int, joint))
            jointlist.extend([x, y, '{:.2f}'.format(c)])
        writer.writerow(jointlist)


def render_ids(skeletons, img, thickness=5):
    id_text_color_offline_tracking = (51, 153, 255)
    id_text_color_cloud_tracking = (57, 201, 100)
    text_color = id_text_color_offline_tracking
    for skeleton in skeletons:
        if skeleton.id_confirmed_on_cloud == True:
            text_color = id_text_color_cloud_tracking
        for joint in skeleton.joints:
            x, y = tuple(map(int, joint))
            if x < 0 or y < 0:
                continue
            cv2.putText(img, f'{skeleton.id}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 5, text_color, thickness)
            break


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[fill_row_coords, fill_col_coords] = 255
    return mask


def bpoly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool_)
    mask[fill_row_coords, fill_col_coords] = True 
    return mask


def localPerspective(img, results):
    global pts1, pts2, debug, red, green, blue
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    if debug:
        cv2.imwrite('.\\output\\warpPerspective.png', dst)

    mask = poly2mask(pts2[:,1], pts2[:,0], img.shape[:2])
    if debug:
        cv2.imwrite('.\\output\\poly2mask.png', mask)

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img, img, mask = mask_inv)
    img2_fg = cv2.bitwise_and(dst, dst, mask = mask)
    dst = cv2.add(img1_bg, img2_fg)

    for i in range(4):  
      center_coordinates = ((int)(pts1[i][0]), (int)(pts1[i][1])) 
      cv2.circle(img, center_coordinates, radius, green, thickness)
      cv2.circle(dst, center_coordinates, radius, green, thickness)
      center_coordinates = ((int)(pts2[i][0]), (int)(pts2[i][1])) 
      cv2.circle(img, center_coordinates, radius, blue, thickness)
      cv2.circle(dst, center_coordinates, radius, blue, thickness)

    return dst


def blocalPerspective(img):
    global pts1, pts2, debug
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    if debug:
        cv2.imwrite('.\\output\\warpPerspective.png', dst)
    cond = bpoly2mask(pts2[:,1], pts2[:,0], img.shape[:2])
    dst_r, dst_g, dst_b = cv2.split(dst)
    img_r, img_g, img_b = cv2.split(dst)
    dst = np.where(cond, dst, img)
    return dst


def kLocalPerspective(img, kIndex, results, type, debug):
    global radius, red, green, blue

    n, _ = kIndex.shape
    height, width, _ = img.shape
    if type == 0 or type == 1:
        face_landmarks1 = results.multi_face_landmarks[0]
        face_landmarks2 = results.multi_face_landmarks[1]
    elif type == 2:
        face_landmarks1 = results.multi_face_landmarks[1]
        face_landmarks2 = results.multi_face_landmarks[0]
    else:
        return

    for i in range(n):
        p1, p2 = [], []
        for j in range(4):
            landmark1 = face_landmarks1.landmark[kIndex[i,j]]
            landmark2 = face_landmarks2.landmark[kIndex[i,j]]
            p1.append([landmark1.x*width, landmark1.y*height])
            p2.append([landmark2.x*width, landmark2.y*height])

        pts1 = np.float32(p1)
        pts2 = np.float32(p2)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst12 = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
        if type == 0:
            M = cv2.getPerspectiveTransform(pts2, pts1)
            dst21 = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        for j in range(4):  
            center_coordinates = ((int)(pts1[j][0]), (int)(pts1[j][1])) 
            cv2.circle(img, center_coordinates, radius, green, thickness)
            center_coordinates = ((int)(pts2[j][0]), (int)(pts2[j][1])) 
            cv2.circle(img, center_coordinates, radius, blue, thickness)

        mask2 = poly2mask(pts2[:,1], pts2[:,0], img.shape[:2])
        if type == 0:
            mask1 = poly2mask(pts1[:,1], pts1[:,0], img.shape[:2])

        if debug:
            cv2.imwrite('.\\output\\warpPerspective12' + str(i) + '.png', dst12)
            cv2.imwrite('.\\output\\poly2mask2' + str(i) + '.png', mask2)
            if type == 0:
                cv2.imwrite('.\\output\\warpPerspective21' + str(i) + '.png', dst21)
                cv2.imwrite('.\\output\\poly2mask1' + str(i) + '.png', mask1)

        mask2_inv = cv2.bitwise_not(mask2)
        img1_bg = cv2.bitwise_and(img, img, mask = mask2_inv)
        img2_fg = cv2.bitwise_and(dst12, dst12, mask = mask2)
        dst = cv2.add(img1_bg, img2_fg)

        if type == 0:
            mask1_inv = cv2.bitwise_not(mask1)
            img1_bg = cv2.bitwise_and(dst, dst, mask = mask1_inv)
            img2_fg = cv2.bitwise_and(dst21, dst21, mask = mask1)
            dst = cv2.add(img1_bg, img2_fg)

        img = dst.copy()

        if debug:
            cv2.imwrite('.\\output\\pannotated_image' + str(i) + '.png', dst)

    return dst

"""The 33 pose landmarks.
class PoseLandmark(enum.IntEnum):
  NOSE = 0
  LEFT_EYE_INNER = 1
  LEFT_EYE = 2
  LEFT_EYE_OUTER = 3
  RIGHT_EYE_INNER = 4
  RIGHT_EYE = 5
  RIGHT_EYE_OUTER = 6
  LEFT_EAR = 7
  RIGHT_EAR = 8
  MOUTH_LEFT = 9
  MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  LEFT_PINKY = 17
  RIGHT_PINKY = 18
  LEFT_INDEX = 19
  RIGHT_INDEX = 20
  LEFT_THUMB = 21
  RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32
"""