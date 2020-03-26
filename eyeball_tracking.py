from __future__ import division
from collections import deque
from scipy.spatial import distance as dist
import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords
#imgpath = 'F:\\sriram\\opencv-master\\data\\haarcascades\\'
path2 = "F:\\opencv\\eye_ball tracking\\shape_predictor_68_face_landmarks.dat"   
detector = dlib.get_frontal_face_detector()
predector = dlib.shape_predictor(path2)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
pts = deque(maxlen=70)
def main(): 
    
    cap = cv2.VideoCapture(0)
  
    if cap.isOpened :
        ret, frame = cap.read()
    else:
         ret = False
    while(ret):
      ret, frame = cap.read()
       
      
       #dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
      frame =imutils.resize(frame, width = 400)
      image = frame.copy()
      img1=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
     

      dect = detector(img1, 1)
      if len(dect) >0:
          for p, k in enumerate(dect):
              shape = predector(img1, k)
              shape = shape_to_np(shape)
              left_eye = shape[lStart:lEnd]  
              right_eye = shape[rStart:rEnd]
              leftEyeCenter = left_eye.mean(axis=0).astype("int")
              #print((leftEyeCenter[0], leftEyeCenter[1]))
             # cv2.line(shape, left_eye[0], leftEyeCenter, 1, (0, 0, 255), 1)  
             # print(C)
             # print(right_eye)
              left_eye_hull = cv2.convexHull(left_eye)  
              right_eye_hull = cv2.convexHull(right_eye)
              #print(right_eye_hull)
              
              cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
              cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
              (x, y, w, h) = cv2.boundingRect(np.array([left_eye]))
              roi2 = image[y:y+h, x:x+w]
              roi2 = imutils.resize(roi2, width=75, inter=cv2.INTER_CUBIC)
              out = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
              #blur = cv2.GaussianBlur(out,(5,5),0)
              block_size = 131
              constant = 11
              th1 = cv2.adaptiveThreshold (out, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, constant)
         
         
              k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
              erosion2 = cv2.erode( th1, k, iterations  =2)
              dilatilon = cv2.dilate(erosion2, k, iterations  = 2)
              c1, contours, hie = cv2.findContours(dilatilon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             # cv2.drawContours(roi2, contours, -1, (0, 255, 0), -1)
              
              if imutils.is_cv2():
                  cnt = contours[0]
              else:
                  cnt = contours
                  
              if len(cnt) > 0: 
              
              
                 M = cv2.moments(c1)
                 if M['m00'] != 0:
                     cx = int(M['m10']/M['m00'])
                     cy = int(M['m01']/M['m00'])
                     c = max(cnt, key= cv2.contourArea)
                     ((x, y), radius) = cv2.minEnclosingCircle(c)
                     #cv2.circle(roi2, (leftEyeCenter[0], leftEyeCenter[1]),1,(0, 0, 255), -1)
                     if radius > 9 :
                      d = dist.euclidean(left_eye[0], (x, y))
                      #print(d)
                      f =  cv2.absdiff(c, d)
                      print(f)
                      pts.appendleft((cx, cy))
                      #cv2.circle(roi2, (cx, cy), int(radius),(0, 255, 255), -1)
                      cv2.circle(roi2, (int(x), int(y)), 3,(0, 0, 255), -1)
                     
              for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
	        #if pts[i - 1] is None or pts[i] is None:
                #                 continue
                    
			
 
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		                thickness = int(np.sqrt(int(70) / float(i + 1)) * 2.5)
		                cv2.line(roi2, pts[i - 1], pts[i], (0, 0, 255), 2)
		                				                                                 
				                                      

                     
              cv2.imshow('windowName1', roi2)
              
              cv2.imshow('windowName2', th1)             
            
             
              
              
  
             
      cv2.imshow('windowName', frame)
        
      if cv2.waitKey(1) == 27:
          break
        
    cv2.destroyAllWindows()
    cap.release()
if __name__ == "__main__":
    main()
               #Points 0 to 16 is the Jawline
#Points 17 to 21 is the Right Eyebrow
#Points 22 to 26 is the Left Eyebrow
#Points 27 to 35 is the Nose
#Points 36 to 41 is the Right Eye
##Points 42 to 47 is the Left Eye
#Points 48 to 60 is Outline of the Mouth
#Points 61 to 67 is the Inner line of the Mouth  
    
 
