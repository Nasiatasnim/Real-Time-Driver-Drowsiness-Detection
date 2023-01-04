# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:17:51 2021

@author:
"""
import cv2 as cv
import mediapipe as mp

def draw_landmarks(image, outputs, land_mark, color):
    height, width =image.shape[:2]
             
    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        
        point_scale = ((int)(point.x * width), (int)(point.y*height))
        
        cv.circle(image, point_scale, 2, color, 1)

face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0,255,0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0,0,255), thickness=1, circle_radius=1)


STATIC_IMAGE = False
MAX_NO_FACES = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)
COLOR_GREEN = (0,255,0)

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
       185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

FACE=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
       377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces= MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)

path = "C:/MEDIA_PIPE/VID-20211028-WA0002.mp4"

capture = cv.VideoCapture(path)

while True:
    result, image = capture.read()
    
    if result:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        outputs = face_model.process(image_rgb)

        if outputs.multi_face_landmarks:
            
            draw_landmarks(image, FACE, outputs, COLOR_GREEN)
                
                
            draw_landmarks(image, LIPS, outputs, COLOR_RED)
                
            draw_landmarks(image, RIGHT_EYEBROW, outputs, COLOR_BLUE)
            draw_landmarks(image, LEFT_EYEBROW, outputs, COLOR_BLUE)
            
            '''
            height, width =image.shape[:2]
             
            for face in FACE:
                point = outputs.multi_face_landmarks[0].landmark[face]
                
                point_scale = ((int)(point.x * width), (int)(point.y*height))
                
                cv.circle(image, point_scale, 2, (0,255,0), 1)
                
                
                draw_utils.draw_landmarks(image = image,
                                          landmark_list= face,
                                          connections = face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=landmark_style,
                                          connection_drawing_spec=connection_style)
                
                '''
        cv.imshow("FACE MESH", image)
        if cv.waitKey(30) & 255 == 27:
            break
        
        
capture.release()
cv.destroyAllWindows()
