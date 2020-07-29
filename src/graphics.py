import cv2
import numpy as np
from math import cos, sin, pi

class DetectionGraphics:

    def __init__(self, frame, stream_graphics):

        self.frame = frame
        self.stream_graphics = stream_graphics

        self.prev_w = 600
        self.prev_h = 600
        
        self.scale = 50
        self.focal_len = 950.0
        self.dtype='float32'
        
        self.axis_length = 0.5
        
        self.color_head = (90, 10, 250)
        self.color_eye = (150, 30, 10)
        self.color_gaze = (255, 150, 200)
        
        self.z_axis = (255, 0, 0)
        self.x_axis = (0, 0, 255) 
        self.y_axis = (0, 255, 0)  
        self.thick = 2

    def face_detection(self, face_coords):
        cv2.rectangle(self.frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), self.color_head, self.thick)
    
    def eyes_detection(self, eyes_coord, face_coords, crop_face_shape_1):
        eye_width = int(crop_face_shape_1 / 10 + 5)
        self.eye_detection(eyes_coord[:2], face_coords, eye_width)
        self.eye_detection(eyes_coord[2:4], face_coords, eye_width)


    def eye_detection(self, eyes_coord, face_coords, eye_width):

        xmin  = eyes_coord[0] + face_coords[0] - eye_width
        xmax = eyes_coord[0] + face_coords[0] + eye_width
        ymin = eyes_coord[1] + face_coords[1] - eye_width
        ymax = eyes_coord[1] + face_coords[1] + eye_width

        cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), self.color_eye, self.thick)

    def head_pose_estimation(self, head_pose_angles,face_coords,crop_face):   
    

        yaw=head_pose_angles[0]
        pitch=head_pose_angles[1]
        roll=head_pose_angles[2]
        

        sinYaw = sin (yaw * pi / 180)
        sinPitch = sin(pitch * pi / 180)
        sinRoll = sin(roll * pi / 180)
        
        cosYaw = cos (yaw * pi / 180)
        cosPitch = cos(pitch * pi / 180)
        cosRoll = cos(roll * pi / 180)
        
        if self.stream_graphics: 
            x_stream_graphics = -1
        else:
            x_stream_graphics = 1
        xaxis = np.array(([x_stream_graphics * self.scale, 0, 0]), dtype=self.dtype).reshape(3, 1)
        yaxis = np.array(([0, -1 * self.scale, 0]), dtype=self.dtype).reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * self.scale]), dtype=self.dtype).reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * self.scale]), dtype=self.dtype).reshape(3, 1)

        o = np.array(([0, 0, 0]), dtype=self.dtype).reshape(3, 1)
        o[2] = self.focal_len

        Rx = np.array([[1, 0, 0], [0, cosPitch, -sinPitch], [0, sinPitch, cosPitch]])
        Ry = np.array([[cosYaw, 0, -sinYaw],  [0, 1, 0], [sinYaw, 0, cosYaw]])
        Rz = np.array([[cosRoll, -sinRoll, 0], [sinRoll, cosRoll, 0], [0, 0, 1]])

        R = Rz @ Ry @ Rx

        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o

        axis_system = (xaxis, yaxis, zaxis, zaxis1)
        
        xaxis, yaxis, zaxis, zaxis1 = axis_system
        
        xa, ya = (int(face_coords[0] + crop_face.shape[1] / 2), int(face_coords[1] + crop_face.shape[0] / 2))

        xp_x = (xaxis[0] / xaxis[2] * self.focal_len) + xa
        xp_y = (xaxis[1] / xaxis[2] * self.focal_len) + ya
        xp = (int(xp_x), int(xp_y))
        cv2.line(self.frame, (xa, ya), xp, self.x_axis, self.thick)  

        yp_x = (yaxis[0] / yaxis[2] * self.focal_len) + xa
        yp_y = (yaxis[1] / yaxis[2] * self.focal_len) + ya
        yp = (int(yp_x), int(yp_y))
        cv2.line(self.frame, (xa, ya), yp, self.y_axis, self.thick)  

        zp1_x = (zaxis1[0] / zaxis1[2] * self.focal_len) + xa
        zp1_y = (zaxis1[1] / zaxis1[2] * self.focal_len) + ya
        zp1 = (int(zp1_x), int(zp1_y))
        

        zp_x = (zaxis[0] / zaxis[2] * self.focal_len) + xa
        zp_y = (zaxis[1] / zaxis[2] * self.focal_len) + ya
        zp = (int(zp_x), int(zp_y))

        cv2.line(self.frame, zp1, zp, self.z_axis, self.thick) 
        cv2.circle(self.frame, zp, 3, self.z_axis, self.thick) 

    def eyes_gaze_estimation(self, eyes_coord, face_coords, gaze, crop_face):
   
        gaze_length = (self.axis_length * crop_face.shape[1], self.axis_length * crop_face.shape[0])

        self.eye_gaze_estimation(eyes_coord[:2], face_coords, gaze, gaze_length)
        self.eye_gaze_estimation(eyes_coord[2:4], face_coords, gaze, gaze_length)

    def eye_gaze_estimation(self, eyes_coords, face_coords, gaze, gaze_length):

        eye_xcenter = int(eyes_coords[0] + face_coords[0])
        eye_ycenter = int(eyes_coords[1] + face_coords[1])

        gaze_x = (gaze[0] * gaze_length[0]) + eye_xcenter
        gaze_y = (-gaze[1] * gaze_length[1]) + eye_ycenter
        gp = (int(gaze_x), int(gaze_y))

        cv2.arrowedLine(self.frame, (eye_xcenter, eye_ycenter), gp, self.color_gaze, self.thick)

