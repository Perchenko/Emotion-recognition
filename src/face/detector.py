import cv2
import ctypes
import numpy as np

class Detection(ctypes.Structure):
    _fields_ = [('box', ctypes.c_float * 4),
                ('landmarks', ctypes.c_float * 10),
                ('conf', ctypes.c_float)]

class Detector:
    
    
    def __init__(self, lib_path, det_thresh, nms_thresh):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.init_detector.restype = ctypes.c_void_p
        self.lib.detector_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_float]
        self.lib.detector_predict.restype = ctypes.POINTER(Detection)
        self.detector = self.lib.init_detector()
        self.det_thresh = det_thresh
        self.nms_thresh = ctypes.c_float(nms_thresh)
   
        
    def detect(self, image):
        image_data = image.ctypes.data_as(ctypes.c_char_p)
        
        size = ctypes.c_int(0)
        ptr_size = ctypes.pointer(size)
        
        dets = self.lib.detector_predict(self.detector, ptr_size, image_data, self.nms_thresh)
        
        bboxes, landmarks, confs = [], [], []
        
        for i in range(size.value):
            det = dets[i]
            if det.conf < self.det_thresh:
                continue
            bboxes.append([coord for coord in det.box])
            landmarks.append([coord for coord in det.landmarks])
            confs.append(det.conf)
        
        return np.array(bboxes).reshape(-1, 4), np.array(landmarks).reshape(-1, 5, 2), np.array(confs)


        
    