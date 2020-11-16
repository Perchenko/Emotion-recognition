import cv2
import ctypes
import numpy as np


class Encoder:
    
    
    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.init_encoder.restype = ctypes.c_void_p
        self.lib.encoder_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.encoder_encode.restype = ctypes.POINTER(ctypes.c_float * 512)
        self.encoder = self.lib.init_encoder()
   
        
    def get_embedding(self, image):
        image_data = image.ctypes.data_as(ctypes.c_char_p)
        embedding = self.lib.encoder_encode(self.encoder, image_data)
        embedding = [item for item in embedding.contents]
        return np.array(embedding)