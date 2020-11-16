import cv2
import numpy as np
from face.detector import Detector
from face.encoder import Encoder
from face.face_align import norm_crop
from face.resizer import Resizer


class FaceRecognizer:
    
    
    def __init__(self, 
                 det_lib_path='tensorrtx/retinaface/build/libretina_r50.so', 
                 enc_lib_path='tensorrtx/arcface/build/libarcface-r50.so', 
                 config_path='config', 
                 det_thresh=0.9, 
                 nms_thresh=0.4, 
                 rec_thresh=0.3, 
                 keep_aspect_ratio=True):
        self.rec_thresh = rec_thresh
        self.detector = Detector(det_lib_path, det_thresh, nms_thresh)
        self.encoder = Encoder(enc_lib_path)
        
        with open(config_path) as f:
            height = int(''.join([c if c.isdigit() else '' for c in f.readline()]))
            width = int(''.join([c if c.isdigit() else '' for c in f.readline()]))
            
        self.resizer = Resizer((height, width), keep_aspect_ratio)
        
    
    def encode(self, image, landmarks):
        embeddings = []
        for i in range(landmarks.shape[0]):
            aligned_face_image = norm_crop(image, landmarks[i], mode='arcface')
            embedding = self.encoder.get_embedding(aligned_face_image).flatten()
            normed_embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(normed_embedding)
        embeddings = np.array(embeddings).reshape(-1, 512)
        return embeddings
        
    
    def detect(self, image, min_height=None, min_width=None, only_full_face=False):
        original_size = image.shape[:2]
        image = self.resizer.resize_image(image)
        bboxes, landmarks, confs = self.detector.detect(image)
        bboxes, landmarks = self.resizer.resize_bboxes_landmarks(bboxes, landmarks, original_size)
        bboxes = bboxes.round().astype(int)
        bboxes[bboxes < 0] = 0
        landmarks = landmarks.round().astype(int)
        if bboxes.size:
            arg_bboxes = self.filter_bboxes(bboxes, min_height, min_width)
            bboxes, landmarks, confs = bboxes[arg_bboxes], landmarks[arg_bboxes], confs[arg_bboxes]
        if bboxes.size:
            if only_full_face:
                arg_landmarks = np.array([self.is_full_face(x) for x in landmarks])
                bboxes, landmarks, confs = bboxes[arg_landmarks], landmarks[arg_landmarks], confs[arg_landmarks]          
        return bboxes, landmarks, confs
    
    
    @staticmethod
    def filter_bboxes(bboxes, min_height=None, min_width=None):
        if min_height is not None and min_width is not None:
            return (bboxes[:, 3] - bboxes[:, 1] > min_height) & (bboxes[:, 2] - bboxes[:, 0] > min_width)
        elif min_height is not None and min_width is None:
            return bboxes[:, 3] - bboxes[:, 1] > min_height
        elif min_height is None and min_width is not None:
            return bboxes[:, 2] - bboxes[:, 0] > min_width
        return np.ones(bboxes.shape[0], dtype=bool)
    
    
    @staticmethod
    def is_full_face(landmarks):
        right_eye, left_eye, nose, right_mouth_corner, left_mouth_corner = landmarks
        if right_eye[0] > left_eye[0]:
            return False
        if right_mouth_corner[0] > left_mouth_corner[0]:
            return False
        if nose[0] > left_eye[0] and nose[0] > left_mouth_corner[0]:
            return False
        if nose[0] < right_eye[0] and nose[0] < right_mouth_corner[0]:
            return False
        return True
    
    
#     def identify(self, embedding):
#         max_matches = 0
#         most_similar = None
#         for name, embeddings in self.database.items():
#             matches = np.sum(self.cosine_similarity(embedding, embeddings) > self.rec_thresh)
#             if matches > max_matches:
#                 max_matches = matches
#                 most_similar = name
#         return most_similar
    
    
    def identify(self, embeddings):
        res = self.cosine_similarity(embeddings, self.database)
        retval = []
        for i in range(res.shape[0]):
            argmax = res[i].argmax()
            if res[i][argmax] > self.rec_thresh:
                retval.append(self.names[argmax])
            else:
                retval.append(None)
        return retval
    
    
    def extract_faces(self, image, bboxes):
        faces = []
        for x1, y1, x2, y2 in bboxes:
            faces.append(image[y1:y2, x1:x2])
        return faces
    
    
#     def upload_database(self, face_database):
#         self.database = face_database.read()
        
        
    def upload_database(self, face_database):
        self.database = []
        self.names = []
        for name, embeddings in face_database.read().items():
            self.names.append(name)
            self.database.append(embeddings.mean(axis=0))
        self.database = np.array(self.database)
        
        
    @staticmethod
    def cosine_similarity(normalized_x, normalized_y):
        return np.dot(normalized_x, normalized_y.T)
    
    
    @staticmethod
    def annotate(image, bboxes, landmarks=None, names=None,  color=(150, 0, 150)):
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            if landmarks is not None:
                for x, y in landmarks[i]:
                    cv2.circle(image, (x, y), 2, color, -1)
            if names is not None:
                if names[i] is not None:
                    cv2.putText(image, names[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return image




#     def recognize_v2(self, embedding):
#         max_similarity = self.rec_thresh
#         name = None
#         for key, embeddings in self.database.items():
#             if cosine_similarity(embedding, embeddings.mean(axis=0)) > max_similarity:
#                 max_similarity = cosine_similarity(embedding, embeddings)
#                 name = key + str(max_similarity)
#         return name
    

            

# def euclidean_square_distance(x, y):
#     if x.ndim == 2:
#         if x.shape[0] == 1 or x.shape[1] == 1:
#             x = x.flatten()
#     if y.ndim == 2:
#         if y.shape[0] == 1 or y.shape[1] == 1:
#             y = y.flatten()
#     if x.ndim == 1 or y.ndim == 1:
#         return np.sum(np.square(x - y), axis=-1)
#     return np.sum(np.square(np.stack([x[i] - y for i in range(len(x))])), axis=-1)


# def euclidean_similarity(x, y):
#     return 1 / (1 + np.sqrt(euclidean_square_distance(x, y)))


# def manhattan_distance(x, y):
#     if x.ndim == 2:
#         if x.shape[0] == 1 or x.shape[1] == 1:
#             x = x.flatten()
#     if y.ndim == 2:
#         if y.shape[0] == 1 or y.shape[1] == 1:
#             y = y.flatten()
#     if x.ndim == 1 or y.ndim == 1:
#         return np.sum(np.abs(x - y), axis=-1)
#     return np.sum(np.abs(np.stack([x[i] - y for i in range(len(x))])), axis=-1)


# def manhattan_similarity(x, y):
#     return 1 / (1 + manhattan_distance(x, y))
