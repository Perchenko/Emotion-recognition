import os
import cv2
import numpy as np


class FaceDatabase:
    
    
    def __init__(self, database_path, verbose=True):
        self.database_path = database_path
        self.verbose = verbose
        
        
    def read(self):
        database = dict()
        for file_name in os.listdir(self.database_path):
            file_path = os.path.join(self.database_path, file_name)
            database[file_name.split('.')[0]] = self.read_file(file_path)
        return database
    
    
    def read_file(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(list(map(lambda x: float(x), line.split())))
        data = np.array(data)
        return data

    
    def add(self, path, face_recognizer):
        for dir_path in list(map(lambda x: os.path.join(path, x), os.listdir(path))):
            name = dir_path.split('/')[-1]
            embeddings = self.encode(dir_path, face_recognizer)
            output_path = os.path.join(self.database_path, dir_path.split('/')[-1] + '.txt')
            self.write_file(output_path, embeddings)
            
            
    def write_file(self, file_path, data):
        with open(file_path, 'a') as f:
            for line in data:
                line = list(map(lambda x: str(x), line))
                line = ' '.join(line) + '\n'
                f.write(line)    
        
            
    def encode(self, dir_path, face_recognizer):
        embeddings = []
        for image_name in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_name)
            embedding = self.encode_image(image_path, face_recognizer)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings
    
    
    def encode_image(self, image_path, face_recognizer):
        image = cv2.imread(image_path)
        _, landmarks, _ = face_recognizer.detect(image)
        embeddings = face_recognizer.encode(image, landmarks)
        if embeddings.shape[0] == 1:
            embedding = embeddings[0].tolist()
            return embedding
        else:
            if self.verbose:
                print('Warning: {} faces detected - {}'.format(embeddings.shape[0], image_path))
            return