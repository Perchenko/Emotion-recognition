import cv2
import numpy as np


class Resizer:
    
    def __init__(self, size, keep_aspect_ratio=True, color=(127, 127, 127)):
        self.height, self.width = size[:2]
        self.keep_aspect_ratio = keep_aspect_ratio
        self.color = color
            
        
    def resize_image(self, image):
        if self.keep_aspect_ratio:
            _, temp_size, border = self.calculate_ratio_temp_size_border(image.shape[:2])
            image = cv2.resize(image, temp_size[::-1])
            image = cv2.copyMakeBorder(image, *border, cv2.BORDER_CONSTANT, value=self.color)
            return image
        else:
            image = cv2.resize(image, (self.width, self.height))
            return image
        
        
    def resize_bboxes_landmarks(self, bboxes, landmarks, original_size):
        if self.keep_aspect_ratio:
            ratio, _, (top, _, left, _) = self.calculate_ratio_temp_size_border(original_size)
            bboxes[:, [0, 2]] -= left
            bboxes[:, [1, 3]] -= top
            bboxes /= ratio
            landmarks[:, :, 0] -= left
            landmarks[:, :, 1] -= top
            landmarks /= ratio
            return bboxes, landmarks
        else:
            height_ratio, width_ratio = self.height / original_size[0], self.width / original_size[1]
            bboxes[:, [0, 2]] /= width_ratio
            bboxes[:, [1, 3]] /= height_ratio
            landmarks[:, :, 0] /= width_ratio
            landmarks[:, :, 1] /= height_ratio
            return bboxes, landmarks
        
        
    def calculate_ratio_temp_size_border(self, original_size):
        ratio = min(self.height / original_size[0], self.width / original_size[1])
        temp_size = round(original_size[0] * ratio), round(original_size[1] * ratio)
        dy, dx = self.height - temp_size[0], self.width - temp_size[1]
        top, left = dy // 2, dx // 2
        bottom, right = dy - top, dx - left
        border = [top, bottom, left, right]
        return ratio, temp_size, border