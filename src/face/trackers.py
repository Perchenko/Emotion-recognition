import cv2
import numpy as np
from face.sort import Sort


class OpticalFlowTracker:
    
    def __init__(self, frame, bboxes, names):
        # bboxes: np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...], dtype=int)
        # names: list(str)
        self.names = names
        bboxes = self.tlbr2tlwh(bboxes)
        self.trackers = cv2.MultiTracker_create()
        for x, y, w, h in bboxes:
            tracker = cv2.TrackerMedianFlow_create()
            self.trackers.add(tracker, frame, (x, y, w, h))
    
    def track(self, frame):
        ret, bboxes = self.trackers.update(frame)
        bboxes = np.array(bboxes).reshape(-1, 4).round().astype(int)
        bboxes = self.tlwh2tlbr(bboxes)
        return ret, bboxes, self.names
    
    @staticmethod
    def tlbr2tlwh(bboxes):
        bboxes = bboxes.copy()
        bboxes[:, [2, 3]] -= bboxes[:, [0, 1]]
        return bboxes
    
    @staticmethod
    def tlwh2tlbr(bboxes):
        bboxes = bboxes.copy()
        bboxes[:, [2, 3]] += bboxes[:, [0, 1]]
        return bboxes
    
# tracker.save("TrackerMedianFlow.json")
# fs = cv2.FileStorage("TrackerMedianFlow.json", cv2.FileStorage_READ)
# tracker.read(fs.getFirstTopLevelNode())
# fs.release()
#
# try tracking with multiprocessing.dummy


class KalmanFilterTracker:
    
    def __init__(self, bboxes, confs, names, max_age=20, min_hits=0, iou_threshold=0.25):
        # bboxes: np.array([[x1, y1, x2, y2], [x1, y1, x2, y2], ...], dtype=int)
        # confs: np.array([float])
        # names: list(str)
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        bboxes = np.hstack((bboxes, confs.reshape(-1, 1)))
        bboxes = self.tracker.update(bboxes)
        self.idx2name = dict()
        for idx, name in zip(bboxes[:, -1].astype(int).tolist(), names):
            self.idx2name[idx] = name    
        
    def track(self, bboxes, confs):
        bboxes = np.hstack((bboxes, confs.reshape(-1, 1)))
        bboxes = self.tracker.update(bboxes)
        names = []
        ret = True
        for idx in bboxes[:, -1].astype(int):
            if idx in self.idx2name:
                names.append(self.idx2name[idx])
            else:
                names.append(None)
                ret = False
        bboxes = bboxes[:, :-1].round().astype(int)
        return ret, bboxes, names