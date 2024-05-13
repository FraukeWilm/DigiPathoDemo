import openslide
import cv2
from shapely import geometry
from pathlib import Path
import json
import numpy as np
import random

class SlideContainer:
    def __init__(self, file: str,
                 annotation_file,
                 level: int = 0,
                 patch_size: int = 256):
        self.file = Path(file)
        self.slide = openslide.open_slide(file)
        self.down_factor = self.slide.level_downsamples[level]
        self._level = level
        self.patch_size = patch_size
        with open(annotation_file) as f:
            data = json.load(f)
        self.classes = dict(zip([cat["name"] for cat in data["categories"]],[cat["id"] for cat in data["categories"]]))
        image_id = [i["id"] for i in data["images"] if i["file_name"] ==  self.file.name][0]
        self.annotations = [anno for anno in data['annotations'] if anno["image_id"] == image_id]
        self.labels = set([poly["category_id"] for poly in self.annotations])

    @property
    def slide_shape(self):
        return self.slide.level_dimensions[self._level]

    def get_slide_map(self, factor=100):
        thumb = np.array(self.slide.get_thumbnail((self.slide.dimensions[0]/factor, self.slide.dimensions[1]/factor)))[:,:,:3]
        map = -1*np.ones(shape=(thumb.shape[0], thumb.shape[1]), dtype=np.int8)
        return map, thumb

    def get_patch(self, x: int = 0, y: int = 0):
        patch = np.array(self.slide.read_region(location=(int(x * self.down_factor), int(y * self.down_factor)),
                                               level=self._level, size=(self.patch_size, self.patch_size)))
        patch[patch[:, :, -1] == 0] = [255, 255, 255, 0]
        return patch[:,:,:3]

    def get_new_train_coordinates(self):
        pass

    def __str__(self):
        return str(self.file)
    

class SegmentationContainer(SlideContainer):
    def __init__(self, file: str, annotation_file, level: int = 0, patch_size: int = 256):
        super().__init__(file, annotation_file, level, patch_size)
        self.white = 235
        self._label_dict = {'Bg': 0, 'Bone': 1, 'Cartilage': 1, 'Dermis': 1, 'Epidermis': 1, 'Subcutis': 1,
                    'Inflamm/Necrosis': 1, 'Melanoma': 2, 'Plasmacytoma': 2, 'Mast Cell Tumor': 2, 'PNST': 2,
                    'SCC': 2, 'Trichoblastoma': 2, 'Histiocytoma': 2}
    
    def get_slide_map(self, factor=100):
        map, thumb = super().get_slide_map(factor)
        inv_map = {v: k for k, v in self.classes.items()}

        for poly in self.annotations:
            coordinates = np.array(poly['segmentation']).reshape((-1,2))/ factor
            label = self._label_dict[inv_map[poly["category_id"]]]
            cv2.drawContours(map, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        white_mask = cv2.cvtColor(thumb,cv2.COLOR_RGB2GRAY) > self.white
        excluded = (map == -1)
        map[np.logical_and(white_mask, excluded)] = 0
        return map, thumb
    
    def get_y_patch(self, x: int = 0, y: int = 0):
        y_patch = -1*np.ones(shape=(self.patch_size, self.patch_size), dtype=np.int8)
        inv_map = {v: k for k, v in self.classes.items()}

        for poly in self.annotations:
            coordinates = np.array(poly['segmentation']).reshape((-1,2))/ self.down_factor
            coordinates = coordinates - (x, y)
            label = self._label_dict[inv_map[poly["category_id"]]]
            cv2.drawContours(y_patch, [coordinates.reshape((-1, 1, 2)).astype(int)], -1, label, -1)

        white_mask = cv2.cvtColor(self.get_patch(x,y),cv2.COLOR_RGB2GRAY) > self.white
        excluded = (y_patch == -1)
        y_patch[np.logical_and(white_mask, excluded)] = 0
        return y_patch
    
    def get_new_train_coordinates(self):
        # default sampling method
        label = random.choices(list(self.labels))[0]
        found = False
        while not found:
            iter = 0
            polygon = np.random.choice([poly for poly in self.annotations if poly["category_id"] == label])
            coordinates = np.array(polygon['segmentation']).reshape((-1, 2))
            minx, miny, xrange, yrange = polygon["bbox"]
            while iter < 25 and not found:
                iter += 1
                pnt = geometry.Point(np.random.uniform(minx, minx + xrange), np.random.uniform(miny, miny + yrange))
                if geometry.Polygon(coordinates).contains(pnt):
                    xmin = pnt.x // self.down_factor - self.patch_size / 2
                    ymin = pnt.y // self.down_factor - self.patch_size / 2
                    found = True
        return xmin, ymin


class DetectionContainer(SlideContainer):
    def __init__(self, file: str, annotation_file, level: int = 0, patch_size: int = 256):
        super().__init__(file, annotation_file, level, patch_size)
        self._label_dict = {'mitotic figure': 1, 'not mitotic figure': 0}

    def get_slide_map(self, factor=10):
        map, thumb = super().get_slide_map(factor)
        inv_map = {v: k for k, v in self.classes.items()}

        for box in self.annotations:
            coordinates = np.array(box['bbox'])/ factor
            label = self._label_dict[inv_map[box["category_id"]]]
            x_c = int((coordinates[0]+coordinates[2])/2)
            y_c = int((coordinates[1]+coordinates[3])/2)
            cv2.circle(map, (x_c, y_c), 5, label, -1)
        return map, thumb
    
    def get_bboxes(self, x: int = 0, y: int = 0):
        inv_map = {v: k for k, v in self.classes.items()}
        bboxes = []
        labels = []

        for box in self.annotations:
            coordinates = np.array(box['bbox'])/ self.down_factor
            coordinates = coordinates - (x, y, x, y)
            if any([c in range(0, self.patch_size) for c in coordinates[::2]]) and any([c in range(0, self.patch_size) for c in coordinates[1::2]]):
                coordinates = [c.clip(0, self.patch_size) for c in coordinates]
                if coordinates[0] != coordinates[2] and coordinates[1] != coordinates[3]:
                    bboxes.append(coordinates)
                    labels.append(self._label_dict[inv_map[box["category_id"]]])
        
        bboxes = np.array([box for box in bboxes]) if len(np.array(bboxes).shape) == 1 else np.array(bboxes)
        bboxes = bboxes.reshape((-1, 4))
        labels = np.array(labels)
        return bboxes, labels
    
    def get_new_train_coordinates(self):
        # default sampling method
        anno = random.choices(list(self.annotations))[0]
        coordinates = np.array(anno['bbox'])/ self.down_factor
        xmin = (coordinates[0] + coordinates[2])//2 - random.randint(0, self.patch_size)
        ymin = (coordinates[1] + coordinates[3])//2 - random.randint(0, self.patch_size)
        xmin = min(max(0, xmin), self.slide_shape[0] - self.patch_size)
        ymin = min(max(0, ymin), self.slide_shape[1] - self.patch_size)
        return xmin, ymin