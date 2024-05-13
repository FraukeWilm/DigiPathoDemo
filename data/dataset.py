from data.slide_container import SegmentationContainer, DetectionContainer
from torch.utils.data import Dataset
import torch
import os

class WSIDataset(Dataset):
    def __init__(self, image_path, anno_file_path, num_per_slide, patch_size, ds_level, transforms=None):
        self._image_path = image_path
        self._img_l = [file for file in os.listdir(image_path)]
        self._anno_file_path = anno_file_path
        self.num_per_slide = int(num_per_slide)
        self._samples = int(num_per_slide)*len(self._img_l)
        self._patch_size = patch_size
        self._ds_level = ds_level
        self._transforms = transforms
        self.slide_objs = {}

    def __len__(self):
        if len(self._img_l) > 0:
            return self._samples
        else:
            return 0
        
    def __slide_object__(self, slide_path):
        pass

    def __getitem__(self, idx):
        # sample wsi
        wsi_idx = idx//self.num_per_slide
        slide_path = self._img_l[wsi_idx]
        # create openslide object
        if self.slide_objs.get(slide_path) is None:
            self.slide_objs[slide_path] = self.__slide_object__(slide_path)
        slide_obj = self.slide_objs[slide_path]
        return slide_obj

    def __getoverview__(self, idx):
        # sample wsi
        wsi_idx = idx//self.num_per_slide
        slide_path = self._img_l[wsi_idx]
        # create openslide object
        if self.slide_objs.get(slide_path) is None:
            self.slide_objs[slide_path] = self.__slide_object__(slide_path)
        slide_obj = self.slide_objs[slide_path]
        map, thumb = slide_obj.get_slide_map()
        return map, thumb
    

class SegmentationDataset(WSIDataset):
    def __init__(self, image_path, anno_file_path, num_per_slide, patch_size, ds_level, transforms=None):
        super().__init__(image_path, anno_file_path, num_per_slide, patch_size, ds_level, transforms)
        self._img_l = [file for file in self._img_l if file.__contains__('cs2')]
        self._samples = int(num_per_slide)*len(self._img_l)
    
    def __slide_object__(self, slide_path):
        return SegmentationContainer(os.path.join(self._image_path, slide_path), self._anno_file_path, self._ds_level, self._patch_size)
    
    def __getitem__(self, idx):
        slide_obj =  super().__getitem__(idx)
        x, y = slide_obj.get_new_train_coordinates()
        image = slide_obj.get_patch(x, y)
        label = slide_obj.get_y_patch(x, y)
        if self._transforms:
            image = self._transforms(image)/255.
            label = self._transforms(label)
        return (image, label)



class DetectionDataset(WSIDataset):
    def __init__(self, image_path, anno_file_path, num_per_slide, patch_size, ds_level, transforms=None):
        super().__init__(image_path, anno_file_path, num_per_slide, patch_size, ds_level, transforms)
    
    def __slide_object__(self, slide_path):
        return DetectionContainer(os.path.join(self._image_path, slide_path), self._anno_file_path, self._ds_level, self._patch_size)
    
    """
    def __getitem__(self, idx):
        slide_obj =  super().__getitem__(idx)
        x, y = slide_obj.get_new_train_coordinates()
        image = slide_obj.get_patch(x, y)
        bboxes, labels = slide_obj.get_bboxes(x, y)
        if self._transforms:
            image = self._transforms(image)/255.

        targets = {
            'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
        }
        return image, targets
    """

    def __getitem__(self, idx):
        slide_obj =  super().__getitem__(idx)
        x, y = slide_obj.get_new_train_coordinates()
        image = slide_obj.get_patch(x, y)/255.
        boxes, records = slide_obj.get_bboxes(x, y)
        if self._transforms:
            image = self._transforms(image)
            image = torch.as_tensor(image, dtype=torch.float32)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        #labels = torch.as_tensor(records + 1, dtype=torch.int64)
        # combine mitotic figures and impostors so there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd

        return image, target, self._img_l[idx] 
        #return image, target


