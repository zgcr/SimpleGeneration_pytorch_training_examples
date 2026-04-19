import os
import collections
import cv2
import json
import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset


class ImageEditPairDataset(Dataset):

    def __init__(self,
                 root_dir,
                 dataset_name=[
                     'edit_dataset_v1',
                 ],
                 set_name='val',
                 transform=None):
        assert set_name in ['train', 'val'], 'Wrong set name!'

        self.source_image_name_list = []
        self.source_image_path_dict = collections.OrderedDict()
        self.edited_image_path_dict = collections.OrderedDict()
        self.image_edit_text_pair_dict = collections.OrderedDict()
        for per_dataset_name in tqdm(dataset_name):
            per_dataset_image_dir = os.path.join(root_dir, per_dataset_name,
                                                 set_name)
            per_dataset_text_json_path = os.path.join(
                root_dir, per_dataset_name,
                f'{per_dataset_name}_{set_name}.json')
            with open(per_dataset_text_json_path, encoding='utf-8') as f:
                per_dataset_text_dict = json.load(f)

            for per_source_image_name in sorted(
                    os.listdir(per_dataset_image_dir)):
                if per_source_image_name.endswith(
                        '.jpg'
                ) and not per_source_image_name.endswith('_edit.jpg'):
                    per_source_image_path = os.path.join(
                        per_dataset_image_dir, per_source_image_name)

                    per_edited_image_name = per_source_image_name.replace(
                        '.jpg', '_edit.jpg')
                    per_edited_image_path = os.path.join(
                        per_dataset_image_dir, per_edited_image_name)
                    if os.path.exists(
                            per_source_image_path) and os.path.exists(
                                per_edited_image_path):
                        self.source_image_name_list.append(
                            per_source_image_name)
                        self.source_image_path_dict[
                            per_source_image_name] = per_source_image_path
                        self.edited_image_path_dict[
                            per_source_image_name] = per_edited_image_path
                        self.image_edit_text_pair_dict[
                            per_source_image_name] = per_dataset_text_dict[
                                per_source_image_name]['web_caption']

        assert len(self.source_image_name_list) == len(
            self.edited_image_path_dict) == len(
                self.source_image_path_dict) == len(
                    self.image_edit_text_pair_dict)

        self.transform = transform

        print(f'Dataset Size:{len(self.source_image_name_list)}')

    def __len__(self):
        return len(self.source_image_name_list)

    def __getitem__(self, idx):
        source_image_path = self.source_image_path_dict[
            self.source_image_name_list[idx]]
        edited_image_path = self.edited_image_path_dict[
            self.source_image_name_list[idx]]

        source_image = self.load_source_image(idx)
        edited_image = self.load_edited_image(idx)
        text = self.load_text(idx)

        sample = {
            'source_image_path': source_image_path,
            'edited_image_path': edited_image_path,
            'source_image': source_image,
            'edited_image': edited_image,
            'text': text,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_source_image(self, idx):
        source_image = cv2.imdecode(
            np.fromfile(
                self.source_image_path_dict[self.source_image_name_list[idx]],
                dtype=np.uint8), cv2.IMREAD_COLOR)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

        return source_image.astype(np.float32)

    def load_edited_image(self, idx):
        edited_image = cv2.imdecode(
            np.fromfile(
                self.edited_image_path_dict[self.source_image_name_list[idx]],
                dtype=np.uint8), cv2.IMREAD_COLOR)
        edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)

        return edited_image.astype(np.float32)

    def load_text(self, idx):
        text = self.image_edit_text_pair_dict[self.source_image_name_list[idx]]

        return text


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import image_edit_pair_dataset_path

    from tqdm import tqdm

    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    from SimpleGeneration.universal_image_edit.image_edit_common import Opencv2PIL, PIL2Opencv, TorchResize, ImageEditPairCollater

    image_edit_pair_dataset = ImageEditPairDataset(
        root_dir=image_edit_pair_dataset_path,
        dataset_name=[
            'edit_dataset_v1',
        ],
        set_name='val',
        transform=transforms.Compose([
            Opencv2PIL(),
            TorchResize(resize=256, interpolation=InterpolationMode.BICUBIC),
            PIL2Opencv(),
        ]))

    count = 0
    for per_sample in tqdm(image_edit_pair_dataset):
        print('1111', per_sample['source_image_path'])
        print('2222', per_sample['edited_image_path'])

        print('3333', per_sample['source_image'].shape,
              per_sample['edited_image'].shape)
        print('4444', type(per_sample['edited_image']),
              type(per_sample['source_image']), type(per_sample['text']))

        temp_dir = './temp1'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        source_image = np.ascontiguousarray(per_sample['source_image'],
                                            dtype=np.uint8)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)

        edited_image = np.ascontiguousarray(per_sample['edited_image'],
                                            dtype=np.uint8)
        edited_image = cv2.cvtColor(edited_image, cv2.COLOR_RGB2BGR)

        text = per_sample['text']
        print('4444', text)

        cv2.imencode('.jpg', source_image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_source.jpg'))
        cv2.imencode('.jpg', edited_image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_edit.jpg'))

        if count < 2:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = ImageEditPairCollater()
    train_loader = DataLoader(image_edit_pair_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        source_images, edited_images, texts = data['source_image'], data[
            'edited_image'], data['text']
        print('1111', source_images.shape, edited_images.shape, len(texts))
        print('2222', source_images.dtype, edited_images.dtype)

        for per_image_text in texts:
            print(per_image_text)

        if count < 2:
            count += 1
        else:
            break
