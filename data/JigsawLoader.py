import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import torch
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = random.sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


class JigsawIADataset(data.Dataset):
    def __init__(self, names, labels, data_path, img_transformer=None):
        self.data_path = data_path
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index] - 1)

    def __len__(self):
        return len(self.names)

class JigsawTestIADataset(JigsawIADataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index] - 1)


class JigsawTestIADataset_idx(JigsawIADataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index] - 1), index


class DistillCLIPDataset(data.Dataset):
    def __init__(self, names, labels, data_path, img_transformer=None, clip_transformer=None):
        self.data_path = data_path
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self._CLIP_transformer = clip_transformer

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), self._CLIP_transformer(img), int(self.labels[index] - 1)


    def __len__(self):
        return len(self.names)

class DistillCLIPTestDataset(DistillCLIPDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), self._CLIP_transformer(img), int(self.labels[index] - 1)


class BaseJsonDataset(data.Dataset):
    def __init__(self, confi_imag, confi_dis, mode='train', n_shot=None, transform=None):
        self.transform = transform
        # self.image_path = image_path
        # self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        self.shot_predict_list = []
        # txt_tar = open(json_path).readlines()
        # samples = []
        samples = confi_imag
        shot_predict = confi_dis
        # cls_val, shot_predict = torch.max(confi_dis, 1)
        self.shot_predict_list = shot_predict.cpu().numpy().tolist()
        # for line in txt_tar:
        #     # line=line.rstrip("\n")
        #     line_split = re.split(' ',line)
        #     samples.append(line_split)
        for s in samples:  # s:['Faces/image_0353.jpg', 0, 'face']
            self.image_list.append(s[0])
            # s[1] = s[1])
            self.label_list.append(s[1])
            # self.shot_predict_list.append(s[1])
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        pesu_label = self.shot_predict_list[idx]
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long(), torch.tensor(pesu_label), idx
