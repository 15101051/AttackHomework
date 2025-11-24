from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms


class Lesson(Dataset):
    def __init__(self, images_path='./resources/lesson/images/',
                 label_path='./resources/lesson/label.txt'):
        self.labels = {}
        with open(label_path) as f:
            for line in f:
                name, label = line.strip().split()
                self.labels[name] = int(label)
        self.images = os.listdir(images_path)
        self.images.sort()
        self.images_path = images_path
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name = self.images[item]
        x = Image.open(os.path.join(self.images_path, name))
        y = self.labels[name]
        return name, self.transforms(x), y


def get_lesson_loader(batch_size=64,
                      num_workers=8,
                      pin_memory=True,
                      shuffle=False,
                      **kwargs,
                      ):
    set = Lesson(**kwargs)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                        shuffle=shuffle)
    return loader
