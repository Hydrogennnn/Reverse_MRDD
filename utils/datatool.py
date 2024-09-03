


class EdgeMNISTDataset(torchvision.datasets.MNIST):
    """
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')

        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(self.targets)
        return [view0, view1], self.targets[idx]


class EdgeFMNISTDataset(torchvision.datasets.FashionMNIST):

    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img.numpy(), mode='L')

        # original-view transforms
        view0 = img
        # edge-view transforms
        view1 = edge_transformation(img)
        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return [view0, view1], self.targets[idx]


class COIL20Dataset(Dataset):

    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=2) -> None:

        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(20))
        self.train = train
        self.views = views
        self.to_pil = transforms.ToPILImage()
        X_train, X_test, y_train, y_test = coil(
            self.root, n_objs=20, n_views=self.views)
        if self.train:
            self.data = X_train
            self.targets = torch.from_numpy(y_train).long()
        else:
            self.data = X_test
            self.targets = torch.from_numpy(y_test).long()

    def __getitem__(self, index):
        views = [np.transpose(self.data[view, index, :], (1, 2, 0))
                 for view in range(self.views)]
        target = self.targets[index]

        views = [self.to_pil(v) for v in views]

        if self.transform:
            views = [self.transform(x) for x in views]

        if self.target_transform:
            target = self.target_transform(target)

        return views, target

    def __len__(self) -> int:
        return self.data.shape[1]


class COIL100Dataset(Dataset):

    def __init__(self, root: str, train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=2) -> None:

        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(range(100))
        self.train = train
        self.views = views
        self.to_pil = transforms.ToPILImage()
        X_train, X_test, y_train, y_test = coil(
            self.root, n_objs=100, n_views=self.views)
        if self.train:
            self.data = X_train
            self.targets = torch.from_numpy(y_train).long()
        else:
            self.data = X_test
            self.targets = torch.from_numpy(y_test).long()

    def __getitem__(self, index):
        views = [np.transpose(self.data[view, index, :], (1, 2, 0))
                 for view in range(self.views)]
        target = self.targets[index]

        views = [self.to_pil(v) for v in views]

        if self.transform:
            views = [self.transform(x) for x in views]

        if self.target_transform:
            target = self.target_transform(target)

        return views, target

    def __len__(self) -> int:
        return self.data.shape[1]


class MultiViewClothingDataset(Dataset):
    """
    **Note: Before using this dataset, you have to run the `generate_mvc_dataset` function.**

    Refers to: Kuan-Hsien Liu, Ting-Yen Chen, and Chu-Song Chen.
    MVC: A Dataset for View-Invariant Clothing Retrieval and Attribute Prediction, ACM ICMR 2016.
    Total: 161260 images. 10 classes.
    (In fact, I found that it has many fails when I downloaded them. So, subject to the actual number)
    The following the size of number is my actual number:
        2 views train size: 29706, test size: 7427
        3 views train size: 29104, test size: 7277
        4 views train size: 28903, test size: 7226
        5 views train size: 8080, test size: 2021
        6 views train size: 2263, test size: 566
    """

    def __init__(self, root: str = '/mnt/disk3/data/mvc-10', train: bool = True,
                 transform=None,
                 target_transform=None, download: bool = False, views=2) -> None:

        super().__init__()
        self.classes_name = {
            "Shirts & Tops": 0,
            "Coats & Outerwear": 1,
            "Pants": 2,
            "Dresses": 3,
            "Underwear & Intimates": 4,
            "Jeans": 5,
            "Sweaters": 6,
            "Swimwear": 7,
            "Sleepwear": 8,
            "Underwear": 9
        }
        self.target2class = {v: k for k, v in self.classes_name.items()}
        self.classes = [k for k, v in self.classes_name.items()]
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.views = views
        # image loader.
        self.loader = pil_loader

        self.indices = torch.load(os.path.join(
            self.root, f'{self.views}V-indices.pth'))
        self.data, self.targets = self.indices['train'] if self.train else self.indices['test']

    def __getitem__(self, index: int):
        try:
            raw_data = [self.loader(os.path.join(self.root, path))
                        for path in self.data[index]]
        except:
            print([os.path.join(self.root, path)
                   for path in self.data[index]])
            raise

        if self.transform:
            views_data = [self.transform(x) for x in raw_data]
        else:
            views_data = raw_data
        target = torch.tensor(self.classes_name[self.targets[index]]).long()
        if self.target_transform:
            target = self.target_transform(target)

        return views_data, target

    def __len__(self) -> int:
        return len(self.data)


class Office31(Dataset):
    """
    Before use our Office31, you should firstly run the `align_office31` function.
    After that, you will get the alignment dataset, train.json, and test.json files.
    Stats:
        Totoal number: (2817, 3) (2817,)
        Train set: (2253, 3) (2253, )
        Test set: (564, 3) (564, )
    """

    views_mapping = {
        'A': 'amazon/images',
        'D': 'dslr/images',
        'W': 'webcam/images'
    }

    classes = ['paper_notebook', 'desktop_computer', 'punchers', 'desk_lamp', 'tape_dispenser',
               'projector', 'calculator', 'file_cabinet', 'back_pack', 'stapler', 'ring_binder',
               'trash_can', 'printer', 'bike', 'mug', 'scissors', 'bike_helmet', 'mouse', 'bookcase',
               'pen', 'bottle', 'keyboard', 'phone', 'ruler', 'headphones', 'speaker', 'letter_tray',
               'monitor', 'mobile_phone', 'desk_chair', 'laptop_computer']

    def __init__(self, root='./data/Office31', train: bool = True,
                 transform=None, target_transform=None, download: bool = False, views=3) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.load_image_path(train)

    def load_image_path(self, train):
        import json
        if train:
            self.data = json.load(open(os.path.join(self.root, 'train.json')))
        else:
            self.data = json.load(open(os.path.join(self.root, 'test.json')))

    def __getitem__(self, index):
        a, d, w, target = self.data[index]
        view0 = pil_loader(os.path.join(self.root, a))
        view1 = pil_loader(os.path.join(self.root, d))
        view2 = pil_loader(os.path.join(self.root, w))
        target = torch.tensor(target).long()

        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)
            view2 = self.transform(view2)

        return [view0, view1, view2], target

    def __len__(self):
        return len(self.data)


class FFDataset(Dataset):
    """
    Base on FaceForensics++ dataset. Before using this dataset,
    you have to run the function `generate_deepfake_dataset()`.
    Two views shape: X -> (79257, 2) Y -> (79257,)
    Three views shape: X -> (49549, 3) Y -> (49549,)
    """

    classes = ['youtube', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

    def __init__(self, root='/mnt/disk3/data/DeepfakeBench/FaceForensics++', train: bool = True,
                 transform=None, target_transform=None, download: bool = False, views=3) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.views = views
        self.load_file()

    def load_file(self):
        if self.train:
            idx = torch.load(f"./data/ffdataset/view-{self.views}-train.idx")
            self.data, self.targets = idx['data'], idx['targets']
        else:
            idx = torch.load(f"./data/ffdataset/view-{self.views}-test.idx")
            self.data, self.targets = idx['data'], idx['targets']

    def __getitem__(self, index):
        view0, view1, target = self.data[index, 0], self.data[index, 1], self.targets[index]
        if self.views == 3:
            viwe2 = self.data[index, 2]
        view0 = pil_loader(os.path.join(self.root, view0))
        view1 = pil_loader(os.path.join(self.root, view1))
        if self.views == 3:
            view2 = pil_loader(os.path.join(self.root, viwe2))
        target = torch.tensor(target).long()

        if self.transform:
            view0 = self.transform(view0)
            view1 = self.transform(view1)
            if self.views == 3:
                view2 = self.transform(view2)

        if self.views == 3:
            return [view0, view1, view2], target
        return [view0, view1], target

    def __len__(self):
        return len(self.targets)


__dataset_dict = {
    'EdgeMnist': EdgeMNISTDataset,
    'FashionMnist': EdgeFMNISTDataset,
    'mvc-10': MultiViewClothingDataset,
    'coil-20': COIL20Dataset,
    'coil-100': COIL100Dataset,
    'office-31': Office31,
    'ff++': FFDataset
}


def get_train_dataset(args, transform):
    data_class = __dataset_dict.get(args.dataset.name, None)
    if data_class is None:
        raise ValueError("Dataset name error.")
    train_set = data_class(root=args.dataset.root, train=True,
                           transform=transform, download=True, views=args.views)
    # Subset
    # reduce_size = len(train_set) // 500
    # indices = random.sample(range(len(train_set)), reduce_size)
    # train_set=Subset(train_set,indices)

    return train_set


def get_val_dataset(args, transform):
    data_class = __dataset_dict.get(args.dataset.name, None)
    if data_class is None:
        raise ValueError("Dataset name error.")
    val_set = data_class(root=args.dataset.root, train=False,
                         transform=transform, views=args.views)

    # reduce_size = len(val_set) // 500
    # indices = random.sample(range(len(val_set)), reduce_size)
    # val_set=Subset(val_set,indices)

    return val_set


if __name__ == '__main__':
    # dataset = MultiViewClothingDataset(train=False, views=3)
    dataset = COIL100Dataset(root='/home/xyzhang/guanzhouke/MyData', train=True, views=4)
    print(dataset[0])
    pass
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=2)
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=3)
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=4)
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=5)
    # generate_mvc_dataset('/mnt/disk3/data/mvc-10', views=6)