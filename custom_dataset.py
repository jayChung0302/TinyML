import os, sys
import omegaconf
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataset(datacfg):
    image_datasets = {}
    data_transforms = get_transform(datacfg)
    if datacfg.name in datasets.__dict__.keys():
        image_datasets['train'] = getattr(datasets, datacfg.name)(root='./dataset', train=True, download=True, transform=data_transforms['train_transform'])
        image_datasets['val'] = getattr(datasets, datacfg.name)(root='./dataset', train=False, download=True, transform=data_transforms['val_transform'])
    else:
        for x, y in zip(['train', 'val'], data_transforms.keys()):
            image_datasets[x] = datasets.ImageFolder(os.path.join(datacfg.data_dir, x), data_transforms[y])
    return image_datasets

def get_transform(datacfg):
    data_transforms = {}
    for mode in ['train_transform', 'val_transform']:
        config_ls = getattr(datacfg, mode)
        transform_ls = []
        for transform_name in config_ls:
            param = None
            if isinstance(transform_name, omegaconf.listconfig.ListConfig):
                transform, param = transform_name
            else:
                transform = transform_name

            if transform in transforms.__dict__.keys():
                if transform == 'Normalize':
                    mean = datacfg.mean
                    std = datacfg.std
                    transform_ls.append(getattr(transforms, transform)(mean=mean, std=std))
                else:
                    if param is not None:
                        transform_ls.append(getattr(transforms, transform)(param))
                    else:
                        transform_ls.append(getattr(transforms, transform)())
        #TODO: RandAugment config
        # N, M=3, 13
        # train_transform.transforms.insert(0, RandAugment(N, M))
        data_transforms[mode] = transforms.Compose(transform_ls)

    return data_transforms
