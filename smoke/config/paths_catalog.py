import os


class DatasetCatalog():
    DATA_DIR = "datasets"
    DATASETS = {
        "kitti_train": {
            "root": "kitti/training/",
        },
        "kitti_test": {
            "root": "kitti/testing/",
        },
        "nusc_train_mini": {
            "root": "nuscenes/",
            "extra": {"json_file": "smoke_convert/train_mini.json"}
        },
        "nusc_train_half": {
            "root": "nuscenes/",
            "extra": {"json_file": "smoke_convert/train_half.json"}
        },
        "nusc_train_full": {
            "root": "nuscenes/",
            "extra": {"json_file": "smoke_convert/train_full.json"}
        },
        "nusc_val_mini": {
            "root": "nuscenes/",
            "extra": {"json_file": "smoke_convert/val_mini.json"}
        },
        "nusc_val_full": {
            "root": "nuscenes/",
            "extra": {"json_file": "smoke_convert/val_full.json"}
        }
    }

    @staticmethod
    def get(name):
        if name not in DatasetCatalog.DATASETS:
            raise RuntimeError("Dataset not available: {}".format(name))

        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["root"]),
        )

        if "extra" in attrs:
            args.update(attrs["extra"])

        if "kitti" in name:
            factory="KITTIDataset"
        elif "nusc" in name:
            factory="NuScenesDataset"
        else:
            raise RuntimeError("Dataset not implemented: {}".format(name))

        return dict(
            factory=factory,
            args=args,
        )


class ModelCatalog():
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        url = ModelCatalog.IMAGENET_MODELS[name]
        return url
