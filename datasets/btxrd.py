import os
import json
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets

@DATASET_REGISTRY.register()
class BTXRD(DatasetBase):

    dataset_dir = "btxrd"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # Classnames mapping
        self.dataset_classnames = [
            'giant cell tumor',
            'multiple osteochondromas',
            'osteochondroma',
            'osteofibroma',
            'osteosarcoma',
            'other bt',
            'other mt',
            'simple bone cyst',
            'synovial osteochondroma'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.dataset_classnames)}

        # [1] Đọc toàn bộ dữ liệu từ các thư mục con train, val, test
        train = self.read_data("train")
        val = self.read_data("val")
        test = self.read_data("test")

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            # Tên file cache để lưu lại tập few-shot đã sample (giúp tái lập kết quả)
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                # [2] Lấy mẫu Few-Shot:
                # Gọi hàm generate_fewshot_dataset từ base_dataset.py (Dassl)
                # Hàm này nhóm data theo label, sau đó dùng random.sample() 
                # chọn đúng `num_shots` (ví dụ 16) ảnh cho TỪNG class một.
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        ori_train = train
        ori_val = val
        ori_test = test
        # [3] Lọc class (Subsample Classes):
        # Cắt lấy tập "base", "new", hoặc "all" classes cho train/val/test theo config.
        # Ví dụ: nếu base, nó chỉ lấy 5 class đầu tiên để đưa vào tập train.
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        
        # [4] Chuẩn bị tập ID và OOD cho lúc test:
        # Tập ID (In-Distribution): Lấy toàn bộ ảnh thuộc base classes từ tập test gốc.
        _, _, id = OxfordPets.subsample_classes(ori_train, ori_val, ori_test, subsample='base')
        # Tập OOD (Out-Of-Distribution): Lấy toàn bộ ảnh thuộc new classes (chưa từng thấy lúc train) từ tập test gốc.
        _, _, ood = OxfordPets.subsample_classes(ori_train, ori_val, ori_test, subsample='new')
        self.id = id
        self.ood = ood

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split):
        split_dir = os.path.join(self.dataset_dir, split)
        images_dir = os.path.join(split_dir, "images")
        annotations_dir = os.path.join(split_dir, "annotations")
        
        imnames = listdir_nohidden(images_dir)
        items = []

        for imname in sorted(imnames):
            impath = os.path.join(images_dir, imname)
            base = os.path.splitext(imname)[0]
            json_path = os.path.join(annotations_dir, base + ".json")
            
            classname = "unknown"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    shapes = data.get("shapes", [])
                    if len(shapes) > 0:
                        classname = shapes[0]["label"]
            
            label = self.class_to_idx.get(classname, -1)
            if label == -1:
                print(f"Warning: Classname {classname} not recognized in {imname}")
                
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items
