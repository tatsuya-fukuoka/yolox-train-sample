import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        self.data_dir = "datasets/person-detect-1"
        self.train_ann = "train_annotations.coco.json"
        self.val_ann = "valid_annotations.coco.json"

        self.num_classes = 1      #今回はpersonのみ検出のため1
        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 1

        self.save_history_ckpt = False # last_ckpt.pthとbest_ckpt.pthのみ出力