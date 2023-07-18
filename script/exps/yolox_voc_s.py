# encoding: utf-8
import os

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 8
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import VOCDetection, TrainTransform
        return VOCDetection(
            # data_dir="/data",
            data_dir="/home/rvl224/文件/wilbur_data",
            image_sets=[('train')],
            img_size=self.input_size,
            rot=self.rotate,
            preproc=TrainTransform(
                max_labels=80,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

        # return MVTECDetection(
        #     data_dir="/home/rvl224/文件/MVTEC",
        #     image_sets=[('train'),('val')],
        #     img_size=self.input_size,
        #     preproc=TrainTransform(
        #         max_labels=120,
        #         flip_prob=self.flip_prob,
        #         hsv_prob=self.hsv_prob),
        #     cache=cache,
        #     cache_type=cache_type,
        # )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import VOCDetection, ValTransform,MVTECDetection
        legacy = kwargs.get("legacy", False)

        return VOCDetection(
            # data_dir="/data",
            data_dir="/home/rvl224/文件/wilbur_data",
            image_sets=[('test')],
            rot=self.rotate,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )