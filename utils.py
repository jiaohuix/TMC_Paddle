import os
import random
import numpy as np
import paddle
import logging
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(loggername,save_path='.'):
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.INFO)
    save_path = save_path

    log_path = os.path.join(save_path,"logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logname = os.path.join(log_path,f'{loggername}.log')
    fh = logging.FileHandler(logname, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def save_model(model,save_dir):
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    paddle.save(model.state_dict(),os.path.join(save_dir, "model.pdparams"))

if __name__ == '__main__':
    path="test_tipc/lite_data/tiny_sample.npy"
    import numpy as np
    data=np.load(path,allow_pickle=True)[0]["data"]
    import matplotlib.pyplot as plt
    # for x in data.values():
    #     print(x.shape)
    #     plt.imshow(x)
    plt.imshow(data[0].reshape(12,20))
    plt.show()
    from reprod_log import ReprodDiffHelper
    helpper=ReprodDiffHelper()
    info1=helpper.load_info("./output_predict_engine.npy")
    print(info1)
    info2=helpper.load_info("./output_inference_engine.npy")
    print(info2)
