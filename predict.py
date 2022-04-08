import sys
import os
import paddle
import paddle.nn.functional as F
import numpy as np
import argparse
from reprod_log import ReprodLogger
from paddle.io import DataLoader
from model import TMC
from data import Multi_view_data

# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(__dir__, '/../')))
# parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# sys.path.append(parentddir)
# sys.path.append('../')

def get_args(add_help=True):
    parser = argparse.ArgumentParser(
        description='MULTI-VIEW CLASSIFICATION', add_help=add_help)

    parser.add_argument(
        '--data-path',
        type=str,
        default="datasets/handwritten_6views",
        help='Data path of handwritten_6views.')

    parser.add_argument(
        "--model-path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["gpu", "cpu"],
        help="device", )

    parser.add_argument(
        '--classes',
        type=int,
        default=10,
        help='Num classes of handwritten_6views.')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        metavar='N',
        help='Input batch size for training [default: 100].')

    args = parser.parse_args()
    args.dims = [[240], [76], [216], [47], [64], [6]]
    args.views = len(args.dims)

    return args


@paddle.no_grad()
def main(args):
    paddle.set_device(args.device)
    # define model
    model = TMC(classes=args.classes, views=args.views, classifier_dims=args.dims)
    if args.model_path:
        state_path = os.path.join(args.model_path, "model.pdparams")
        state = paddle.load(state_path)
        model.set_dict(state)
        print(f"Load from ckpt {args.model_path} success.")
    model.eval()

    # dataset
    test_set = Multi_view_data(args.data_path, train=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    correct_num, data_num = 0, 0
    result=[]
    for batch_idx, (data, target) in enumerate(test_loader):
        data_num += target.shape[0]
        with paddle.no_grad():
            data = paddle.concat(list(data.values()), axis=-1)
            target = paddle.cast(target, dtype='int64')
            evidence, evidence_a, alpha, alpha_a = model(data)
            predicted_id = paddle.argmax(evidence_a, 1)
            correct_num += (predicted_id == target).sum().item()
            result.append(evidence_a)
            # break

    acc = correct_num / data_num
    print("test acc: %s, " % ( acc))
    evidence_a=paddle.concat(result,axis=0)
    predicted_id = paddle.argmax(evidence_a, 1).numpy()
    probs = F.softmax(evidence_a, axis=-1).numpy()[0]
    print(f"scores:{probs}")
    prob = probs[predicted_id]
    print(f"predicted_id: {predicted_id[0]}, prob: {prob[0]}")
    return predicted_id[0], prob[0],probs


if __name__ == "__main__":
    args = get_args()
    label_id, prob,probs = main(args)
    reprod_logger = ReprodLogger()
    reprod_logger.add("label_id", np.array([label_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.add("probs", np.array([probs]))
    reprod_logger.save("output_predict_engine.npy")
