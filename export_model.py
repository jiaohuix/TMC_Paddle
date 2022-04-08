# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import paddle
import argparse
from model import TMC

# __dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

def get_args(add_help=True):
    """get_args

    Parse all args using argparse lib

    Args:
        add_help: Whether to add -h option on args

    Returns:
        An object which contains many parameters used for inference.
    """
    parser = argparse.ArgumentParser(
        description='MULTI-VIEW CLASSIFICATION', add_help=add_help)

    parser.add_argument(
        "--model-path",
        default=None,
        type=str,
        required=True,
        help="Path of the trained model to be exported.", )

    parser.add_argument(
        '--save-inference-dir',
        default='./infer_output',
        help='path where to save')

    parser.add_argument(
        '--classes',
        type=int,
        default=10,
        help='Num classes of handwritten_6views.')


    args = parser.parse_args()
    args.dims = [[240], [76], [216], [47], [64], [6]]
    args.views = len(args.dims)

    return args


def export(args):
    # build model
    model = TMC(classes=args.classes, views=args.views, classifier_dims=args.dims)
    if args.model_path:
        state_path = os.path.join(args.model_path, "model.pdparams")
        print(state_path)
        state = paddle.load(state_path)
        model.set_dict(state)
    model.eval()

    # decorate model with jit.save
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="float32"),  # [bsz,total_dims] total_dims=sum([240, 76, 216, 47, 64, 6])
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(
        f"inference model  have been saved into {args.save_inference_dir}"
    )


if __name__ == "__main__":
    args = get_args()
    export(args)
