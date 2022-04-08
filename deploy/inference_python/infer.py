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
import os
from paddle import inference
import numpy as np
from reprod_log import ReprodLogger

def softmax(logits):
    t = np.exp(logits)
    a = np.exp(logits) / np.sum(t, axis=1)
    return a

def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="MULTI-VIEW CLASSIFICATION", add_help=add_help)

    parser.add_argument(
        '--data-path',
        type=str,
        default="datasets/tiny_sample.npy",
        help='Data path of handwritten_6views.')

    parser.add_argument(
        "--model-dir", default=None, help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")

    args = parser.parse_args()
    return args


class InferenceEngine(object):
    """InferenceEngine

    Inference engina class which contains preprocess, run, postprocess

    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.

        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))


    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor

        initialize the inference engine

        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_tensors = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_tensors = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return predictor, config, input_tensors, output_tensors

    def preprocess(self, args):
        """preprocess

        Preprocess to the input.

        Args:
            data: data.

        Returns: Input data after preprocess.
        """
        infer_data = np.load(args.data_path, allow_pickle=True)[0]
        data, target = infer_data["data"], infer_data["target"]
        data=np.concatenate(list(data.values()),axis=-1)
        data = data.reshape([args.batch_size, -1])
        return data

    def postprocess(self, output):
        """postprocess

        Postprocess to the inference engine output.

        Args:
            output: Inference engine output.

        Returns: Output data after argmax.
        """
        evidence_a= output
        probs=softmax(evidence_a)[0]
        print(f"scores:{probs}")
        label_id = evidence_a.argmax()
        prob = probs[label_id]
        return label_id, prob, probs

    def run(self, data):
        """run

        Inference process using inference engine.

        Args:
            x: Input data after preprocess.

        Returns: Inference engine output
        """
        self.input_tensors[0].copy_from_cpu(data)
        self.predictor.run()
        output = self.output_tensors[6].copy_to_cpu() # evidence_a
        return output




def infer_main(args):
    """infer_main

    Main inference function.

    Args:
        args: Parameters generated using argparser.

    Returns:
        label_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="classification",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # dataset preprocess
    data = inference_engine.preprocess(args)
    data = data.reshape([args.batch_size,-1])
    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(data)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    label_id, prob, probs = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print(f"label_id: {label_id}, prob: {prob}")
    return label_id, prob, probs


if __name__ == "__main__":
    args = get_args()
    label_id, prob,probs = infer_main(args)

    reprod_logger = ReprodLogger()
    reprod_logger.add("label_id", np.array([label_id]))
    reprod_logger.add("prob", np.array([prob]))
    reprod_logger.add("probs", np.array([probs]))
    reprod_logger.save("output_inference_engine.npy")
