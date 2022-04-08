import os
import time
import argparse
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
from model import TMC
from data import Multi_view_data
from utils import AverageMeter,set_seed,save_model,get_logger

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-path',
        type=str,
        default="datasets/handwritten_6views",
        help='Data path of handwritten_6views.')

    parser.add_argument(
        '--classes',
        type=int,
        default=10,
        help='Num classes of handwritten_6views.')

    parser.add_argument(
        '--eval',
        action='store_true',
        help="Evaluation on devset.")

    parser.add_argument(
        '--batch-size',
        type=int,
        default=200,
        metavar='N',
        help='Input batch size for training [default: 100].')

    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        metavar='N',
        help='Number of epochs to train [default: 500].')

    parser.add_argument(
        '--model-path',
        type=str,
        default="",
        help='Pretrained ckpt for evaluadtion.')

    parser.add_argument(
        "--max-steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        '--lambda-epochs',
        type=int,
        default=50,
        metavar='N',
        help='Gradually increase the value of lambda from 0 to 1.')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.003,
        metavar='LR',
        help='learning rate')

    parser.add_argument(
        "--output-dir",
        default="output",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Log every X updates steps.")

    parser.add_argument(
        "--save-steps",
        type=int,
        default=400,
        help="Save checkpoint every X updates steps.", )

    parser.add_argument(
        "--seed",
        default=1024,
        type=int,
        help="Random seed for initialization.")

    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu"],
        help="The device to select to train the model, is must be cpu/gpu.", )

    args = parser.parse_args()
    args.dims = [[240], [76], [216], [47], [64], [6]]
    args.views = len(args.dims)
    return args

@paddle.no_grad()
def evaluate(model, epoch, data_loader,logger):
    model.eval()

    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data_num += target.shape[0]
        with paddle.no_grad():
            data=paddle.concat(list(data.values()),axis=1)
            target = paddle.cast(target, dtype='int64')
            evidence, evidence_a, alpha, alpha_a=model(data)
            loss=model.criterion(alpha, alpha_a,target, epoch)
            predicted = paddle.argmax(evidence_a, 1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())

    loss=loss_meter.avg
    acc=correct_num/data_num
    logger.info("eval loss: %f, acc: %s, " % (loss, acc))

    model.train()
    return loss,acc


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    logger=get_logger(loggername="TMC",save_path=args.output_dir)
    # datasets
    train_set=Multi_view_data(args.data_path, train=True)
    test_set=Multi_view_data(args.data_path, train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    N_mini_batches = len(train_loader)
    logger.info('The number of training images = %d' % N_mini_batches)

    # TMC model
    model = TMC(classes=args.classes, views=args.views, classifier_dims=args.dims, lambda_epochs=args.lambda_epochs)

    # Optimizer
    optimizer = optim.Adam(learning_rate=args.lr,parameters=model.parameters(), weight_decay=1e-5)

    # Train loop
    max_acc = 0.0
    global_step = 0
    tic_train = time.time()
    num_training_steps = (args.max_steps if args.max_steps > 0 else  (len(train_loader) * args.epochs))
    loss_meter = AverageMeter()
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):

            global_step += 1
            data=paddle.concat(data.values(),axis=1)
            target = paddle.cast(target,dtype='int64')
            # refresh the optimizer
            optimizer.clear_grad()
            # evidences, evidence_a, loss = model(data, target, epoch)
            # evidences, evidence_a, loss = model(data, target, epoch)
            evidence, evidence_a, alpha, alpha_a=model(data)
            loss=model.criterion(alpha, alpha_a,target, epoch)
            # compute gradients and take step
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            if (global_step % args.logging_steps == 0 or global_step == num_training_steps):
                logger.info(
                    "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                    % (
                        global_step,
                        num_training_steps,
                        epoch,
                        batch_idx,
                        paddle.distributed.get_rank(),
                        loss,
                        optimizer.get_lr(),
                        args.logging_steps / (time.time() - tic_train),))
                tic_train = time.time()

            if global_step % args.save_steps == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                # evaluate
                loss,acc=evaluate(model,epoch,data_loader=test_loader,logger=logger)
                logger.info("eval done total : %s s" % (time.time() - tic_eval))
                logger.info("=" * 100)
                # save model
                if paddle.distributed.get_rank() == 0:
                    output_dir=os.path.join(args.output_dir,"step%d"%(global_step))
                    save_model(model,output_dir)
                    # save best
                    if acc > max_acc:
                        max_acc = acc
                        logger.info(f"max_acc:{acc}")
                        best_dir = os.path.join(args.output_dir,"model_best")
                        save_model(model,best_dir)

            if global_step >= num_training_steps:
                return

def print_info_arguments(args):
    """logger.info arguments"""
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")

if __name__ == "__main__":
    args = parse_args()
    print_info_arguments(args)
    n_gpu = len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
    if not args.eval:
        if args.device in "gpu" and n_gpu > 1:
            paddle.distributed.spawn(do_train, args=(args,), nprocs=n_gpu)
        else:
            do_train(args)
    else:
        logger=get_logger(loggername="TMC",save_path=args.output_dir)
        test_set = Multi_view_data(args.data_path, train=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        model = TMC(classes=args.classes, views=args.views, classifier_dims=args.dims, lambda_epochs=args.lambda_epochs)
        if args.model_path:
            state=paddle.load(os.path.join(args.model_path,"model.pdparams"))
            model.set_dict(state)
            logger.info(f"Loaded pretrained ckpt from {args.model_path}.")
        evaluate(model, args.epochs, test_loader,logger)
