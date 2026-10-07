"""Minimal headless training entry point for Mamba-PFAN experiments."""

import time

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions


def main():
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    print(f"The number of training images = {len(dataset)}", flush=True)

    model = create_model(opt)
    model.setup(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start = time.time()
        epoch_iter = 0
        for data in dataset:
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if epoch_iter % (opt.print_freq * opt.batch_size) == 0:
                losses = model.get_current_losses()
                values = " | ".join(f"{name}: {value:.4f}" for name, value in losses.items())
                print(f"[Epoch {epoch}] {epoch_iter}/{len(dataset)} | {values}", flush=True)

        model.update_learning_rate()
        model.save_networks("latest")
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks(epoch)
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
