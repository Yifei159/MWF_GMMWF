import os
import argparse
from config import Config, cfg_to_json
from io_utils import list_train_samples
from gmm1d import train_source_gmm, train_noise_gmm_per_channel

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--root', type=str, default="./multi_yen_2-6")
    p.add_argument('--process_sr', type=int, default=96000)
    p.add_argument('--n_channels', type=int, default=5)
    p.add_argument('--ref_ch', type=int, default=0)
    p.add_argument('--stats_dir', type=str, default='./stats')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        root=args.root,
        process_sr=args.process_sr,
        ref_ch=args.ref_ch,
        n_channels=args.n_channels,
        stats_dir=args.stats_dir,
    )
    print(cfg_to_json(cfg))
    train_dirs = list_train_samples(cfg.root)
    if not train_dirs:
        print('no train samples:', os.path.join(cfg.root, 'train'))
        return
    train_source_gmm(cfg, train_dirs)
    train_noise_gmm_per_channel(cfg, train_dirs)


if __name__ == '__main__':
    main()
