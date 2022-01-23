import argparse
import torch
from pipeline import optim_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-s', '--start', type=int, default=0)
    args = parser.parse_args()
    optim_pipeline(args.cuda, args.start)


if __name__=="__main__":
    main()