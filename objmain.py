import argparse
import torch
from pipeline import optim_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-o', '--object', type=str, default='mug')
    parser.add_argument('-e', '--end', type=int, default=100)
    args = parser.parse_args()
    optim_pipeline(args.cuda, args.start, args.object, args.end)


if __name__=="__main__":
    main()