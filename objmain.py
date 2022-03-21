import argparse
import torch
from pipeline import optim_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-s', '--start', type=int, default=0)
    parser.add_argument('-o', '--object', type=str, default='bowl')
    parser.add_argument('-e', '--end', type=int, default=100)
    parser.add_argument('-i', '--index', type=int, default=0)
    args = parser.parse_args()
    if args.object == "mug_special":
        from pipeline import optim_pipeline_mug
        optim_pipeline_mug(args.cuda, args.start, "mug_tmp", args.end)
    elif args.object == "bottle_retrain":
        from pipeline import bottle_retrain_pipeline
        bottle_retrain_pipeline(args.cuda, args.start, "bottle", args.end)
    elif args.object == "bowl":
        from pipeline import bowl_tmp
        bowl_list = [
            [
                "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N01/S55/s1/T1/",
                "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/001.obj",
                True
            ],
            [
                "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N05/S55/s3/T1/",
                "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/005.obj",
                True
            ],
            [
                "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N07/S55/s4/T1/",
                "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/007.obj",
                False,
            ],
            [
                "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N14/S64/s2/T1/",
                "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/014.obj",
                True
            ],
            [
                "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N26/S79/s5/T1/",
                "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/026.obj",
                True
            ]
        ]
        t = bowl_list[args.start]
        bowl_tmp(args.cuda, t[0], t[1], t[2], str(args.start)+ 'bowl_check/', 1, args.index)
    else:
        optim_pipeline(args.cuda, args.start, args.object, args.end)

if __name__=="__main__":
    main()