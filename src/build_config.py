import os
import yaml
from argparse import ArgumentParser

def list_type(arg) -> list:
    return [val for val in arg.split(",")]

def list_int_type(arg) -> list:
    return [int(val) for val in arg.split(",")]

if __name__ == "__main__":

    ds_cfg_path = "configs/datasphere.yaml"
    train_cfg_path = "configs/train_config.yaml"
    with open(ds_cfg_path, "r") as ds_f:
        ds_config = yaml.load(
            ds_f, 
            Loader=yaml.SafeLoader
        )
   

    tr_config = {}
    parser = ArgumentParser()
    parser.add_argument(
        "--gpu-type", 
        type=str, 
        default="g1.1",
        help="GPU setup for traning"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization"
    )
    parser.add_argument(
        "--epochs-per-log",
        type=int,
        default=10,
        help="Number of epochs per log"
    )
    parser.add_argument(
        "--WH",
        help="expected width and height for imgs",
        type=list_int_type,
        default=[128, 128]
    )
    parser.add_argument(
        "--points-n",
        type=int,
        default=10
    )
    parser.add_argument(
        "--log-features-list",
        type=list_type,
        default=[
            "asal_map", 
            "g_splats", 
            "audo_features", 
            "visual_features",
            "fused_features"
        ],
        help="List of features to be logged"
    )
    parser.add_argument(
        "--gs-terms-list",
        type=list_type,
        default=[
            "rigid_loss",
            "rot_loss",
            "iso_loss"        
        ],
        help="""
            List of terms that was used to
            fuse GS from original imeplentation
            and those obtained through audo-visual flow    
        """
    )
    args = parser.parse_args()
    tr_config.update({
        "epochs": args.epochs,
        "WH": args.WH,
        "learning_rate": args.learning_rate,
        "epochs_per_log": args.epochs_per_log,
        "points_n": args.points_n,
        "log_features_list": args.log_features_list,
        "gs_terms_list": args.gs_terms_list,
    })
    
    with open(ds_cfg_path, "w") as ds_f:
        yaml.dump(
            ds_config,
            ds_f,
            Dumper=yaml.SafeDumper
        )
    with open(train_cfg_path, "w") as tr_f:
        yaml.dump(
            tr_config,
            tr_f,
            Dumper=yaml.SafeDumper
        )

