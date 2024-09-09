from argparse import ArgumentParser
import json
import os
from pathlib import Path

from scripts.run_train import train
from scripts.run_kni_scoring import run_kni_on_folder


if __name__ == '__main__':

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--config_root_dir",
        type=str,
    )
    args = parser.parse_args()

    files = os.listdir(args.config_root_dir)
    files = [f for f in files if '.json' in f]

    run_root_dir = os.path.join("exp_logs", os.path.normpath(args.config_root_dir).split(os.sep)[-1])

    print("Run root folder:", run_root_dir)
    print("Config files being run:")

    for i,f in enumerate(files):
        print(f"\t - {i} {f}")

    print("Begin training...")

    # print("skipping: ", files)

    for i,f in enumerate(files):
        print(f"____________{i}/{len(files)}___{f}____________")
        with open(os.path.join(args.config_root_dir, f)) as json_file:
            cfg = json.load(json_file)

        run_dir = os.path.join(run_root_dir, Path(f).stem)

        cfg["datamodule"]["dataloader_args"]["num_workers"] = 4

        cfg["pl_trainer"]["default_root_dir"] = run_dir
        cfg["datamodule"]["root_dir"] = run_dir

        train(cfg)

    # run kni
    run_kni_on_folder(run_root_dir)