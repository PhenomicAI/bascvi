import argparse
from .embed import PaiEmbeddings
from .utils.option_choices import tissue_organ_option_choices


def app():
    parser = argparse.ArgumentParser("pai")
    sub_parser = parser.add_subparsers(dest="command")

    embed = sub_parser.add_parser("embed")
    embed.add_argument("--tmp-dir", required=True, type=str, help="the root output directory where the downloaded zip files (zips/) and unzipped directories (results/) will be output")
    embed.add_argument("--h5ad-path", required=True, type=str, help="the path to the single cell RNA .h5ad file intended to be uploaded and embeded")
    embed.add_argument("--tissue-organ", required=True, type=str, choices=tissue_organ_option_choices, help="specifies the tissue/organ associated wrt. the single cells")

    args = parser.parse_args()

    if args.command == "embed":
        pai_embeddings = PaiEmbeddings(args.tmp_dir)
        pai_embeddings.inference(args.h5ad_path, args.tissue_organ)


if __name__ == "__main__":
    app()
