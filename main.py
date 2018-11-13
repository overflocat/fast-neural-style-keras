from argparse import ArgumentParser
import json
from helper_functions import train, temp_view, predict


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('-c', type=str, dest='config_path', help='config path',
                        metavar='CONFIG_PATH', required=True)
    parser.add_argument('-m', type=str, dest='mode', help='train, predict or temp_view',
                        metavar='MODE', required=True)
    parser.add_argument('-i', type=str, dest='image_path', help='image for transformation or viewing',
                        metavar='IMAGE_PATH')
    parser.add_argument('-o', type=str, dest='image_output_path', help='image output path',
                        metavar='IMAGE_OUTPUT_PATH')
    parser.add_argument('--iters', type=int, dest='iters', help='iter times, only for temp_view mode',
                        metavar='ITER_TIMES', default=500)

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    with open(args.config_path) as f_config:
        options = json.load(f_config)

    if args.mode == 'train':
        train(options)
    elif args.mode == 'predict':
        predict(options, args.image_path, args.image_output_path)
    elif args.mode == 'temp_view':
        temp_view(options, args.image_path, args.image_output_path, args.iters)

