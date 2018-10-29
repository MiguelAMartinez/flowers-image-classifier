import argparse
from utils_ic import read_jason, predict
from model_ic import load_model

parser = argparse.ArgumentParser(description="Predict image with classifier model")
parser.add_argument("img_path", help="set path to image")
parser.add_argument("checkpoint", help="set path to checkpoint")
parser.add_argument("--top_k", type=int, default=1, help="Select top K predictions")
parser.add_argument("--category_names", help="choose category names")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")

args = parser.parse_args()

if args.category_names:
    cat_to_name = read_jason(args.category_names)
else: 
    cat_to_name = None

model_cp = load_model(args.checkpoint)

predict(image_path=args.img_path, model=model_cp, topk=args.top_k, device=args.gpu, cat_to_name=cat_to_name)