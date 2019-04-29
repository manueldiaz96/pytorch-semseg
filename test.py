import os
import torch
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


def test(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img = cv2.imread(args.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data_loader = get_loader(args.dataset)
    loader = data_loader(root=None, is_transform=True, img_norm=args.img_norm, test_mode=True)
    n_classes = loader.n_classes

    resized_img = cv2.resize(img, (loader.img_size[1], loader.img_size[0]), interpolation=cv2.INTER_CUBIC)

    orig_size = img.shape[:-1]
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        # uint8 with RGB mode, resize width and height which are odd numbers
        img = cv2.resize(img, (orig_size[1] // 2 * 2 + 1, orig_size[0] // 2 * 2 + 1))
    else:
        img = cv2.resize(img, (loader.img_size[1], loader.img_size[0]))

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path, map_location='cpu')["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    outputs = model(images)

    if args.dcrf:
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        unary = np.ascontiguousarray(unary)

        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = args.out_path[:-4] + "_drf.png"
        cv2.imsave(dcrf_path, decoded_crf)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        pred = pred.astype(np.float32)
        # float32 with F mode, resize back to orig_size
        pred = cv2.imresize(pred, orig_size, "nearest", mode="F")

    decoded = loader.decode_segmap(pred)
    print("Classes found: ", np.unique(pred))
    #cv2.imwrite(args.out_path, decoded)
    print("Segmentation Mask Saved at: {}".format(args.out_path))
    plt.imshow(decoded)
    return (decoded*255).astype(int)


#if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Params")
parser.add_argument(
    "--model_path",
    nargs="?",
    type=str,
    default="segnet_cityscapes_best_model.pkl",
    help="Path to the saved model",
)
parser.add_argument(
    "--dataset",
    nargs="?",
    type=str,
    default="cityscapes",
    help="Dataset to use ['pascal, camvid, ade20k etc']",
)

parser.add_argument(
    "--img_norm",
    dest="img_norm",
    action="store_true",
    help="Enable input image scales normalization [0, 1] \
                          | True by default",
)
parser.add_argument(
    "--no-img_norm",
    dest="img_norm",
    action="store_false",
    help="Disable input image scales normalization [0, 1] |\
                          True by default",
)
parser.set_defaults(img_norm=True)

parser.add_argument(
    "--dcrf",
    dest="dcrf",
    action="store_true",
    help="Enable DenseCRF based post-processing | \
                          False by default",
)
parser.add_argument(
    "--no-dcrf",
    dest="dcrf",
    action="store_false",
    help="Disable DenseCRF based post-processing | \
                          False by default",
)
parser.set_defaults(dcrf=False)

parser.add_argument(
    "--img_path", nargs="?", type=str, default='results/munich_000009_000019_leftImg8bit.png', help="Path of the input image"
)
parser.add_argument(
    "--out_path", nargs="?", type=str, default='results/148k-it-lille-v2.png', help="Path of the output segmap"
)
args = parser.parse_args()
imger = test(args)

