import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from nets.yolo import YoloBody


from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use')
    parser.add_argument('--input-dir', type=str, default=r'images',
                        help='Input directory with images')
    # parser.add_argument(
    #     '--image-path',
    #     type=str,
    #     default='./examples/both.png',
    #     help='Input image path')
    parser.add_argument('--aug-smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen-smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise', 'kpcacam'
                        ],
                        help='CAM method')

    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory to save the images')
    args = parser.parse_args()
    
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM
    }

    input_shape = [640, 640]
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 8
    phi = 'l'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YoloBody(anchors_mask, num_classes, phi, False).to(device)
    # model.load_state_dict(torch.load("E:/wt/yolo_decouple_P34/yolo_decouple_P34/logs/ep110-loss0.311-val_loss0.367.pth", map_location=device), strict=False)
    model = model.eval()

    # 图像路径
    # image_path = r'D:\Deep_Learning_folds\Datasets\State-Air\VOC2007\JPEGImages\1697446959026_h_30.50m_roll_-1.70_pitch_-1.00_yaw_-2.30.jpg'
    image_path = 'E:/wt/yolo_decouple_P34/yolo_decouple_P34/VOCdevkit/VOC2007/JPEGImages/FLIR_08907.jpg'
    caption = 'AAAA'


    state_dict = torch.load(r'E:\wt\yolo_decouple_P34\yolo_decouple_P34\logs\ep110-loss0.311-val_loss0.367.pth')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    else:
        state_dict = state_dict
    model.load_state_dict(state_dict)
    print("state_dict"+ str(state_dict))

    # [out0, out1, out2], object_feature, noise_feature, logits_per_image = model(image, caption)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    target_layers = [model.layer4]

    image_pathes = [os.path.join(args.input_dir, img) for img in os.listdir(args.input_dir)]
    for image_path in image_pathes:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f'Processing {base_name}')
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).to(args.device)

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [ClassifierOutputTarget(281)]
        # targets = [ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                        target_layers=target_layers) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # gb_model = GuidedBackpropReLUModel(model=model, device=args.device)
        # gb = gb_model(input_tensor, target_category=None)

        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)

        os.makedirs(args.output_dir, exist_ok=True)

        cam_output_path = os.path.join(args.output_dir, f'{base_name}_cam.jpg')
        # gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
        # cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

        cv2.imwrite(cam_output_path, cam_image)
        # cv2.imwrite(gb_output_path, gb)
        # cv2.imwrite(cam_gb_output_path, cam_gb)
