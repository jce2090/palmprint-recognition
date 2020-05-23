'''
@Author: your name
@Date: 2020-05-13 12:46:22
@LastEditors: wei
@LastEditTime: 2020-05-18 10:45:36
@Description: file content
'''
import torch
import onnx
import numpy as np
from PIL import Image
from networks import create_model
from utils import get_configuration
import torchvision.transforms as transform
import caffe2.python.onnx.backend as backend


def transform_to_onnx(origin_im_tensor, model):
    """Transform model to onnx
    """
    output_onnx = 'palmprint.onnx'
    x = origin_im_tensor
    print('==> Exporting model to ONNX format at {}'.format(output_onnx))
    input_names = ['input0']
    output_names = ['output0']

    torch_out = torch.onnx._export(model, x, output_onnx, export_params=True,
                                   verbose=False, input_names=input_names, output_names=output_names)

    print('==> Loading and checking exported model from {}'.format(output_onnx))
    onnx_model = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_model)

    return torch_out, onnx_model

def pre_process(cfg):
    """Pre_process image
    """
    src_img = 'data/IITD/flip/ROI/test/001_5Class190.bmp'
    input_size = 150
    mean_vals = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
    std_vals = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)

    imagenet_ids = []

    #################prepare model#######################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints = torch.load('./checkpoints/facenet/checkpoint_0.pth')
    model = create_model(cfg)
    model.load_state_dict(checkpoints['model'])
    model.eval()

    #################prepare image#######################
    img = Image.open(src_img).convert('L')
    img = img.resize((227, 227), resample=Image.BILINEAR)
    img_tensor = transform.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0)

    return model, img_tensor


def test_onnx_model(origin_img_tensor, onnx_model, torch_out):
    """Test onnx model

    Arguments:
        model {str} -- filename of onnx model
    """
    print('==> Loading onnx model into caffe2 backend')
    caff2_backend = backend.prepare(onnx_model)
    B = {'input0': origin_img_tensor.data.numpy()}
    output = caff2_backend.run(B)['output0']

    print('==> compare torch output and caffe2 output')
    np.testing.assert_almost_equal(torch_out.data.numpy(), output, decimal=5)
    print('==> Passed')


if __name__ == "__main__":
    cfg = get_configuration()
    model, img = pre_process(cfg)
    torch_out, onnx_model = transform_to_onnx(img, model)
    test_onnx_model(img, onnx_model, torch_out)