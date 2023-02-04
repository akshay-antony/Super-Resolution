import torch
import torch_tensorrt
from torch_model import Network


def main():
    model = Network().eval().cuda()
    inputs = [torch_tensorrt.Input((1, 1, 16, 1800), type=torch.float)]
    trt_model = torch_tensorrt.compile(model,
                                       inputs=inputs,
                                       enabled_precisions={torch.float})
    torch.save(trt_model, "trt_fp32.ts")

if __name__ == "__main__":
    main()