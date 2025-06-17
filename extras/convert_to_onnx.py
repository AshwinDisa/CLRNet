from clrnet.models.registry import build_net
import torch

def convert_to_onnx(cfg):
    
    pth_path = "models/culane_r18.pth"
    onnx_path = "models/culane_r18.onnx"

    model = build_net(cfg)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(pth_path)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']

    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    dummy_input = torch.randn(24, 3, 320, 800).cuda()   # input tensor shape, 24 is batch size
    torch.onnx.export(
        model.module,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Exported to {onnx_path}")