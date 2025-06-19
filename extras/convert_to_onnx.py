from clrnet.models.registry import build_net
import torch

def convert_to_onnx(cfg):
    
    pth_path = "models/culane_r18.pth"
    onnx_path = "models/culane_r18_b1.onnx"

    model = build_net(cfg)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(pth_path)
    if 'net' in checkpoint:
        checkpoint = checkpoint['net']

    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 320, 800).cuda()
    torch.onnx.export(
        model.module,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Exported to {onnx_path}")