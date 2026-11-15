import torch, numpy as np, cv2
from train import PageNet
from fastgen import scene, OH, OW
net = PageNet(); net.load_state_dict(torch.load("/root/net/pagenet.pt", map_location="cpu")); net.eval()
x = torch.zeros(1, 3, OH, OW)
try:
    torch.onnx.export(net, x, "/root/net/pagenet.onnx", input_names=["image"], output_names=["page"], opset_version=11, dynamo=False)
except TypeError:
    torch.onnx.export(net, x, "/root/net/pagenet.onnx", input_names=["image"], output_names=["page"], opset_version=11)
# Same answers in OpenCV DNN (the engine the app uses) as in PyTorch?
cvnet = cv2.dnn.readNetFromONNX("/root/net/pagenet.onnx")
worst = 0
for s in range(5):
    img, mask, _ = scene(555000 + s)
    blob = cv2.dnn.blobFromImage(img, 1.0, (OW, OH), (0, 0, 0), False, False)
    cvnet.setInput(blob); a = cvnet.forward()[0, 0]
    with torch.no_grad(): b = net(torch.from_numpy(img).permute(2, 0, 1)[None].float())[0, 0].numpy()
    worst = max(worst, float(np.abs(a - b).max()))
import os
print("OpenCV vs PyTorch max difference:", worst, " size KB:", os.path.getsize("/root/net/pagenet.onnx") // 1024)
