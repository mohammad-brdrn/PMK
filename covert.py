from tool import darknet2pytorch
import torch

# load weights from darknet format
model = darknet2pytorch.Darknet('yolov4-cust.cfg', inference=True)
model.load_weights('yolov4-cust_last.weights')

# save weights to pytorch format
torch.save(model.state_dict(), 'yolov4-pytorch.pth')

