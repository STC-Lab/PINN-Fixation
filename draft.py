import torch

model = torch.load('./model/model_parameter.pkl')
print(model.state_dict())
