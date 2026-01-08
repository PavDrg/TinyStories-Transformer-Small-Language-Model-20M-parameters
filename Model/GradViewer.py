import torch
from torch.utils import tensorboard


class GradViewer():
    def __init__(self,model:torch.nn.Module,table_name = None):
        self.model = model
        
        self.step = 0
        self.summary_writer = tensorboard.SummaryWriter()
        if table_name == None:
            self.table_name = f"{type(model)} gradient"
        else:
            self.table_name = table_name

    def view_grad(self):
        grad_norms = []
        grad_norm=0
        for name,parametr in self.model.named_parameters():
            if parametr.grad is not None:
                grad_norms.append(parametr.grad.norm(2).to("cpu").detach().item())
            else:
                grad_norms.append(0.0)
            
        
        parametr_norm = sum(grad_norms) / len(grad_norms)
        self.summary_writer.add_scalar(self.table_name,parametr_norm,self.step)
        self.step +=1
        
