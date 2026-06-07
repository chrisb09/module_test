import torch
import torch.nn as nn

class MultiInputModel(nn.Module):
    def __init__(self):
        super(MultiInputModel, self).__init__()

    def forward(self, input1, input2):
        # input1 and input2 are expected to be [1, 9]
        # Concatenate them and return a simple sum or similar
        combined = torch.cat((input1, input2), dim=1) # [1, 18]
        # Just return the first element for simplicity, matching "perfect" model logic mostly
        return combined[:, 0:1]

if __name__ == "__main__":
    model = MultiInputModel()
    model.eval()
    
    # Create example inputs for tracing (though we can just script it)
    example1 = torch.randn(1, 9)
    example2 = torch.randn(1, 9)
    
    # Use scripting to handle multiple arguments easily
    scripted_model = torch.jit.script(model)
    
    scripted_model.save("multi_input_model.pt")
    print("Multi-input model saved to multi_input_model.pt")
