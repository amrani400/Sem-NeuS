import torch
import torch.nn as nn

class SemanticNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_hidden,
                 n_layers,
                 d_out):
        super().__init__()
        
        # Takes geometric features (d_feature) -> Hidden -> Output (d_out)
        dims = [d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        
        self.num_layers = len(dims)
        
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
            
        self.relu = nn.ReLU()

    def forward(self, feature_vectors):
        x = feature_vectors
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        return x # Output predicted embedding