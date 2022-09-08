import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.tanh(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = F.tanh(x)
        x = self.dropout3(x)

        x = self.output(x)
        output = F.sigmoid(x)
        return output


class NetPisaRun2(nn.Module):
    def __init__(self, name, input_dim, nlayers, nnodes):
        super(NetPisaRun2, self).__init__()

        assert nlayers == len(nnodes)
        self.nlayers = nlayers
        self.name = name
        inp_shape = input_dim
        self.layers = {}
        for i in range(self.nlayers):
            out_shape = nnodes[i]
            self.layers[f"{name}_layer_{i}"] = nn.Linear(inp_shape, out_shape)
            self.layers[f"{name}_dropout_{i}"] = nn.Dropout(0.2)
            self.layers[f"{name}_bn_{i}"] = nn.BatchNorm1d(out_shape)
            inp_shape = out_shape

        self.layers[f"{name}_output"] = nn.Linear(inp_shape, 1)
        self.layers = nn.ModuleDict(self.layers)

    def forward(self, x):
        for i in range(self.nlayers):
            x = self.layers[f"{self.name}_layer_{i}"](x)
            x = self.layers[f"{self.name}_bn_{i}"](x)
            x = F.tanh(x)
            x = self.layers[f"{self.name}_dropout_{i}"](x)
        x = self.layers[f"{self.name}_output"](x)
        x = F.sigmoid(x)
        return x


class NetPisaRun2Combination(nn.Module):
    def __init__(self, name, nlayers, nnodes, subnetworks, freeze):
        super(NetPisaRun2Combination, self).__init__()
        assert nlayers == len(nnodes)
        self.subnetworks = subnetworks
        self.freeze = freeze
        self.nlayers = nlayers
        self.name = name
        inp_shape = 4
        self.layers = {}
        for i in range(self.nlayers):
            out_shape = nnodes[i]
            self.layers[f"{name}_layer_{i}"] = nn.Linear(inp_shape, out_shape)
            self.layers[f"{name}_dropout_{i}"] = nn.Dropout(0.2)
            self.layers[f"{name}_bn_{i}"] = nn.BatchNorm1d(out_shape)
            inp_shape = out_shape

        self.layers[f"{name}_output"] = nn.Linear(inp_shape, 1)
        self.layers = nn.ModuleDict(self.layers)

    def forward(self, inp_nomass, inp_mass):
        all_inputs = torch.cat((inp_mass, inp_nomass), 1)
        for key in self.subnetworks.keys():
            if key in self.freeze:
                for param in self.subnetworks[key].parameters():
                    param.requires_grad = False
            else:
                for param in self.subnetworks[key].parameters():
                    param.requires_grad = True

        self.output1 = self.subnetworks["sig_vs_ewk"](all_inputs)
        self.output2 = self.subnetworks["sig_vs_dy"](all_inputs)
        self.output3 = self.subnetworks["no_mass"](inp_nomass)
        self.output4 = self.subnetworks["mass"](inp_mass)
        self.combine_input = torch.cat(
            (self.output1, self.output2, self.output3, self.output4), 1
        )
        x = self.combine_input
        for i in range(self.nlayers):
            x = self.layers[f"{self.name}_layer_{i}"](x)
            x = self.layers[f"{self.name}_bn_{i}"](x)
            x = F.tanh(x)
            x = self.layers[f"{self.name}_dropout_{i}"](x)
        x = self.layers[f"{self.name}_output"](x)
        x = F.sigmoid(x)

        return x


class MvaCategorizer(nn.Module):
    def __init__(self, name, input_dim, nlayers, nnodes):
        super(MvaCategorizer, self).__init__()

        assert nlayers == len(nnodes)
        self.nlayers = nlayers
        self.name = name
        inp_shape = input_dim
        self.layers = {}
        for i in range(self.nlayers):
            out_shape = nnodes[i]
            self.layers[f"{name}_layer_{i}"] = nn.Linear(inp_shape, out_shape)
            self.layers[f"{name}_dropout_{i}"] = nn.Dropout(0.2)
            self.layers[f"{name}_bn_{i}"] = nn.BatchNorm1d(out_shape)
            inp_shape = out_shape

        self.layers[f"{name}_output"] = nn.Linear(inp_shape, 1)
        self.layers = nn.ModuleDict(self.layers)

    def forward(self, x):
        for i in range(self.nlayers):
            x = self.layers[f"{self.name}_layer_{i}"](x)
            x = self.layers[f"{self.name}_bn_{i}"](x)
            x = F.tanh(x)
            x = self.layers[f"{self.name}_dropout_{i}"](x)
        x = self.layers[f"{self.name}_output"](x)
        x = F.sigmoid(x)
        return x


class JetPairClassifier(nn.Module):
    def __init__(self, name, input_dim, output_dim, nlayers, nnodes):
        super(JetPairClassifier, self).__init__()

        assert nlayers == len(nnodes)
        self.nlayers = nlayers
        self.name = name
        inp_shape = input_dim
        self.layers = {}
        for i in range(self.nlayers):
            out_shape = nnodes[i]
            self.layers[f"{name}_layer_{i}"] = nn.Linear(inp_shape, out_shape)
            self.layers[f"{name}_dropout_{i}"] = nn.Dropout(0.2)
            self.layers[f"{name}_bn_{i}"] = nn.BatchNorm1d(out_shape)
            inp_shape = out_shape

        self.layers[f"{name}_output"] = nn.Linear(inp_shape, output_dim)
        self.layers = nn.ModuleDict(self.layers)

    def forward(self, x):
        for i in range(self.nlayers):
            x = self.layers[f"{self.name}_layer_{i}"](x)
            x = self.layers[f"{self.name}_bn_{i}"](x)
            x = F.tanh(x)
            x = self.layers[f"{self.name}_dropout_{i}"](x)
        x = self.layers[f"{self.name}_output"](x)
        x = F.sigmoid(x)
        return x
