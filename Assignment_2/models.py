import torch


class GRUFusion(torch.nn.Module):

    def __init__(self, time_var_input_dim, time_var_hidden_dim, time_inv_input_dim, time_inv_hidden_dim, output_dim=2):
        super().__init__()
        self.time_var_input_dim = time_var_input_dim
        self.time_var_hidden_dim = time_var_hidden_dim
        self.time_inv_input_dim = time_inv_input_dim
        self.time_inv_hidden_dim = time_inv_hidden_dim
        self.output_dim = output_dim

        self.GRU = torch.nn.GRU(input_size=self.time_var_input_dim, hidden_size=self.time_var_hidden_dim,
                                batch_first=True)
        self.DNN_1 = torch.nn.Sequential(
            torch.nn.Linear(self.time_inv_input_dim, self.time_inv_hidden_dim),
            torch.nn.LeakyReLU()
        )

        self.DNN_2 = torch.nn.Sequential(
            torch.nn.Linear(self.time_var_hidden_dim + self.time_inv_hidden_dim, 20),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, self.output_dim)
        )

    def forward(self, time_inv, time_var):
        time_inv_hidden = self.DNN_1(time_inv)
        _, time_var_hidden = self.GRU(time_var)
        hidden = torch.cat((time_inv_hidden, time_var_hidden.squeeze(dim=0)), dim=1)

        output = self.DNN_2(hidden)
        return output
