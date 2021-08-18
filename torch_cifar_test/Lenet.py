import torch


class Lenet(torch.nn.Module):
    def __init__(self, num_cate):
        super(Lenet, self).__init__()
        self.conv_unit = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1,padding=0),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc_unit = torch.nn.Sequential(
            torch.nn.Linear(16*5*5, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, num_cate)
        )
        # test shape
        # x = torch.rand(1, 3, 32, 32)
        # out = self.conv_unit(x)
        # print(out.shape)
        self.cirtizen = torch.nn.CrossEntropyLoss()

    def forward(self, input_):
        x = self.conv_unit(input_)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        logits = self.fc_unit(x)
        # loss = self.cirtizen(logits, y)
        return logits

