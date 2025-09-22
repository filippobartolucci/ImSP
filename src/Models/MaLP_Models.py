import torch.nn as nn
        
class LocalizationModule(nn.Module):
    def __init__(self, num_layers=10, num_features=64, out_num=1):
        super(LocalizationModule, self).__init__()
        
        layers_0 = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        
        layers_1=[]
        layers_2=[]
        layers_3=[]
        layers_4=[]
        for i in range(4):
            layers_1.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                      nn.ReLU(inplace=True)))
        
        for i in range(3):
            layers_2.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        for i in range(4):
            layers_3.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
            
        self.layers_0 = nn.Sequential(*layers_0)
        self.layers_1 = nn.Sequential(*layers_1)
        self.layers_2 = nn.Sequential(*layers_2)
        self.layers_3 = nn.Sequential(*layers_3)
        
        self.layers_5=nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True))
        
        
        self.layers_6=nn.Sequential(nn.Conv2d(num_features, 3, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(3),
                                        nn.ReLU(inplace=True))
        
    
    def forward(self, inputs):
        output = self.layers_0(inputs)
        output = self.layers_1(output)
        output_1 = self.layers_2(output)
        output_2 = self.layers_3(output)
        
        fakeness_map = self.layers_5(output_1)
        signal = self.layers_6(output_2)
        
        return fakeness_map, signal
    

class DetectionModule(nn.Module):
    def __init__(self, num_layers=6, num_features=64):
        super(DetectionModule, self).__init__()
        
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for _ in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Sequential(nn.Conv2d(num_features, 1, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(1),
                                        nn.ReLU(inplace=True)))
        self.layers = nn.Sequential(*layers)
        self.layers2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers4=nn.Sequential(nn.Linear(65536, 512), nn.ReLU(), nn.Linear(512, 256), 
                              nn.ReLU(), nn.Linear(256, 1))

    

    def forward(self, inputs):
        output1 = self.layers(inputs)
        output1 = self.layers2(output1)
        output1 = self.layers3(output1)
        output2 = output1.reshape(output1.size(0), -1)
        output2 = self.layers4(output2)
        
        return output2
        