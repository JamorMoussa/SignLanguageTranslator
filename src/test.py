

class Module:

    def forward(self): ...

    def __call__(self, msg : str):
        self.forward(msg)


class NeuralNet(Module):

    def __init__(self): 
        pass 

    def forward(self, x):
        print("From forward")
        print(x)




model = NeuralNet()("Anas")