import types

class Lambda(nn.Module):
    def __init__(self, lambd: callable):
        """Creates a module that applies a lambda function to its input.

        Taken from: https://discuss.pytorch.org/t/a-small-snippet-for-lambda-modules/38590

        Args:
            lambd (callable): Lambda function to apply to input.
        """
        super().__init__()

        assert type(lambd) is types.LambdaType, "lambd must be a lambda function"

        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
