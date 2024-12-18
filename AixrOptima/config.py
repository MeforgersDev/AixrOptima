class AixrOptimaConfig:
    def __init__(self,
                 rank=4,
                 bits=4,
                 use_quantization=True,
                 per_channel_quant=True,
                 lambda_reg=1e-4,
                 lr=1e-3,
                 initial_temp=1.0,
                 cooling_rate=0.99):
        self.rank = rank
        self.bits = bits
        self.use_quantization = use_quantization
        self.per_channel_quant = per_channel_quant
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
