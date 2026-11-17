def init_lstm(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias_ih" in name:
            p.data.fill_(0)
            # Set forget-gate bias to 1
            n = p.size(0)
            p.data[(n // 4) : (n // 2)].fill_(1)
        elif "bias_hh" in name:
            p.data.fill_(0)
