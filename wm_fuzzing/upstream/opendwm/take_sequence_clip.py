# Extracted from SenseTime-FVG/OpenDWM at b0ecc3d4020612376ea5a87500f98bc76893428f.
# src/dwm/functional.py lines 172–181.

def take_sequence_clip(item, start: int, stop: int):
    if isinstance(item, (int, float, bool, str)):
        return item
    elif isinstance(item, torch.Tensor):
        return item if len(item.shape) <= 1 else item[:, start:stop]
    elif isinstance(item, list):
        assert len(item) > 0 and all([isinstance(i, list) for i in item])
        return [i[start:stop] for i in item]
    else:
        raise Exception("Unsupported type to take sequence clip.")
