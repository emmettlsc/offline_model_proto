import sys


def main():
    try:
        import torch
    except ImportError as e:
        sys.exit(f"torch not installed: {e}")

    print(f"torch {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda  {cuda}")

    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    print(f"cpu   shape={tuple(x.shape)} sum={x.sum().item()}")

    if cuda:
        print(f"gpu   {torch.cuda.get_device_name(0)}")
        print(f"build cuda {getattr(torch.version, 'cuda', '?')}")
        y = x.to("cuda")
        print(f"gpu   device={y.device} sum={y.sum().item()}")


if __name__ == "__main__":
    main()
