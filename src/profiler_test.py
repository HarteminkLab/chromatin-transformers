import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


def main():

    print("Loading resnet")
    model = models.resnet18()
    inputs = torch.randn(2, 3, 24, 24)

    print("Profiling")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


if __name__ == '__main__':
    main()
