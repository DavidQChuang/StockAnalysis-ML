wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torchvision-0.21.0%2Brocm6.4.1.git4040d51f-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp310-cp310-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torchaudio-2.6.0%2Brocm6.4.1.gitd8831425-cp310-cp310-linux_x86_64.whl
pip3 uninstall torch torchvision pytorch-triton-rocm
pip3 install torch-2.6.0+rocm6.4.1.git1ded221d-cp310-cp310-linux_x86_64.whl torchvision-0.21.0+rocm6.4.1.git4040d51f-cp310-cp310-linux_x86_64.whl torchaudio-2.6.0+rocm6.4.1.gitd8831425-cp310-cp310-linux_x86_64.whl pytorch_triton_rocm-3.2.0+rocm6.4.1.git6da9e660-cp310-cp310-linux_x86_64.whl