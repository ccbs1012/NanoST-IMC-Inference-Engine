# NanoST IMC Inference engine

The platform was developed by Prof. Tuo-Hung Hou's group (National Yang Ming Chiao Tung University). This project aims at building an easy-to-use framework for neural network mapping on in memory computing(IMC) accelerator with nonvolatile memorys (NVMs) (e.g., ReRAM, MRAM, etc.).

**_C.-C. Chang, S.-T. Li, T.-L. Pan, C.-M. Tsai, I.-T. Wang, T.-S. Chang and T.-H. Hou, ※[Device Quantization Policy in Variation-Aware In-Memory Computing Design](https://www.nature.com/articles/s41598-021-04159-x)._**

## Dependencies:
  - Python 3.8
  - PyTorch 1.9.0
  - cuDNN



## Usage
1. Quantize IMC model training
```
python train.py
```

2. Inference
```
python inference.py
```


## References related to this tool 
1. github.com/itayhubara/BinaryNet.pytorch
