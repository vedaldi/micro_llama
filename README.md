
# Micro LLAMA

This is a tiny implementation of the LLAMA 3 model architecture for didactical purposes. The entire implementation is approximately 180 lines of code, hence the name "micro".

The code uses the smallest LLAMA 3 model, i.e., the 8B parameters one. This model is still 15GB in size, and requires about 30GB of memory to execute.
The code by defaults runs this on the CPU, but beware of the memory impact.

Start exploring the code using the notebook `micro_llama.ipynb`.

The model's code itself is entirely contained in the `micro_llama.py` file.

## Requirements

Use the following instruction to create a suitable Conda environment, called `micro_llama`:

```bash
conda env create --file conda-env.yaml --yes
conda activate micro_llama
```

You can get rid of the Conda enviroment as follows:

```bash
conda remove -n micro_llama --all --y
```

## References

This implementation is inspired by:

* [building-llama-3-from-scratch](https://lightning.ai/fareedhassankhan12/studios/building-llama-3-from-scratch)
* [Building-llama3-from-scratch](https://github.com/FareedKhan-dev/Building-llama3-from-scratch)