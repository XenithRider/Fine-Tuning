# Fine-Tuning Project 

This project focuses on the **Supervised Fine-Tuning (SFT)** of Large Language Models (LLMs) using the Hugging Face ecosystem. It specifically demonstrates how to fine-tune the `Llama-3-8B` model using **QLoRA** (Quantized Low-Rank Adaptation) to make high-performance model training accessible on consumer-grade hardware.

##  Features

* **Efficient Fine-Tuning**: Utilizes 4-bit quantization and LoRA to significantly reduce VRAM requirements while maintaining model performance.
* **Unsloth Integration**: Leverages the Unsloth library to speed up the training process and reduce memory usage.
* **Dataset Processing**: Includes scripts for loading and formatting instruction-based datasets (e.g., `vicgalle/alpaca-gpt4`).
* **Inference Pipeline**: Provides code to test the fine-tuned model's performance on custom prompts after training.
* **Model Export**: Supports saving the fine-tuned adapters or merging them with the base model for deployment.

##  Technical Architecture

The fine-tuning pipeline is divided into several key stages:

1. **Environment Setup**: Configuration of high-performance libraries including `unsloth`, `bitsandbytes`, and `peft`.
2. **Model Loading**: Loading a pre-quantized base model (`unsloth/llama-3-8b-bnb-4bit`) to fit within limited GPU memory.
3. **PEFT Configuration**: Applying LoRA adapters to specific linear layers of the transformer model to enable parameter-efficient training.
4. **Training**: Executing the SFT (Supervised Fine-Tuning) trainer using specialized hyperparameters like `learning_rate`, `weight_decay`, and `adamw_8bit` optimizer.
5. **Evaluation**: Generating responses from the fine-tuned model to verify improvements in instruction-following capabilities.


### Usage

1. **Fine-Tuning**: Execute `finetuned001.py` or run the `FineTuned001.ipynb` notebook to start the training process.
2. **Testing**: Use the inference section of the notebook to input custom prompts and observe the model's specialized output.

##  Performance Optimization

* **Memory Management**: By using 4-bit quantization, the model can be trained on a single GPU with as little as 16GB of VRAM.
* **Speed**: Unsloth provides optimized kernels that can make fine-tuning up to 2x faster than standard Hugging Face implementations.
