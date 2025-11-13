# DeepSeek-OCR Companion Files

Notebooks and examples related to our DeepSeek-OCR videos:
https://www.youtube.com/@Integral_Business_Intelligence

## What's Here

- **DeepSeek-OCR Research Paper** - The original paper from DeepSeek
- **00_setup_environment.ipynb** - Runpod environment setup notebook
- **01_vllm-two-pass.ipynb** - Two-pass OCR pipeline for batch PDF processing
- **deepseek_ocr_paper.md** - The DeepSeek-OCR paper processed through its own model

## Usage

1. Run the setup notebook to prepare your environment
2. Upload PDFs to `/workspace/pdfs`
3. Run the processing notebook

The pipeline does two-pass OCR with figure extraction and description generation.

## Notes

These are companion materials for our YouTube channel. The notebooks assume a Runpod-style environment but can be adapted to other setups.
