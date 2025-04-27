# EcomPriceGen: AI-Powered Pricing & Discount Calculator

## Project Overview

EcomPriceGen is an AI-powered model that automates product pricing calculations, especially focusing on discount calculations. Using advanced NLP techniques, the model takes in input prompts with product details (such as price, category, and discount) and generates accurate responses with the final price after applying the discount.

The project uses HuggingFace's **Zephyr-7B** model for fine-tuning with **Low-Rank Adaptation (LoRA)** to efficiently train the model for specific pricing and discounting tasks. The model is trained on an e-commerce dataset with product details like price, discount percentage, and category, which allows the model to answer related queries in real-time.

## Features

- **AI-Powered Pricing**: Generates the final price of products after applying discounts.
- **Custom Fine-Tuning**: The model is fine-tuned using an e-commerce dataset for accurate price calculations.
- **Prompt Engineering**: The model takes human-readable prompts and generates clear, accurate, and relevant responses.
- **Efficient Training**: Uses LoRA for efficient fine-tuning with reduced training time and memory usage.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Setup](#model-setup)
- [Training](#training)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Installation

Ensure that you have all the necessary dependencies installed. You can set up the environment by running the following commands:

```bash
# Clone the repository
git clone https://github.com/yourusername/ecompricegen.git
cd ecompricegen

# Install required dependencies
pip install -r requirements.txt
