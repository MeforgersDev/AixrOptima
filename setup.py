from setuptools import setup, find_packages

setup(
    name="AixrOptima",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.0",
        "fairscale>=0.4.13"
    ],
    description="AixrOptima: LLM optimization techniques (LoRA, quantization, quantum-inspired optimization)",
    author="MeforgersDev",
    license="MIT",
    python_requires=">=3.7"
)
