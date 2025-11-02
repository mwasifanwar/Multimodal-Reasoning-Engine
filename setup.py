from setuptools import setup, find_packages

setup(
    name="multimodal-reasoning-engine",
    version="1.0.0",
    author="mwasifanwar",
    description="Advanced AI that can reason across text, images, audio, and video simultaneously",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "opencv-python>=4.7.0",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
    ],
    python_requires=">=3.8",
)