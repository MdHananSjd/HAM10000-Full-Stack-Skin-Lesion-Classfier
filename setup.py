from setuptools import setup, find_packages

setup(
    name="skin_lesion_classifier",
    version="0.1.0",
    author="Hanan",
    description="An end-to-end skin lesion classification system using ResNet50 and ONNX.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "albumentations>=1.3.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0"
    ],
)