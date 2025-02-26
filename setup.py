
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Diffusion-OOD", 
    version="0.1",
    author="Robine Thibault",
    author_email="thibaultrobine68@gmail.com",
    description="Diffusion ood benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ThibaultRobine/Diffusion-ood", 
    packages=setuptools.find_packages(), 
    include_package_data=True,
    install_requires=[
        # Dependencies from improved_diffusion
        "blobfile>=1.0.5",
        "torch>=1.13.1",
        "tqdm",

        # Dependencies from OpenOOD
        "torchvision>=0.13",
        "scikit-learn",
        "json5",
        "matplotlib",
        "scipy",
        "pyyaml>=5.4.1",
        "pre-commit",
        "opencv-python>=4.4.0.46",
        "imgaug>=0.4.0",
        "pandas",
        "diffdist>=0.1",
        "Cython>=0.29.30",

        "faiss-gpu>=1.7.2",
        "gdown>=4.7.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
