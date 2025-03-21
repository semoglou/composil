from setuptools import setup, find_packages

setup(
    name="statistical_composite_silhouette",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "joblib>=1.4.2"
    ],
    python_requires=">=3.8",
    description="A robust clustering evaluation framework that combines micro- and macro-averaged silhouette scores into a composite metric using statistical weighting.",
    url="https://github.com/semoglou/statistical_composite_silhouette",  
    author="Angelos Semoglou",
    author_email="a.semoglou@outlook.com",
    license="Apache-2.0"
)
