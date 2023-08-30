import setuptools

setuptools.setup(
    name="Kiran Sathynarayana",
    version="v0.3",
    author="Kiran Sathyanarayana",
    author_email="kiran.sathyanara@tigeranalytics.com",
    description=" ML package",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
