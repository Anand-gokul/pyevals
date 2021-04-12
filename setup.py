import setuptools
 
with open("README.md", "r") as fh:
    long_description = fh.read()
 
setuptools.setup(
    #Here is the module name.
    name="pyevals",
 
    #version of the module
    version="1.3",
 
    #Name of Author
    author="Gokul and Anand",
 
    #your Email address
    author_email="adsp.tsgkr@gmail.com",
 
    #Small Description about module
    description="A simple Python Package for Model Evalutaion",
 
    long_description=long_description,
 
    #Specifying that we are using markdown file for description
    long_description_content_type="text/markdown",
 
    #Any link to reach this module, if you have any webpage or github profile
    url="https://github.com/Anand-gokul/pyevals",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["sklearn","numpy","pandas","PyPDF2","seaborn","matplotlib"],
    #classifiers like program is suitable for python3, just leave as it is.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)