# It is used to build the package and install it.

from setuptools import setup, find_packages
from typing import List

# This function is used to get the requirements from the requirements.txt file
def get_requirements_from_file(filepath:str)->List[str]:
    requirements = []
    
    # This is used to read the requirements.txt file
    with open(filepath) as file:
        requirements = file.readlines()
        
        # This is used to remove the \n from the requirements
        requirements = [req.replace('\n','') for req in requirements]
        
        if '-e .' in requirements:
            requirements.remove('-e .')
            
    return requirements
        
        
    

setup(
        name='EduLens',
        version='0.1',
        author= "Nithin Reddy",
        author_email= "nithinreddy1747@gmail.com",
        packages=find_packages(),
        install_requires = get_requirements_from_file('requirements.txt'),
        description='A package for educational data analysis',
        
    )