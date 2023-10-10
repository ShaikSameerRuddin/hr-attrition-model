from setuptools import find_packages,setup
from typing import List



def get_requirements(filePath : str) -> List[str]:
    HYPHEN_E_DOT='-e .'
    requirements = []
    with open(filePath) as file_obj:
        requirements=file_obj.readline()
        requirements=[req.replace('/n', "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)



setup(
    name='hr-attrition-model',
    version='0.0.1',
    author='Sameer Ruddin Shaik',
    author_email="sameerruddinshaik@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)