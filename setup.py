import os
from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

this_dir = os.path.dirname(os.path.abspath(__file__))
pip_requirements = parse_requirements(
    os.path.join(this_dir, "requirements.txt"), PipSession()
)
reqs = [pii.requirement for pii in pip_requirements]

readme_path = os.path.join(this_dir, "README.md")

with open(readme_path, "r") as f:
    long_description = f.read()

setup(
    name="matbertner",
    version="0.1",
    description="Training and inference for Materials Science named entity recognition (NER)",
    long_description=long_description,
    url="https://github.com/walkernr/MatBERT_NER",
    author=["Nick Walker"],
    author_email="walkernr@lbl.gov",
    license=None,
    packages=find_packages(),
    package_data={},
    zip_safe=False,
    install_requires=reqs,
    # extras_require=extras_dict,
    classifiers=[],
    # tests_require=extras_list,
    scripts=[],
)
