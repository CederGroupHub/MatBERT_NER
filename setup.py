import os
from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

this_dir = os.path.dirname(os.path.abspath(__file__))
pip_requirements = parse_requirements(
    os.path.join(this_dir, "requirements.txt"), PipSession()
)

reqs_clean = []
# Ensure git-required repositories on github are fetched correctly
for req in pip_requirements:
    r = req.requirement
    if r == "git+https://github.com/pabloppp/pytorch-tools/@0.2.7#egg=torchtools":
        r = "torchtools @ " + r
    reqs_clean.append(r)

print(reqs_clean)

# raise ValueError

readme_path = os.path.join(this_dir, "README.md")

with open(readme_path, "r") as f:
    long_description = f.read()

setup(
    name="matbert_ner",
    version="0.2",
    description="Training and inference for Materials Science named entity recognition (NER)",
    long_description=long_description,
    url="https://github.com/walkernr/MatBERT_NER",
    author=["Nick Walker"],
    author_email="walkernr@lbl.gov",
    license=None,
    packages=find_packages(),
    package_data={},
    zip_safe=False,
    install_requires=reqs_clean,
    # extras_require=extras_dict,
    classifiers=[],
    # tests_require=extras_list,
    scripts=[],
)
