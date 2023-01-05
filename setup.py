from setuptools import find_packages, setup
from pathlib import Path

this_path = Path(__file__).parent

readme_path = this_path / "README.md"
requirements_path = this_path / "requirements.txt"

long_description = readme_path.read_text(encoding="utf-8")

with open(requirements_path, "r", encoding="utf-8") as requirements_file:
    requirements = requirements_file.read().splitlines()

if __name__ == "__main__":
    setup(
        name="speechline",
        description="A speech labeling pipeline",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="w11wo",
        author_email="wilson@bookbotkids.com",
        url="https://github.com/bookbot-kids/speech-pipeline-v2",
        license="Apache License",
        packages=find_packages(),
        install_requires=requirements,
        include_package_data=True,
        platforms=["linux", "unix", "windows"],
        python_requires=">=3.7",
    )
