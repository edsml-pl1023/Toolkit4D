from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='ToolKit4D',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    # description='A brief description of your package',
    author='Peiyi Leng',  # Replace with your name
    # author_email='your.email@example.com',  # Replace with your email
    # url='https://github.com/yourusername/your-repo',  # Replace with your URL
    python_requires='>=3.6',  # Specify the Python version compatibility
)
