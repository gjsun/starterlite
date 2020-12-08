from __future__ import print_function
import os, io, re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


starter_link = 'https://github.com/gjsun/starterlite'

setup(name='starterlite',
      version='0.1',
      description='Scientific Toolbox for Analyzing and Representing Tomography of the Epoch of Reionization',
      author='Guochao (Jason) Sun',
      author_email='jsun.astro@gmail.com',
      url=starter_link,
      packages=['starterlite', 'starterlite.analysis', 'starterlite.physics', 'starterlite.simulation', 'starterlite.util'],
     )

# Set up $HOME/.starterlite
HOME = os.getenv('HOME')
if not os.path.exists('{!s}/.starterlite'.format(HOME)):
    try:
        os.mkdir('{!s}/.starterlite'.format(HOME))
    except:
        pass

# Create files for defaults and labels in HOME directory
for fn in ['defaults', 'labels']:
    if not os.path.exists('{0!s}/.starterlite/{1!s}.py'.format(HOME, fn)):
        try:
            f = open('{0!s}/.starterlite/{1!s}.py'.format(HOME, fn), 'w')
            print("pf = {}", file=f)
            f.close()
        except:
            pass