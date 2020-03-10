import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='acctelemetry',
    version='0.1',
    include_package_data=False,
    license='GPLv2',
    packages=[
        'acctelemetry',
    ],
    package_dir={'acctelemetry': '.'},
    package_data={'acctelemetry': ['templates/acctelemetry/*']},
    # description='A simple Django app to conduct Web-based polls.',
    long_description=README,
    # url='https://www.example.com/',
    # author='Your Name',
    # author_email='yourname@example.com',
    classifiers=[
        'Environment :: Web Environment',
        'Framework :: Django',
        # 'Framework :: Django :: X.Y',  # replace "X.Y" as appropriate
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPLv2',  # example license
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
    ],
)