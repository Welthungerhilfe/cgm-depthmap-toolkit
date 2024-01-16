from setuptools import setup, find_packages

setup(
    name='depthmap_toolkit',
    version='0.1.0',

    license='GPL',

    author='Shashank Mahale',
    author_email='smahale@childgrowthmonitor.org',

    packages=find_packages(),
    include_package_data=True,

    description="Code for loading and processing depthmaps collected by CGM app",
    long_description=open('README.md').read(),
    long_description_content_type = "text/markdown",

    classifiers=[
        'Intended Audience :: Healthcare Industry',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
    ]
)
