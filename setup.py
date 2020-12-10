from setuptools import setup, find_packages


setup(
    name="procyclist",
    version="1",
    packages=find_packages(),

    install_requires=[
        'matplotlib==1.5.3',
        'pandas==0.19.0',
        'scipy==0.18.1',
        'tensorflow==2.4.0rc4',
    ],
    extras_require={
        'GPU': ['tensorflow-gpu']
    },

    # Include config and parameter name files
    package_data={
        'procyclist': ['config/*.cfg', 'config/*.csv'],
    },

    entry_points={
        'console_scripts': [
            'procyclist_train = procyclist.train:main'
        ]
    },

    author="Agrin Hilmkil",
    license="MIT",
    keywords="cycling",
    url="https://github.com/agrinh/procyclist_performance"
)
