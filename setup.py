import setuptools

setuptools.setup(
    name="minitouch",
    version="0.0.1",
    description="MiniTouch benchmark",
    project_urls={
        "Source": "https://github.com/ServiceNow/MiniTouch",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    eager_resources=["*"],
    include_package_data=True,
    install_requires=['gymnasium==0.29.1', 'pybullet'],
    python_requires=">=3.8",
)