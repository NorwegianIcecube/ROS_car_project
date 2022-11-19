from setuptools import setup
from setuptools import find_packages

package_name = 'turtlebot'

setup(
    name=package_name,
    version='0.0.0',
     packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eirik',
    maintainer_email='eirik@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'turtlebot_controller_publisher = turtlebot.turtlebot_controller_publisher:main'
            'turtlebot_drive = turtlebot.controller_wit_MV:main'
        ],
    },
)
