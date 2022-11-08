from setuptools import setup
from setuptools import find_packages

package_name = 'snapping_turtle_2'

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
    maintainer='patrick',
    maintainer_email='patkar07.pk@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'snap_snap = snapping_turtle_2.snap_snap:main',
            'chomp_chomp = snapping_turtle_2.chomp_chomp:main',
        ],
    },
)
