from setuptools import find_packages, setup

package_name = 'px4_offboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rachit Kaival Shah',
    maintainer_email='rachitshah2901@gmail.com',
    description='ROS 2 offboard control nodes for PX4 (SITL with Gazebo)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'square_mission = px4_offboard.square_mission:main',
            'pd_square = px4_offboard.pd_square:main',
        ],
    },
)
