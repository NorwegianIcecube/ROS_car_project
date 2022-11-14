import subprocess


def bash_command(cmd):
    return subprocess.Popen(['/bin/bash', '-c', cmd], shell=False)


commands = [
            "echo 'source /opt/ros/foxy/setup.bash' >> ~/.bashrc",
            "export TURTLEBOT3_MODEL=waffle_pi",
            "source ~/ros2_ws/install/setup.bash",
            "ros2 run turtlebot turtlebot_controller_publisher",
            ]

execute_command = "; ".join(commands)

print(execute_command)

p1 = bash_command(execute_command)
p1.wait()

if p1.returncode == 0:
    print("Command : Success")

else:
    print("Command : Failed")
    
