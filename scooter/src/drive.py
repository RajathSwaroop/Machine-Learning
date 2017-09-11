#!/usr/bin/env python
import rospy
import serial
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import serial
pub = rospy.Publisher('arduino', String, queue_size=10)
action = ""
def callback(msg):
	rospy.loginfo("received a msg from cmd_vel")
	rospy.loginfo("Linear components are [%f, %f, %f]"%(msg.linear.x,msg.linear.y,msg.linear.z))
	rospy.loginfo("Angular components are [%f, %f, %f]"%(msg.angular.x,msg.angular.y,msg.angular.z))
	port = '/dev/ttyACM0'
	action = ""
	#ser = serial.Serial(port=port,baudrate=57600,timeout=1)
	if msg.linear.x > 0 and (msg.angular.z < 0.4 or msg.angular.z > -0.4):
		action = "F"
	elif msg.linear.x > 0:
		action = "F"
	elif msg.linear.x == 0:
		action = "S"
	if msg.angular.z > 0:
		action = action + "L"
	elif msg.angular.z < 0:
		action = action + "R"
	rospy.loginfo(action)
	pub.publish(action)	
	action = ""
	#ser.write(action)
	#ser.write("\n")

def listener():
	rospy.init_node('cmd_vel_listener')
	rospy.Subscriber("/cmd_vel",Twist,callback)
	rate = rospy.Rate(50)
	
	#rospy.init_node('talker', anonymous=True)
	rospy.loginfo(action)
	rate.sleep()
	rospy.spin()
	

if  __name__ == '__main__':
	listener()
