import rospy
import json
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
import tf.transformations


def get_objects_names_shapes_and_poses(SIM):
    state_string = SIM.get_state()
    state_json = json.loads(state_string)
    entities = state_json["entities"]
    
    objects_names = []
    objects_poses = []
    objects_shapes = []
    
    for entity in entities:
        # print(entity)
        if entity["instance_id"] not in ["camera_0"]:
            
            objects_names.append(entity["instance_id"])
            
            # pose_msg = Pose()
            # print(entity["position"])
            # pose_msg.position.x = entity["position"][0]
            # pose_msg.position.y = entity["position"][1]
            # pose_msg.position.z = entity["position"][2]

            # quaternion = tf.transformations.quaternion_from_euler(entity["euler_xyzr"][0], entity["euler_xyzr"][1], entity["euler_xyzr"][2])
            # print(quaternion)
            # pose_msg.orientation.x = quaternion[0]
            # pose_msg.orientation.y = quaternion[1]
            # pose_msg.orientation.z = quaternion[2]
            # pose_msg.orientation.w = quaternion[3]
            
            # objects_poses.append(pose_msg)

                        
            primitive = SolidPrimitive()
            if entity['collision_shapes']:
                # print("Count collision shapes: ", len(entity['collision_shapes']))
                if entity['collision_shapes'][0]['type'] == "RCSSHAPE_SSL":
                    primitive.type = SolidPrimitive.CYLINDER
                    # print("extents: ", entity['collision_shapes'][0]['extents'])
                    primitive.dimensions = [entity['collision_shapes'][0]['extents'][2], entity['collision_shapes'][0]['extents'][0]] 
                    objects_shapes.append(primitive)
                if entity['collision_shapes'][0]['type'] == "RCSSHAPE_BOX":
                    primitive.type = SolidPrimitive.BOX
                    primitive.dimensions = [entity['collision_shapes'][0]['extents'][0],entity['collision_shapes'][0]['extents'][1], entity['collision_shapes'][0]['extents'][2]]
                    objects_shapes.append(primitive)
                if entity['collision_shapes'][0]['type'] == "RCSSHAPE_CYLINDER":
                    primitive.type = SolidPrimitive.CYLINDER
                    primitive.dimensions = [entity['collision_shapes'][0]['extents'][2], entity['collision_shapes'][0]['extents'][0]]
                    objects_shapes.append(primitive)
                position = entity['collision_shapes'][0]["position"]
                orientation = entity['collision_shapes'][0]["euler_xyzr"]
                
                print("entity:", entity["instance_id"])
                print("Posiition:", position)
                print("Orientation: ", orientation)
                pose_msg = Pose()
                if position[1] < 0:
                    position[1] += 0.07
                    position[0] += 0.03
                # if entity["instance_id"] == "bottle_of_milk":
                #     position = [-0.2261419556711581, -0.23116066090970393, 0.7157461214988561]
                #     orientation = [0.03879707047234144, -0.06547503007851327, 2.4638475419403982]
                if entity["instance_id"] == "small_bowl":
                    position = [-0.00179565891543248, -0.3087222454434205, 0.6614663094230591]
                    orientation = [0.08186882436503265, 0.21141441525785168, 2.4128136097345254]


                pose_msg.position.x = position[0]
                pose_msg.position.y = position[1]
                pose_msg.position.z = position[2]
                quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])
                print("quaternion: ", quaternion)
                pose_msg.orientation.x = quaternion[0]
                pose_msg.orientation.y = quaternion[1]
                pose_msg.orientation.z = quaternion[2]
                pose_msg.orientation.w = quaternion[3]
                objects_poses.append(pose_msg)
    return objects_names, objects_poses, objects_shapes                          