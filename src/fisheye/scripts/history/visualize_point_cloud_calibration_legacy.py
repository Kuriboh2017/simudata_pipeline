import streamlit as st
import open3d as o3d
import numpy as np
import os
import sys
import tifffile
import time
import matplotlib.colors as mcolors
import threading
import csv

def read_csv(folder):
    csv_file_path=os.path.join(folder,"data.csv")
    data_file_path=os.path.join(folder,"data")
    timestamp_map={}
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            timestamp_map[row[0]]=os.path.join(data_file_path,row[1])
    return timestamp_map

def depth_map_to_point_cloud(depth_map, intrinsic_matrix, depth_scale=1.0, color="#FF0000"):

    height, width = depth_map.shape
    point_cloud = o3d.geometry.PointCloud()

    fx, fy, cx, cy = intrinsic_matrix[0], intrinsic_matrix[1], intrinsic_matrix[2], intrinsic_matrix[3]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) * depth_map / fx
    y = (v - cy) * depth_map / fy
    z = depth_map * depth_scale
    
    points = np.stack([x, y, z], axis=-1)
    
    rgb_color = mcolors.to_rgba(color)[:3]
    colors = np.tile(rgb_color, (height, width, 1))

    point_cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    
    return point_cloud

class Open3DVisualization:
    
    def __init__(self):
        
        print("init of Open3DVisualization")
        self.shouldClose = False
        self.desired_num_boxes = 3
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.point_clouds={}

        # Create coordinate frame at camera position
        camera_position = [0, 0, 0]  # Replace with actual camera position
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8, origin=camera_position)
        self.vis.add_geometry(coordinate_frame)
        
        # Set camera view and zoom
        self.view_control = self.vis.get_view_control()
        self.control_view(5)
        render_option = self.vis.get_render_option()
        # render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate  # set color as the Z-axis coordinate.
        render_option.point_size = 2.0
        
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
    
    def __del__(self):
        
        self.vis.destroy_window()
        self.thread.join()
    
    def control_view(self,zoom):
        self.view_control.set_lookat([0, 0, 0])  # Set the camera's look-at point to the origin
        self.view_control.set_front([1, 0, 0])   # Set the camera's front direction to the positive X-axis
        self.view_control.set_up([0, 1, 0])      # Set the camera's up direction to the positive Y-axis
        self.view_control.set_zoom(zoom)          # Set the camera's zoom level
    
    def update_point_cloud_color(self, index, color_str):
        if index in self.point_clouds:
            rgb_color = mcolors.to_rgba(color_str)[:3]
            color_array = np.array(rgb_color, dtype=np.float32).reshape(3, 1)
            self.point_clouds[index].paint_uniform_color(color_array)
            self.vis.update_geometry(self.point_clouds[index])
        else:
            print(f"Point cloud with index {index} not found.")
            

    def remove_point_cloud(self, render_index):
        print(f"remove_point_cloud:{render_index}")
        self.vis.remove_geometry(self.point_clouds[render_index])
        del self.point_clouds[render_index]
        self.control_view(0.06)

    def update_num_boxes(self, num_boxes):
        
        self.desired_num_boxes = num_boxes
        print(f"new box size {num_boxes}")
    
    def update_color(self, color):
        
        self.desired_color = color

    def pointcloud_load(self,index,point_cloud):
        self.point_clouds[index]=point_cloud
        self.vis.add_geometry(point_cloud)
        self.control_view(0.06)
        self.vis.update_renderer()
        # self.vis.update_renderer()
        # Set camera view and zoom


    def run_pointcloud(self, data, intrinsic_matrix, render_index):
        color = self.desired_color
        point_cloud = depth_map_to_point_cloud(data, intrinsic_matrix, color=color)
        self.vis.add_geometry(point_cloud)

        self.point_clouds[render_index]=point_cloud
        self.control_view(0.06)
        # self.vis.update_renderer()
        time.sleep(0.03)
        return point_cloud

    def run(self):
        
        num_boxes = -1
        mesh_boxes = [] 
        while not self.shouldClose:
            self.shouldClose = not self.vis.poll_events()
            
            if self.desired_num_boxes !=  num_boxes:
                
                num_boxes = self.desired_num_boxes

                for mesh_box in mesh_boxes:
                    self.vis.remove_geometry(mesh_box)
                
                colors = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]  # Red, Green, Blue colors
                for i in range(num_boxes):
                    color = colors[i % len(colors)]  # Select the color in a cyclic manner
                    position = [0, 0, i * 0.5]  # Adjust the Z position based on the index
                    mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)
                    mesh_box.compute_vertex_normals()
                    mesh_box.paint_uniform_color(color)
                    mesh_box.translate(position)
                    
                    self.vis.add_geometry(mesh_box)
                    mesh_boxes.append(mesh_box)
                
                self.control_view(0.06)
            self.vis.update_renderer()
            # self.vis.run()
            time.sleep(0.03)

def main():
    # for filename in os.listdir(path_in):
    temp_dir=""
    temp_file_path=""
    point_cloud3=o3d.geometry.PointCloud()
    temp_dir_bigModel=""
    temp_file_path_bigModel=""
    point_cloud4=o3d.geometry.PointCloud()
    lock=threading.Lock()
    
    col1, col2, col3, col4 = st.columns(4)
    render_point_cloud1 = col1.checkbox("自标定深度图")
    render_point_cloud2 = col2.checkbox("标定参数深度图")
    render_point_cloud3 = col3.checkbox("三维重建点云")
    render_point_cloud4 = col4.checkbox("大模型点云")

    # if render_point_cloud1 or render_point_cloud2 or render_point_cloud3 or render_point_cloud4:
    if "vis" not in st.session_state:
        print("initialise new Open3DVisualization")
        vis = Open3DVisualization()
        st.session_state["vis"] = vis
    else:
        print("reuse old Open3DVisualization")
        vis = st.session_state["vis"]

    
    #upload self_calib depthmap
    # render_point_cloud1 = st.checkbox("渲染自标定深度图")
    depthmap_cloud_file1 = st.file_uploader("上传自标定深度图文件:", type=["tiff"])

    if depthmap_cloud_file1 is not None and render_point_cloud1:
        image = tifffile.imread(depthmap_cloud_file1)
        data = np.array(image)
        intrinsic_matrix = [325.18002440423413, 325.1285404384907, 560.1266459707691, 560.6442738332462]

        color1 = st.color_picker("选择自标定深度图的颜色:", "#FF0000")

        if "pointcloud1" not in st.session_state:
            vis.update_color(color1)
            with lock:
                vis.run_pointcloud(data, intrinsic_matrix, render_index=1)
            st.session_state["pointcloud1"] = True
            print(f"pointcloud_size:{len(vis.point_clouds)}")
            
        elif color1 != "#FF0000":
            with lock:
                print(f"pointcloud_size:{len(vis.point_clouds)}   update color")
                vis.update_point_cloud_color(1, color1)
        
    elif not render_point_cloud1 and "pointcloud1" in st.session_state and  st.session_state["pointcloud1"]:
        with lock:
            vis.remove_point_cloud(1)
            del st.session_state["pointcloud1"]

    #upload cfg_calib depthmap     
    depthmap_cloud_file2 = st.file_uploader("上传默认标定参数深度图文件:", type=["tiff"])

    if depthmap_cloud_file2 is not None and render_point_cloud2:
        image = tifffile.imread(depthmap_cloud_file2)
        data = np.array(image)
        intrinsic_matrix = [325.18002440423413, 325.1285404384907, 560.1266459707691, 560.6442738332462]

        color2 = st.color_picker("选择默认标定参数深度图的颜色:", "#2400FF")

        if "pointcloud2" not in st.session_state:
            vis.update_color(color2)
            with lock:
                vis.run_pointcloud(data, intrinsic_matrix, render_index=2)
            st.session_state["pointcloud2"] = True
        elif color2 != "#2400FF":
            with lock:
                vis.update_point_cloud_color(2, color2)
            
    elif not render_point_cloud2 and "pointcloud2" in st.session_state and  st.session_state["pointcloud2"]:
        with lock:
            vis.remove_point_cloud(2)
            del st.session_state["pointcloud2"]

    #upload 3d point_cloud
    #render_point_cloud3 = st.checkbox("显示三维重建点云")
    point_cloud_file = st.file_uploader("上传三维重建点云文件:", type=["xyz", "ply"])

    if  render_point_cloud3 and point_cloud_file is not None :
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, point_cloud_file.name)
        with open(temp_file_path, "wb") as file:
            file.write(point_cloud_file.getbuffer())
        
        if "pointcloud3" not in st.session_state:
            point_cloud3 = o3d.io.read_point_cloud(temp_file_path)
            with lock:
                vis.pointcloud_load(3,point_cloud3)
            st.session_state["pointcloud3"]=True
    
    elif not render_point_cloud3 and "pointcloud3" in st.session_state:
        with lock:
            vis.remove_point_cloud(3)
            del st.session_state["pointcloud3"]

    #upload bigModel point_cloud
    #render_point_cloud4 = st.checkbox("显示大模型点云")
    point_cloud_file_bigModel = st.file_uploader("上传大模型点云文件:", type=["xyz", "ply"])

    if  render_point_cloud4 and point_cloud_file_bigModel is not None :
        temp_dir_bigModel = "./temp_bigModel"
        os.makedirs(temp_dir_bigModel, exist_ok=True)
        temp_file_path_bigModel = os.path.join(temp_dir_bigModel, point_cloud_file_bigModel.name)
        with open(temp_file_path_bigModel, "wb") as file:
            file.write(point_cloud_file_bigModel.getbuffer())
        
        if "pointcloud4" not in st.session_state:
            point_cloud4 = o3d.io.read_point_cloud(temp_file_path_bigModel)
            with lock:
                vis.pointcloud_load(4,point_cloud4)
            st.session_state["pointcloud4"]=True
    
    elif not render_point_cloud4 and "pointcloud4" in st.session_state:
        with lock:
            vis.remove_point_cloud(4)
            del st.session_state["pointcloud4"]

    # if "num_boxes" in st.session_state:
    #     del st.session_state["num_boxes"]
    st.number_input("请输入正方体个数(边长0.5m):", min_value=1, value=3, step=1, key='num_boxes')
    
        
    if "vis" in st.session_state: 
        vis.update_num_boxes(st.session_state['num_boxes'])
        
    # vis.run()
    #rm temp files
    # os.remove(temp_file_path)
    # os.rmdir(temp_dir) 
    # os.remove(temp_file_path_bigModel)
    # os.rmdir(temp_dir_bigModel) 

if __name__ == "__main__":
    
    if len(sys.argv)<1:
        sys.stdout.write("Usage: streamlit run point_cloud_visualize.py\n")
    main()
    # euroc_dataset = st.text_input("请选择Euroc数据集路径:","/home/r23092/work/data/truth_dataset/test/sensors_data_2023.06.15-14.34.50_out_euroc")

    # euroc_dataset = os.path.expanduser('/home/r23092/work/data/truth_dataset/test/sensors_data_2023.06.15-14.34.50_out_euroc')


    # with st.form(key="directory_form"):
    #     st.text_input('Euroc Dataset Directory', euroc_dataset, key='containing_foler')
    #     directory_form_submitted = st.form_submit_button("Confirm")
    

    # if directory_form_submitted:  
    #     pointcloud_dataset=os.path.join(euroc_dataset,"mav0/pointcloud_left")
    #     pointcloud_msg=read_csv(pointcloud_dataset)
    #     st.session_state["pointcloud_msg"]=pointcloud_msg
    #     # pointcloud_idx = st.slider("image index", 0, success_num-1, key="success pointcloud")

    # if "pointcloud_msg" in st.session_state:
    #     pointcloud_msg=st.session_state["pointcloud_msg"]
    #     timestamp=st.select_slider("Select Point_cloud Timestamp(Success)", options=pointcloud_msg.keys(), key="selected_timestamp")
