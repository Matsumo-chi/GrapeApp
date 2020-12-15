import pyrealsense2 as rs #RealSenseライブラリ
import numpy as np #numpyライブラリ
import cv2 #openCVライブラリ
import open3d as o3d #Open3Dライブラリ

# ストリーム(Depth/Color)の設定
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# パイプラインを作成
pipeline = rs.pipeline()

# ストリーミング開始
profile = pipeline.start(config)

# デプスセンサの深度スケールを取得
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# オブジェクトの背景を削除
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale


# 整列オブジェクト作成
# rs.align は、他のフレームに対する深度フレームのアラインメント
# align_to "は, 深さ方向のフレームを整列させるストリームの種類を指定
align_to = rs.stream.color
align = rs.align(align_to)

# カメラの情報を取得
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

# 保存データ用
num = 0
try:
    while True:
        # 色と奥行きのフレームセットを取得
        frames = pipeline.wait_for_frames()

        # 奥行きフレームをカラーフレームに合わせる
        aligned_frames = align.process(frames)

        # 整列されたフレームを取得する
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # 両方のフレームが有効であることの検証
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color = o3d.geometry.Image(color_image)

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = (depth_image < clipping_distance) * depth_image
        depth = o3d.geometry.Image(depth_image)

        # 背景除去　clipping_distanceよりも遠いピクセルを灰色に設定
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # ポイントクラウドとテクスチャマッピングの生成
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
        pcd = o3d.open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

        #レンダリング
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('aligned_frame', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('aligned_frame', images)
        key = cv2.waitKey(1)
        
        # Sキーを押すと、ポイントクラウド保存
        if key & 0xFF == ord('s'):
            print("Saving to {0}.ply...".format(num))
            o3d.io.write_point_cloud('{0}.ply'.format(num), pcd)
            print("Done")
            num += 1
            
         # RキーでPLYデータ読み込み
        if key & 0xFF == ord('r'): 
            print("read ply points#############################")
            pcd = o3d.io.read_point_cloud("1.ply")
            print(np.asarray(pcd.points))
            o3d.visualization.draw_geometries([pcd])
            
        
        # ESCかQキーを押すと、ウインドウを閉じて終了
        elif key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()