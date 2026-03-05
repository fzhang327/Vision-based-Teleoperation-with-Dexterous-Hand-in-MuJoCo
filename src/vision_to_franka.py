import cv2
import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig

# 强壮导入法：明确指出我们要用 solutions
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def main():
    # 1. 初始化 Dex-Retargeting 优化器
    print("正在加载逆运动学配置...")
    config = RetargetingConfig.load_from_file("../configs/franka_teleop.yml")
    retargeting = config.build()
    
    # 2. 初始化 MediaPipe 视觉
    print("正在初始化视觉模型...")
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    # 3. 空间映射基准点
    # 将手部的移动比例放大，并平移到 FR3 的真实物理世界前
    workspace_scale = 1.0  
    robot_base_offset = np.array([0.4, 0.0, 0.3]) # 设定操作中心在底座正前方40cm，高30cm处

    print("=======================================")
    print("系统已就绪！请将手放到摄像头前...")
    print("注意：终端里如果弹出找不到 mesh/stl 的警告，请直接无视！")
    print("=======================================")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            print("无法获取摄像头画面！")
            break

        frame = cv2.flip(frame, 1) # 镜像翻转，操作更直观
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 提取手部特征
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 画出骨架连线，方便直观确认摄像头有没有抓准
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 提取 0 号点（手腕）的原始坐标
            wrist = hand_landmarks.landmark[0]
            
            # MediaPipe 坐标系转机器人右手坐标系
            # 假设摄像头画面中心 (0.5, 0.5) 对应机器人的操作中心
            raw_x = (wrist.x - 0.5) 
            raw_y = -(wrist.y - 0.5) # 反转 Y 轴
            raw_z = -wrist.z         # 深度信息

            # 计算出希望 FR3 末端去到的物理绝对坐标 (米)
            target_xyz = np.array([raw_x, raw_y, raw_z]) * workspace_scale + robot_base_offset
            
            # --- 欺骗优化器的关键一步 ---
            # 因为 YAML 里配置了两次 fr3_link8，这里必须把目标坐标复制成两份，规避降维 Bug
            target_xyz_stacked = np.vstack([target_xyz, target_xyz])
            
            # 见证奇迹：底层 C++ 非线性优化器瞬间吐出平滑的 7 个关节角
            q_des = retargeting.retarget(target_xyz_stacked)
            
            print(f"目标 XYZ: {target_xyz.round(3)} ---> FR3 关节角: {q_des.round(2)}")

        # 显示摄像头画面
        cv2.imshow('Vision to Franka Pipeline', frame)
        
        # 选中画面窗口，按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
