import cv2
import numpy as np
import time
import mujoco
import mujoco.viewer
from dex_retargeting.retargeting_config import RetargetingConfig

import mediapipe as mp

def main():
    print("正在加载 MuJoCo 物理世界...")
    xml_path = "../models/mujoco_menagerie/franka_emika_panda/panda.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("正在加载逆运动学大脑...")
    config = RetargetingConfig.load_from_file("../configs/franka_teleop.yml")
    retargeting = config.build()

    print("正在初始化视觉网络...")
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)

    workspace_scale = 1.0  
    
    # 记录机械臂最后的安全状态（防止握拳时发生跳变）
    last_q_des = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

    print("=======================================")
    print("带离合器的阿凡达模式已激活！")
    print("操作逻辑：")
    print(" - [张开手] -> 机械臂跟随你的手移动")
    print(" - [握紧拳] -> 触发离合，机械臂定住不动 (此时你可以随意移动手臂找舒服的姿势)")
    print("=======================================")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        while cap.isOpened() and viewer.is_running():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            # 默认状态
            status_text = "NO HAND"
            status_color = (0, 0, 255) # 红色

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]  # 中指根部
                middle_tip = hand_landmarks.landmark[12] # 中指指尖
                
                # 1. 计算手掌基准大小 (手腕到中指根部)
                hand_size = np.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
                
                # 2. 计算指尖伸展距离 (手腕到中指指尖)
                extension_dist = np.sqrt((wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2)
                
                # 3. 核心：计算伸展比例，判断是否握拳
                extension_ratio = extension_dist / (hand_size + 1e-6)
                
                # 设定阈值：比例大于 1.5 认为是张开手，小于 1.5 认为是握拳
                if extension_ratio > 1.5:
                    status_text = "ACTIVE (Hand Open)"
                    status_color = (0, 255, 0) # 绿色
                    
                    # --- 只有手张开时，才更新目标坐标 ---
                    base_size = 0.12 
                    size_diff = hand_size - base_size
                    
                    dx = wrist.x - 0.5
                    dy = wrist.y - 0.5

                    target_z = -dy * workspace_scale + 0.3
                    target_y = -dx * workspace_scale + 0.0
                    raw_target_x = size_diff * 4.0 + 0.4
                    target_x = np.clip(raw_target_x, 0.3, 0.65)

                    target_xyz = np.array([target_x, target_y, target_z])
                    target_xyz_stacked = np.vstack([target_xyz, target_xyz])
                    
                    # 计算新的关节角，并更新 last_q_des
                    last_q_des = retargeting.retarget(target_xyz_stacked)
                else:
                    status_text = "CLUTCH ENGAGED (Fist)"
                    status_color = (0, 255, 255) # 黄色警告色
                    # 如果握拳，不计算新的 target_xyz，保持 last_q_des 不变

                # --- 物理注入层 ---
                with viewer.lock():
                    # 无论是不是握拳，都持续给底层写入 last_q_des 保持稳定
                    data.qpos[:7] = last_q_des.flatten()
                    mujoco.mj_forward(model, data)
                
                viewer.sync()

            # 在画面左上角极其酷炫地显示当前系统状态
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
            
            cv2.imshow('Vision Teleop - Clutch Mode', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()