import cv2
import numpy as np
import time
import mujoco
import mujoco.viewer
from dex_retargeting.retargeting_config import RetargetingConfig
import mediapipe as mp
from dm_control import mjcf 

def main():
    print("正在组装『弗兰卡-Aero』复合体...")
    arm = mjcf.from_path("../models/mujoco_menagerie/franka_emika_panda/panda.xml")
    hand = mjcf.from_path("../models/mujoco_menagerie/tetheria_aero_hand_open/right_hand.xml")
    
    if getattr(hand, 'option', None) is not None:
        hand.option.integrator = arm.option.integrator

    original_hand = arm.find('body', 'hand')
    if original_hand is not None:
        for geom in original_hand.find_all('geom'):
            geom.rgba = [0, 0, 0, 0]
            geom.material = None
            geom.conaffinity = 0
            geom.contype = 0
            
    link7 = arm.find('body', 'link7')
    attachment_site = link7.add('site', name='my_custom_site', 
                                pos=[0, 0, 0.107], 
                                euler=[0, -1.5708, 0])
    attachment_site.attach(hand)
    
    print("正在编译底层物理世界...")
    model = mujoco.MjModel.from_xml_string(arm.to_xml_string(), arm.get_assets())
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
    last_q_des = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

    print("=======================================")
    print("【真实动力学：腱绳电机驱动】已激活！")
    print("=======================================")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while cap.isOpened() and viewer.is_running():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            status_text = "NO HAND"
            status_color = (0, 0, 255) 
            finger_closures = {'thumb': 0.0, 'index': 0.0, 'middle': 0.0, 'ring': 0.0, 'pinky': 0.0}

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                
                def get_closure(tip_idx, mcp_idx, is_thumb=False):
                    tip = hand_landmarks.landmark[tip_idx]
                    mcp = hand_landmarks.landmark[mcp_idx]
                    ext_dist = np.sqrt((wrist.x - tip.x)**2 + (wrist.y - tip.y)**2)
                    base_dist = np.sqrt((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2)
                    
                    if is_thumb:
                        index_mcp = hand_landmarks.landmark[5]
                        thumb_dist = np.sqrt((tip.x - index_mcp.x)**2 + (tip.y - index_mcp.y)**2)
                        ratio = thumb_dist / (base_dist + 1e-6)
                        return np.clip((1.5 - ratio) / 1.0, 0.0, 1.0)
                    else:
                        ratio = ext_dist / (base_dist + 1e-6)
                        return np.clip((2.0 - ratio) / 0.8, 0.0, 1.0)

                finger_closures['thumb'] = get_closure(4, 2, is_thumb=True)
                finger_closures['index'] = get_closure(8, 5)
                finger_closures['middle'] = get_closure(12, 9)
                finger_closures['ring'] = get_closure(16, 13)
                finger_closures['pinky'] = get_closure(20, 17)
                
                middle_mcp = hand_landmarks.landmark[9]
                hand_size = np.sqrt((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)
                extension_ratio = np.sqrt((wrist.x - hand_landmarks.landmark[12].x)**2 + (wrist.y - hand_landmarks.landmark[12].y)**2) / (hand_size + 1e-6)
                
                if extension_ratio > 1.5:
                    status_text = "ACTIVE (Tendon Dynamics)"
                    status_color = (0, 255, 0)
                    size_diff = hand_size - 0.12 
                    target_xyz_stacked = np.vstack([[
                        np.clip(size_diff * 4.0 + 0.4, 0.3, 0.65), 
                        -(wrist.x - 0.5) * workspace_scale + 0.0, 
                        -(wrist.y - 0.5) * workspace_scale + 0.3
                    ]] * 2)
                    last_q_des = retargeting.retarget(target_xyz_stacked)
                else:
                    status_text = "CLUTCH ENGAGED"
                    status_color = (0, 255, 255) 

                # --- 物理注入层：与真实电机对话 ---
                with viewer.lock():
                    # 1. 向真实存在的电机（Actuators）下发指令，而不是霸道地强扭关节
                    for i in range(model.nu):
                        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                        if not act_name: continue
                        name_lower = act_name.lower()
                        
                        target_closure = None
                        if 'thumb' in name_lower:
                            target_closure = finger_closures['thumb']
                        elif 'index' in name_lower:
                            target_closure = finger_closures['index']
                        elif 'middle' in name_lower:
                            target_closure = finger_closures['middle']
                        elif 'ring' in name_lower:
                            target_closure = finger_closures['ring']
                        elif 'pinky' in name_lower or 'little' in name_lower:
                            target_closure = finger_closures['pinky']
                            
                        if target_closure is not None:
                            # 提取 XML 中这台电机的运行物理极限（极大可能是力矩或伺服位置限制）
                            ctrl_min = model.actuator_ctrlrange[i][0]
                            ctrl_max = model.actuator_ctrlrange[i][1]
                            # 将你的 0~1 意图转换为驱动器的实际脉冲
                            data.ctrl[i] = ctrl_min + (1.0 - target_closure) * (ctrl_max - ctrl_min)

                    # 2. 物理时间演算 (Time Integration)
                    # 因为我们在跑 30fps 的摄像头流，必须连跑约 15 步仿真来同步现实时间，腱绳才能扯拽到位！
                    for _ in range(15):
                        # 机械臂目前继续使用神明模式，强行锁定坐标作为稳固的手臂底座
                        data.qpos[:7] = last_q_des.flatten()
                        data.qvel[:7] = 0.0  # 抽干手臂速度，防止垮塌
                        mujoco.mj_step(model, data) 
                
                viewer.sync()

            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
            cv2.imshow('Vision Teleop - True Dynamics', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()