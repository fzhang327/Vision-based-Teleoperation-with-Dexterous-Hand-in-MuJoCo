# Vision-based Safe Teleoperation with Dexterous Hand in MuJoCo

This repository provides a real-time, vision-based teleoperation framework for the **Franka Emika Panda** manipulator equipped with the underactuated **TetherIA Aero Hand Open**, simulated in **MuJoCo**. 

It utilizes a single RGB web camera to extract human hand landmarks via **MediaPipe** and retargets the Cartesian poses and dexterous finger flexions into the true dynamics (tendon-driven actuators) of the robotic system.

## 🌟 Key Features
* **Zero-Hardware Barrier:** Requires only a standard 2D web camera. No expensive motion-capture suits or VR headsets needed.
* **Monocular Depth Estimation:** Uses 2D palm size variation to intuitively map absolute Z-depth (Forward/Backward).
* **Professional Clutching Mechanism:** Make a fist to disengage the teleoperation (Clutch), allowing the operator to reposition their arm comfortably.
* **True Tendon Dynamics:** Bypasses pure kinematic joint overrides. Finger pinch commands are mapped directly to the Aero Hand's motor actuation (`data.ctrl`), driving the physical tendons.
* **Virtual Tennis Ball Interception:** Includes a dynamic Mid-air Catching testbed with a virtual tennis ball thrower for testing interception algorithms.

## 🛠️ Installation

**1. Create a Conda Environment:**
```bash
conda create -n safe_teleop python=3.10
conda activate safe_teleop

**2. Install Core Dependencies:**

```Bash
pip install mediapipe opencv-python dex-retargeting mujoco

**3. Install DM Control (with NumPy fix):**

```Bash
pip install dm_control "numpy<2.0.0"

**4. Clone the MuJoCo Menagerie Models:**
Ensure you have the latest DeepMind robot models cloned in your models/ directory:

```Bash
mkdir -p models
cd models
git clone [https://github.com/google-deepmind/mujoco_menagerie.git](https://github.com/google-deepmind/mujoco_menagerie.git)
cd ..

**🚀 Usage**
Run the main teleoperation script:

```Bash
cd src
python teleop_mujoco_basic_Withhand.py
