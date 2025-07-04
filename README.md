# Drone Navigation with PPO and Ursina

This project demonstrates reinforcement learning using Stable-Baselines3's PPO algorithm to train a drone agent to reach a target in a 3D space using the Ursina game engine for real-time visualization.

---

## 🐍 Python Version

This project requires **Python 3.10 or higher**.

---

## ⚙️ Setup Instructions

1. **Clone the repository**

    ```bash
    git clone https://github.com/V-i-g-n-e-s-h/Reinforcement_Learning.git
    cd Reinforcement_Learning
    ```

2. **Create a virtual environment**

    ```bash
    python3.10 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Run the Project

### 1. Train the agent

Run the following to train the PPO agent with live 3D visualization:

```bash
python train.py
```

This will open a Ursina window and begin training the drone to reach the goal. The model will be saved as `drone_ppo_vis.zip`.

### 2. Test the trained agent

Once training is complete, run the following to test the trained agent:

```bash
python test.py
```

You will see the trained drone navigating to the goal position with a trail visualizing its path.

---

### 📁 Files Overview
* `environment.py`: Custom Gymnasium environment defining drone physics and rewards.
* `train.py`: Trains the PPO agent with live Ursina rendering.
* `test.py`: Loads the trained agent and runs a visual test.
* `requirements.txt`: Python dependencies for the project.

---

Enjoy training your flying drone! ✈️