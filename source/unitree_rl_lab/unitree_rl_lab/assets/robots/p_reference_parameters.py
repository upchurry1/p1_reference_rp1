"""UESTC-P1-Reference_RP1 电机参数：直接将 None 替换为实际数值，左右共用。"""

# armature: 附加关节惯量，kg·m²
# effort_limit: 最大输出力矩，N·m
# velocity_limit: 空载速度，rad/s
# keep_velocity: 恒力矩段截止速度，rad/s
# rated_effort: 额定力矩，N·m
# rated_velocity: 额定速度，rad/s

P1_REFERENCE_MOTORS = {
    # 髋关节 roll：X8-120，按提供的 48 V 产品参数表填写。
    # RPM 按模组输出端转速换算为 rad/s，不再除以减速比 19.61。
    "hip_roll": {
        "armature": 0.057682815,         # 转子侧惯量折算：1.5 × 1e-4 × 19.61²，kg·m²
        "effort_limit": 120.0,           # 峰值力矩，N·m
        "velocity_limit": 16.54572131,   # 空载 158 RPM × 2π / 60，rad/s
        "keep_velocity": 13.29940890,   # 按用户要求暂用额定 127 RPM 近似，非实测峰值恒力矩拐点
        "rated_effort": 43.0,           # 额定力矩，N·m
        "rated_velocity": 13.29940890,  # 额定 127 RPM × 2π / 60，rad/s
    },
    # 髋关节 pitch：同一款 X8-120。
    "hip_pitch": {
        "armature": 0.057682815,         # 转子侧惯量折算：1.5 × 1e-4 × 19.61²，kg·m²
        "effort_limit": 120.0,
        "velocity_limit": 16.54572131,
        "keep_velocity": 13.29940890,   # 暂用额定 127 RPM 近似，非实测峰值恒力矩拐点
        "rated_effort": 43.0,
        "rated_velocity": 13.29940890,
    },
    # 髋关节 yaw：X6-60，按 48 V 产品参数表填写。
    "hip_yaw": {
        "armature": 0.025385615904,      # 沿用转子侧折算：0.66 × 1e-4 × 19.612²，kg·m²
        "effort_limit": 60.0,           # 峰值力矩，N·m
        "velocity_limit": 18.43067690,  # 空载 176 RPM × 2π / 60，rad/s
        "keep_velocity": 16.02212253,  # 沿用额定 153 RPM 近似，非实测峰值恒力矩拐点
        "rated_effort": 20.0,          # 额定力矩，N·m
        "rated_velocity": 16.02212253, # 额定 153 RPM × 2π / 60，rad/s
    },
    # 膝关节 pitch：同一款 X6-60。
    "knee_pitch": {
        "armature": 0.025385615904,      # 转子侧惯量折算：0.66 × 1e-4 × 19.612²，kg·m²
        "effort_limit": 60.0,
        "velocity_limit": 18.43067690,
        "keep_velocity": 16.02212253,  # 暂用额定 153 RPM 近似，非实测峰值恒力矩拐点
        "rated_effort": 20.0,
        "rated_velocity": 16.02212253,
    },
    # 踝关节 pitch：两台 X4-36，按用户确认的双电机常数近似填写。
    # 数据表工况为 24 V；力矩、惯量取两倍，转速保持单电机值。
    # 未采用并联机构随姿态变化的雅可比映射。
    "ankle_pitch": {
        "armature": 0.07776,            # 2 × (0.3 × 1e-4 × 36²)，kg·m²
        "effort_limit": 68.0,           # 2 × 单电机峰值 34 N·m
        "velocity_limit": 11.62389282,  # 空载 111 RPM × 2π / 60，rad/s
        "keep_velocity": 8.69173967,   # 暂用额定 83 RPM 近似
        "rated_effort": 21.0,          # 2 × 单电机额定 10.5 N·m
        "rated_velocity": 8.69173967,  # 额定 83 RPM × 2π / 60，rad/s
    },
    # 踝关节 roll：同样采用两台 X4-36 的常数近似。
    "ankle_roll": {
        "armature": 0.07776,
        "effort_limit": 68.0,
        "velocity_limit": 11.62389282,
        "keep_velocity": 8.69173967,   # 暂用额定转速近似
        "rated_effort": 21.0,
        "rated_velocity": 8.69173967,
    },
}

# 每组为 (kp, kd)，左右共用。
P1_REFERENCE_PD = {
    "hip_roll": (180.0, 10.0),
    "hip_pitch": (180.0, 10.0),
    "hip_yaw": (150.0, 4.0),
    "knee_pitch": (200.0, 5.0),
    "ankle_pitch": (50.0, 3.0),
    "ankle_roll": (50.0, 3.0),
}

