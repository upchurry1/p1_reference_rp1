# UESTC-P1-Reference_RP1

独立 P1 对照任务，参考本机 `robolab` 的 **RP1-Flat**。旧 KneeLift 入口保留。

## 启动与待填参数

正式电机参数位于
[`p_reference_parameters.py`](source/unitree_rl_lab/unitree_rl_lab/assets/robots/p_reference_parameters.py)。
`P1_REFERENCE_MOTORS` 的六类关节分别覆盖左右两侧。每组必须填写：

| 字段 | 含义与单位 |
| --- | --- |
| `armature` | 附加关节惯量，kg·m²；允许显式 0 |
| `effort_limit` | 最大输出力矩，N·m |
| `velocity_limit` | 电机空载速度，rad/s |
| `keep_velocity` | 恒力矩段截止速度，rad/s；严格小于空载速度 |
| `rated_effort` | 连续额定力矩，N·m；不大于最大输出力矩 |
| `rated_velocity` | 额定速度，rad/s；不大于空载速度 |

六个关节组已展开为独立字典，直接将各字段的 `None` 替换成实际数值即可；左右两侧共用同组参数。PD 也逐组显式列出。
当前先保留参数填写框架，不做启动前的参数完整性检查；等实际参数填写完成后，再验证训练。

当前踝关节 `ankle_pitch`、`ankle_roll` 按用户选定的双 X4-36 常数近似填写，使用 24 V 参数表：`armature=0.07776 kg·m²`、峰值力矩 `68 N·m`、额定力矩 `21 N·m`、空载速度 `11.62389282 rad/s`、额定速度 `8.69173967 rad/s`。力矩和惯量按单电机的两倍取值，转速保持单电机值；`keep_velocity` 暂用额定转速近似。这些值未包含 P1 并联机构随姿态变化的雅可比映射。

填写真实电机参数后，在 `scripts` 目录启动：

```bash
python rsl_rl/train.py --task=UESTC-P1-Reference_RP1 --headless
```

回放使用同名任务及 `--checkpoint`，默认一个环境。图形回放默认启用键盘控制：按住 W/S 前进/后退、A/D 左右平移、Q/E 左右转向，松键归零；按住空格将所有速度指令归零。先点击 Isaac Sim 视口并切换英文输入法。
默认前后请求速度为 0.5 m/s、左右 0.05 m/s，转向为 0.3 rad/s；通过 `--keyboard-vx`、`--keyboard-vy`、`--keyboard-yaw` 调整。键盘 vx 的绝对值上限为 0.5 m/s；手动前进允许超过检查点的课程阶段速度范围，但仍受任务的全局硬限制，后退、横移和转向继续受检查点范围限制。键盘回放关闭随机速度/航向采样和随机推扰动；训练配置保持原 RP1-Flat 行为。传入 `--random-commands` 或使用 `--headless` 时保持原随机指令和推扰动分布。

图形回放默认显示 `Joint Torques` 面板，按仿真的实际关节顺序列出第一个机器人的所有关节。`Applied` 为电机限幅后施加到仿真的力矩，`PD raw` 为限幅前的力矩，单位均为 N·m，刷新频率为 10 Hz。关闭窗口后停止读取显示数据；也可通过 `--no-show-torques` 禁用。该面板也适用于 `--random-commands` 回放，`--headless` 时不会创建窗口。

```bash
# 在 scripts 目录执行；也可将 --checkpoint 替换成其他模型文件。
python rsl_rl/play.py --task UESTC-P1-Reference_RP1 --num_envs 1 --real-time \
  --checkpoint logs/rsl_rl/uestc_p1_reference_rp1/2026-10-27_16-18-04_rp1_flat_reference_from_scratch/model_2600.pt
```

日志位于启动目录下的 `logs/rsl_rl/uestc_p1_reference_rp1/`。

## 保留的观测与动作

Actor：每帧 45 维、15 帧，总计 675；Critic：每帧 48 维、15 帧，总计 720。直接复用现有观测配置，包括每个观测项、拼接顺序、缩放和噪声。

Actor/高斯采样 → `[-1,1]` 裁剪 → 各关节缩放 → 偏移限幅 → 加默认物理姿态。动作算术复用现有 `DeploymentLatencyJointPositionAction`，去掉 PACE 的编码器 bias 步骤。

| 关节 | 动作缩放 | 偏移限幅，rad | kp / kd |
| --- | ---: | --- | --- |
| hip_roll | 0.16 | [-0.22, 0.22] | 180 / 10 |
| hip_pitch | 0.32 | [-0.28, 0.35] | 180 / 10 |
| hip_yaw | 0.10 | [-0.16, 0.16] | 180 / 10 |
| knee_pitch | 0.55 | [-0.22, 0.55] | 180 / 10 |
| ankle_pitch | 0.18 | [-0.14, 0.20] | 40 / 2 |
| ankle_roll | 0.10 | [-0.12, 0.12] | 40 / 2 |

默认物理姿态：hip_pitch=-0.2、knee_pitch=0.2、ankle_pitch=-0.05 rad，其他关节为 0。无 bias；推理、通信和执行器延迟均为 0；关节静摩擦、动摩擦、黏性摩擦固定为 0.1、0.1、0.01。

策略 50 Hz，物理及 PD 200 Hz。网络、PPO、噪声初值 0.35 及其 [0.05,0.35] 边界沿用 KneeLift。电机采用 RP1 的四象限力矩—转速包络；饱和力矩由 `effort_limit / (1 - keep_velocity / velocity_limit)` 推导，物理力矩和速度上限同时设置为填写的最大输出力矩及空载速度。

## RP1-Flat 场景与指令

- 模型：项目同级 `urdf_803_nobattery/urdf/p1.urdf`，当前为 `/home/hr/gongxunp1/urdf_803_nobattery/urdf/p1.urdf`；支持 `P_ROBOT_URDF` 覆盖，并检查关节、刚体和网格。
- 地形：平面，无生成器、无地形课程。地面静/动摩擦 1.0/1.0，恢复系数 0，两种 combine mode 均为 multiply。
- 地面 MDL：`TilesMarbleSpiderWhiteBrickBondHoned`，project_uvw=True、texture_scale=(0.25,0.25)。为本机仅复制平面颜色的 TerrainImporter 补充显式 MDL 绑定。
- 默认 4096 环境，间距 2.5 m；机身和双足高度扫描关闭；保持 PhysX。
- 指令：UniformVelocityCommand，每 10 秒采样，20% 站立；vx∈[-0.6,1.0]、vy∈[-0.5,0.5] m/s。
- Heading：启用，比例 1.0、增益 0.5、目标∈[-π,π]；yaw 速度由航向误差产生，裁剪至 [-1.57,1.57] rad/s；站立环境指令为零。
- 推扰动：每 1–3 秒触发，速度增量 x/y∈[-0.5,0.5]、z∈[-0.2,0.2] m/s，roll/pitch∈[-0.52,0.52]、yaw∈[-0.78,0.78] rad/s；使用本机 `push_by_setting_velocity` 的叠加行为。
- 无原 P1 的指令晋级、回退、加减速限制、bootstrap 和奖励权重课程。其余随机化、终止条件保持原任务设置。

## 奖励与接触力

32 项奖励保留 RP1-Flat 的公式和权重。只适配本机 Torch/PhysX API、命令接口、关节分组和 P1 名称；删去不存在的腰/臂项，原 arms 项中的踝关节子集独立保留。足端高度偏移为 0.0408 m。

| 项目 | 权重 |
| --- | ---: |
| `track_lin_vel_xy_exp` | 1 |
| `track_ang_vel_z_exp` | 1 |
| `lin_vel_z_l2` | -0.2 |
| `ang_vel_xy_l2` | -0.1 |
| `energy` | -0.0001 |
| `joint_torques_l2` | -1e-05 |
| `joint_vel_l2` | -0.0002 |
| `dof_acc_l2` | -5e-07 |
| `action_rate_l2` | -0.2 |
| `action_smoothness_l2` | -0.02 |
| `joint_rated_velocity` | -0.05 |
| `joint_rated_effort` | -0.04 |
| `angular_momentum_rate_l2` | -5e-05 |
| `yaw_angular_momentum_l2` | -0.001 |
| `undesired_contacts` | -0.5 |
| `flat_orientation_l2` | -1 |
| `termination_penalty` | -200 |
| `feet_air_time` | 0.25 |
| `feet_slide` | -0.3 |
| `feet_force` | -0.003 |
| `feet_distance` | 0.1 |
| `knee_distance` | 0.1 |
| `feet_stumble` | -1 |
| `feet_orientation_l2` | -0.1 |
| `dof_pos_limits` | -10 |
| `joint_deviation_hip` | -0.03 |
| `joint_deviation_ankles` | -0.06 |
| `joint_deviation_legs` | -0.01 |
| `feet_contact_without_cmd` | 0.1 |
| `upward` | 0.4 |
| `stand_still` | -0.2 |
| `feet_height` | 0.2 |

足端总力由法向力加全部过滤接触的摩擦力构成。逐刚体传感器的第一个过滤对象固定为地面；足端还逐个过滤其他 P1 刚体。保留 3 个物理步的最新优先历史。自碰撞计数使用法向合力扣除地面法向力，保持参考奖励对法向力、总力的不同口径。

足端接触数据仅用于奖励，不加入观测。动作二阶差分和角动量变化率在复位后执行各自的预热逻辑，支持部分环境复位。RewardManager 按控制步长积分。

`params/deploy.yaml` 中的 `motor_reference` 保存实际填写的参数、参数文件哈希、PD、摩擦、关节顺序、无 bias 坐标约定及零延迟。新任务不会导出 `motor_identification` 或声称使用 PACE CSV。

## 检查与来源

```bash
# 参数填写完成后，再按需运行以下检查。脚本只在内存中注入合成测试数据。
python scripts/check_p_reference_rp1.py --headless --device cpu
python scripts/check_p_reference_rp1.py --headless --device cuda:0 --simulate --num_envs 4 --steps 16
# 仅测零延迟电机计算，不启动完整训练；结果不是整轮 collection 耗时。
python scripts/check_p_reference_rp1.py --headless --device cuda:0 --benchmark_actuator --num_envs 4096
```

采样路径对零延迟电机跳过延迟缓冲读写，保留 PD 和力矩包络；非摩擦接触传感器复用法向力及其历史，足端保留摩擦总力计算；同一个控制步的两个角动量奖励共享计算结果，下一步重新读取质量和惯量。物理步长、历史长度、奖励公式和权重均保持原配置。改动在下次启动训练时加载。

合成电机数据仅用于验证代码，不能作为正式训练的电机数据。USD 缓存和导出均写入临时目录；报告默认写入 `/tmp/p1_reference_check.json`。

检查覆盖观测与动作契约、无 bias PD、力矩包络、单支撑/站立/腾空、摩擦总力、自碰撞、额定工况、角动量、部分复位，以及小规模物理步进和 YAML 导出。合成数据检查不代替填写真实参数后的训练验证，也不验证行走收敛。

来源文件与 SHA-256 记录在 [`P_REFERENCE_RP1_SOURCES.json`](P_REFERENCE_RP1_SOURCES.json)。奖励和电机代码保留 RoboLab 的 BSD-3-Clause 归属及许可声明；运行时不需要安装 robolab。
