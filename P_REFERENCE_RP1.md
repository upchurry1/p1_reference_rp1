# UESTC-P1-Reference_RP1

独立 P1 对照任务，参考本机 `robolab` 的 **RP1-Flat**。旧 KneeLift 入口保留。

## 低速起步微调（2026-10-28）

当前同名任务针对“0.1–0.4 m/s 时双脚压地、仅屈膝晃动”调整了训练目标；不再与原始 RP1-Flat 的指令分布和全部奖励权重完全相同。观测 675/720 维、动作 12 维、PD、电机参数、物理步长和动作处理保持不变，原模型权重可直接初始化新训练。

- 指令每 4–8 秒重采样：20% 站立，60% 低速纯前进，20% 保留原全范围指令。低速均分到 `[0.05,0.15]`、`[0.15,0.25]`、`[0.25,0.40]` m/s 三个区间，横向和 yaw 指令为零，并关闭这些样本的航向控制。
- 线速度跟踪权重从 1 调为 2；命令速度不超过 0.3 m/s 时，指数奖励的 `std` 从 0.5 收紧为 0.15；0.3–0.6 m/s 平滑恢复到 0.5。
- 新增 `low_speed_steps`：在 0.03–0.45 m/s 且小转向指令下，先给摆动脚卸载、实际足底抬升的增量奖励。每个阶段最多分别获得 0.06、0.12 的累计积分奖励；保持偏载、反复晃动和重复失败尝试不会持续得分。
- 完成奖励要求真实单脚支撑、足底净空至少 1 cm、腾空至少 60 ms、落地确认至少 40 ms、两次奖励间隔至少 0.25 秒，以及脚和身体沿指令方向前进；完成后只允许另一脚续接。双脚腾空、原地抬脚、只动膝盖和同脚重复均不能得到完整迈步奖励。奖励使用原 P1 四球足底的最低表面，而非踝关节原点高度。
- 推扰动减为每 5–8 秒一次，x/y 速度增量 ±0.1 m/s、yaw ±0.1 rad/s，关闭 z/roll/pitch 增量；其余随机化和原 32 项奖励保留，其中速度跟踪按上面的新目标替换，总计 33 项奖励。

从已有模型初始化新目标，使用新优化器、固定 `1e-4` 学习率和独立的新运行目录，不使用 `--resume` 恢复旧优化器：

```bash
conda activate my_robot_isaaclab
cd /home/hr/gongxunp1/PACE_P1-main/scripts
python rsl_rl/train.py --task UESTC-P1-Reference_RP1 --headless --num_envs 4096 \
  --init_checkpoint logs/rsl_rl/uestc_p1_reference_rp1/2026-10-28_09-47-19_rp1_flat_resume_2600/model_7900.pt \
  --init_noise_std 0.2 --max_iterations 3000 --run_name rp1_low_speed_start_finetune
```

修改配置不会让旧权重立即获得低速行走能力。训练前后使用同一评估脚本和参数对比，检查实际速度、左右离地落脚次数和跌倒数；此评估关闭观测噪声和推扰动，固定 `vy=yaw=0`：

```bash
# 在项目根目录执行；替换成需要评估的旧/新模型路径。
python scripts/eval_p_reference_speeds.py --headless --device cuda:0 \
  --checkpoint scripts/logs/rsl_rl/uestc_p1_reference_rp1/2026-10-28_09-47-19_rp1_flat_resume_2600/model_7900.pt \
  --speeds 0 0.1 0.2 0.3 0.4 0.5 --duration 12 --repeats 2 --report /tmp/p1_speed_eval.json
```

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

六个关节组已展开为独立字典；左右两侧共用同组参数。PD 也逐组显式列出。当前参数表已填写，本次低速微调保持这些电机参数和 PD 数值。

当前踝关节 `ankle_pitch`、`ankle_roll` 按用户选定的双 X4-36 常数近似填写，使用 24 V 参数表：`armature=0.07776 kg·m²`、峰值力矩 `68 N·m`、额定力矩 `21 N·m`、空载速度 `11.62389282 rad/s`、额定速度 `8.69173967 rad/s`。力矩和惯量按单电机的两倍取值，转速保持单电机值；`keep_velocity` 暂用额定转速近似。这些值未包含 P1 并联机构随姿态变化的雅可比映射。

填写真实电机参数后，在 `scripts` 目录启动：

```bash
python rsl_rl/train.py --task=UESTC-P1-Reference_RP1 --headless
```

回放使用同名任务及 `--checkpoint`，默认一个环境。图形回放默认启用键盘控制：按住 W/S 前进/后退、A/D 左右平移、Q/E 左右转向，松键归零；按住空格将所有速度指令归零。先点击 Isaac Sim 视口并切换英文输入法。
默认前后请求速度为 0.5 m/s、左右 0.05 m/s，转向为 0.3 rad/s；通过 `--keyboard-vx`、`--keyboard-vy`、`--keyboard-yaw` 调整。键盘 vx 的绝对值上限为 0.5 m/s；手动前进允许超过检查点的课程阶段速度范围，但仍受任务的全局硬限制，后退、横移和转向继续受检查点范围限制。键盘回放关闭随机速度/航向采样和随机推扰动。传入 `--random-commands` 或使用 `--headless` 时使用当前训练指令混合分布和推扰动；本节顶部的固定速度评估脚本则显式关闭推扰动。

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
| hip_yaw | 0.10 | [-0.16, 0.16] | 150 / 4 |
| knee_pitch | 0.55 | [-0.22, 0.55] | 200 / 5 |
| ankle_pitch | 0.18 | [-0.14, 0.20] | 50 / 3 |
| ankle_roll | 0.10 | [-0.12, 0.12] | 50 / 3 |

默认物理姿态：hip_pitch=-0.2、knee_pitch=0.2、ankle_pitch=-0.05 rad，其他关节为 0。无 bias；推理、通信和执行器延迟均为 0；关节静摩擦、动摩擦、黏性摩擦固定为 0.1、0.1、0.01。

策略 50 Hz，物理及 PD 200 Hz。网络、PPO、噪声初值 0.35 及其 [0.05,0.35] 边界沿用 KneeLift。电机采用 RP1 的四象限力矩—转速包络；饱和力矩由 `effort_limit / (1 - keep_velocity / velocity_limit)` 推导，物理力矩和速度上限同时设置为填写的最大输出力矩及空载速度。

## RP1-Flat 场景与指令

- 模型：项目同级 `urdf_803_nobattery/urdf/p1.urdf`，当前为 `/home/hr/gongxunp1/urdf_803_nobattery/urdf/p1.urdf`；支持 `P_ROBOT_URDF` 覆盖，并检查关节、刚体和网格。
- 地形：平面，无生成器、无地形课程。地面静/动摩擦 1.0/1.0，恢复系数 0，两种 combine mode 均为 multiply。
- 地面 MDL：`TilesMarbleSpiderWhiteBrickBondHoned`，project_uvw=True、texture_scale=(0.25,0.25)。为本机仅复制平面颜色的 TerrainImporter 补充显式 MDL 绑定。
- 默认 4096 环境，间距 2.5 m；机身和双足高度扫描关闭；保持 PhysX。
- 指令：`LowSpeedVelocityCommand`，采样比例和区间见低速微调说明；保留全范围样本的 vx∈[-0.6,1.0]、vy∈[-0.5,0.5] m/s。
- Heading：全范围样本启用，增益 0.5、目标∈[-π,π]，yaw 裁剪至 [-1.57,1.57] rad/s；低速纯前进样本关闭航向控制，站立环境指令为零。
- 推扰动：使用本机 `push_by_setting_velocity` 的叠加行为，间隔和增量按本节顶部的低速微调设置。
- 未接入旧 PACE 任务的指令晋级、回退和关节屈曲 bootstrap 课程；本次使用固定混合指令及物理卸载/落脚引导，其余随机化和终止条件保持原任务设置。

## 奖励与接触力

当前 33 项奖励以原 RP1-Flat 的 32 项为基础：线速度跟踪改为低速收紧的目标，增加有限预算的 `low_speed_steps`，其余项保持原公式和权重。接口适配本机 Torch/PhysX、关节分组和 P1 名称；删去不存在的腰/臂项，原 arms 项中的踝关节子集独立保留。原足端高度偏移为 0.0408 m，新起步项使用实际四球足底净空。

| 项目 | 权重 |
| --- | ---: |
| `track_lin_vel_xy_exp` | 2 |
| `low_speed_steps` | 1 |
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
