# 公共协议 v0.1（2026-11-06 冻结）

`HomeCareEnv(config: dict, trajectory: dict | None = None)`；`reset(seed: int) -> dict`；
`step(action: dict) -> (state: dict, reward: float, done: bool, info: dict)`。
`BasePlanner.decide(state: dict) -> dict`，构造参数 `config: dict | None`；`last_info` 保存规划诊断。
状态为 JSON 可序列化深拷贝，规划器只接收 state，不接收环境或轨迹对象。

## 状态

|字段|类型 / 单位 / 值|
|---|---|
|schema_version|str = "0.1"|
|time_s, horizon_s|整数秒，从当天 08:00 起，默认终点 43200|
|position|{node: str/null, edge: [str,str]/null, offset_m: float, remaining_distance_m: float}；offset 是从 edge[0] 的距离；移动中 node=null，可原地转向，物理位置连续|
|battery|{energy_wh: float, capacity_wh: float}|
|mode|idle / moving / serving / waiting / charging / failed / finished|
|active_task_id|str/null|
|pending_tasks|已释放且未完成普通及紧急任务，见下表|
|current_events|已释放紧急任务的列表（包括完成/无法响应），不含未来|
|valid_actions|可行 serve、charge、wait 默认时长、continue 的示例枚举；任意合法正整数 wait 仍须再次安全校验|
|public_model|公开 model 配置（见 base.json），增加 distances_m、travel_times_s（嵌套 node→node）、distance_from_position_m、travel_time_from_position_s（node→值）；travel time 按每条边 ceil(length/speed) 求和|

任务字段：id/type/location 为 str；type=medication/patrol/fetch/companionship/cleaning/emergency；
release_time_s/earliest_start_s/deadline_s/service_duration_s/remaining_service_s 为整数秒；priority 为正整数（越大越重要）；preemptible 为 bool。
运行字段 status=pending/active/completed/unresponded；completed_at_s/response_at_s 为 int/null；failure_reason 为 str/null。
超过完成截止时间不会删除。emergency 的 priority 表示严重级别。
轨迹格式 `{seed: int, tasks: [完整原始任务]}`；只给仿真器，结果保存需在运行端读取。

## 动作及错误

- `{"type":"serve","task_id":str}`：从当前位置导航，必要时等到 earliest_start_s，再执行剩余服务。
- `{"type":"charge"}`：前往充电桩并充到目标比例。
- `{"type":"wait","duration_s":int}`：1..max_wait_s，bool 不算 int。
- `{"type":"continue"}`：继续现有行动；空闲时变成 default_wait_s 等待。

拒绝额外字段、未知任务、不满足能量条件、非法时长等；非法建议固定回退可行 EDF，若无任务则可达回充或等待。非法原因与安全覆盖分开记录。
非抢占服务拒绝切换；移动中允许改道。默认任务全部可抢占。紧急任务排队按 (-priority, release_time_s, id)，当前紧急服务不被另一个事件中断；非抢占普通服务会延迟应急，并记原因。

## 安全、时间与统计

接任务所需能量包括路程、提前等待、剩余服务、回桩及 reserve_wh。每条边的整数秒时间以匀速走完该边（有效速度不超过配置速度），途中位置精确保留；改道可从边两端选择最短距离路线（等距按端点名称排序）。能耗按实际米数。
安全层在每次动作前及每个事件结算后执行：无法回桩→失效；可行紧急任务→强制响应；不可行紧急任务→标记 unresponded 且记录 energy_infeasible_emergency 后回充；低电量→回充；为防整数秒越过返桩储备边界，再提前保留下一秒待机/服务消耗或移动时的两倍瞬时消耗（消耗加返桩距离增长上界）。
充电直到目标，普通任务可由策略中断；低电量时保持充电。充满时 charge 转为正时长等待，不制造零时间环。
同一时刻：结算运动/服务/充电 → 完成登记 → 按 id 释放全部任务 → 应急及安全层 → 返回状态。
下一事件包括每条边到达、最早开始、服务完成、等待结束、充电目标、低电量/返桩储备边界、到达和终点。所有非终止 step 至少推进一秒。
不足以支持下一个完整秒的耗能活动时记录失效（保留不足一秒的残余电量），不虚构无电量工作、不生成负电量；失败机器人吸收到 horizon；未来到达仍注册并计入分母，终点恰好完成计完成，终点到达不生成。
reward 恒为 0.0。wall-clock 不注入仿真时间。
info：elapsed_s, events（本步日志）, requested_action, executed_action, invalid_action（原因/null）, override_reason（原因/null）, termination_reason（horizon/robot_failed/null）。事件后安全动作另有 event 日志，不冒充本步最初执行动作。

## 研究指标

普通任务与紧急任务分开：关键普通任务 priority>=critical_priority 且 deadline<=horizon 的按时完成率；全部已释放普通任务未完成率；完成任务 max(0,completion-deadline)，未完成 max(0,horizon-deadline) 的总截尾延误及 priority 加权延误。分母为零记 null/CSV 空白。
紧急响应时间=response_at-release，只统计已到达现场者，并同时报所有事件未响应比例；没有事件时所有响应统计均 null。已响应但未完成另报。故障不提前截断统计终点。
计充电段启动次数、时间、最低电量、电量违规（模型负值/越界）、失效、非法率、干预率及决策耗时。原始事件可追溯计数。所有结果仅 smoke 流程验证。
