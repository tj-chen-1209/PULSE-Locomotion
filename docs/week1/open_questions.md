# Open Questions

记录当前还没有搞清楚的问题。问题解决后在条目前打 `[x]` 并注明结论。

用法：直接告诉 AI「把 XXX 记录到 open_questions」，AI 会在下方追加一条。

格式：

```
- [ ] <问题描述> — *added YYYY-MM-DD*
  - 背景：...（可选）
  - 猜测/方向：...（可选）

- [x] <已解决的问题> — *resolved YYYY-MM-DD*
  - 结论：...
```

---

- [x] 原地转圈 eval 效果极差，是否是训练/eval OOD？ — *resolved 2026-03-31*
  - 结论：两个叠加的 OOD 来源。
    1. 训练用 `heading_command=True`，yaw 命令随时间衰减到 0；eval 用固定恒定 yaw rate，形状从未在训练中出现。
    2. `rel_standing_envs=0.02` 过低，原地旋转几乎没有训练覆盖。
  - 修复：在 `Go2PulseFlatEnvCfg.__post_init__` 中设置 `heading_command=False`、`rel_standing_envs=0.20`、`rel_heading_envs=0.0`，然后从头训练。

