# ByteTrack 概率场速度优化记录

更新时间: 2026-03-15

## 1. 背景

当前仓库中的 ByteTrack 在固定摄像头场景下加入了 `spatial prior` 概率场，用于替代原本较粗糙的矩形区域划分。

问题在于:

- 早期矩形分区版本的关联速度可达约 `220 FPS`
- 加入概率场后，完整 SOMPT22 评估中的关联速度下降到 `18.9 FPS`

本轮工作的目标是:

- 优化概率场相关实现
- 不改变关联策略与阈值
- 不损害现有测试指标

## 2. 结论摘要

本轮优化后:

- 概率场路径的主要热区已显著压缩
- 复用旧 dets/embs 的全序列 SOMPT22 评估中，关联速度从 `18.9 FPS` 提升到 `23.9 FPS`
- `HOTA/MOTA/IDF1/IDSW` 与既有基线保持一致

说明:

- 概率场此前确实是显著瓶颈之一
- 但优化完成后，它已经不再是当前 tracker 的主导瓶颈
- 当前剩余的大头更可能在 ByteTrack 主流程、zombie rescue、IoU 距离矩阵构建，以及 Python 层列表/对象操作

## 3. 本轮优化原则

本轮只做“等价优化”，不改算法决策:

- 不改 `zombie rescue` 匹配逻辑
- 不改任何阈值
- 不改 `entry/core` 语义
- 不改概率场的高斯核定义与归一化方式

换句话说，优化目标是“同样的输入，得到同样的判定”，只是把实现变快。

## 4. 具体优化点

涉及文件:

- `boxmot/trackers/bytetrack/spatial_prior.py`
- `boxmot/trackers/bytetrack/bytetrack.py`
- `tests/unit/test_trackers.py`

### 4.1 Lazy decay 替代逐帧全图衰减

原实现中，`SpatialPriorField.step()` 每帧都会对整张 `support_count` / `birth_count` 做一次乘法。

优化后:

- 只维护一个全局 `decay_scale`
- 在读取概率图或 scale 太小时再物化到数组

效果:

- 避免每帧两次整图写回
- 对长序列稳定场景尤其有效

### 4.2 Gaussian splat 向量化

原实现中，每个点的 splat 都会:

- 单独算一次局部窗口
- 单独生成 `meshgrid`
- 单独执行 `exp`

优化后:

- 改为批量 `add_support_batch()` / `add_birth_batch()`
- 使用预先缓存的 offset
- 用可分离的一维高斯构造二维核
- 用向量化方式一次处理一批点

效果:

- 显著降低 Python 循环和重复分配成本

### 4.3 `_local_sum()` 去掉 Python 双重循环

原实现用 `for dy` / `for dx` 手工累加 3x3 邻域。

优化后:

- 直接用 9 个切片求和

效果:

- 降低解释器循环开销

### 4.4 support / birth 提交批处理

原实现中，`_update_spatial_prior_tracks()` 会对每条 track 分别调用一次 splat。

优化后:

- 先收集 `support_points` / `birth_points`
- 再一次性批量提交到概率场

效果:

- 活跃轨迹数较多时，能明显减少函数调用与 NumPy 启动成本

### 4.5 脚点直接从状态量读取

原实现中，support 点的计算会多次走框格式转换。

优化后:

- 在 `STrack` 上添加 `footpoint()`
- 直接从当前状态的 `center_x / center_y / height` 计算底部中心点

效果:

- 降低频繁的 bbox 格式转换开销

### 4.6 region mask 改为 dirty-bit 按需刷新

原实现中，只要概率场处于可用状态，`update()` 开头就会重建一次 `entry/core mask`。

优化后:

- 只在数据发生变化后标记 `_spatial_region_dirty`
- 真正调用 `_get_spatial_region_label()` 时才刷新

效果:

- 避免无谓的 mask 重建
- 在很多帧里直接省掉一次 region rebuild

## 5. 微基准结果

本地微基准用于隔离概率场热路径，结果如下。

### 5.1 概率场更新热区

场景:

- `48x27` 概率场
- 每帧 `step + 50 个 support splat + 5 个 birth splat`

结果:

- 优化前: `10.869 ms/frame`
- 优化后: `1.715 ms/frame`

约提升:

- `6.3x`

### 5.2 概率场对 tracker 的边际开销

合成压力测试:

- 50 个稳定目标
- 禁用 detector / ReID 生成，只测 `tracker.update`

结果:

- `spatial_off`: `32.866 ms/frame`
- `spatial_on_region`: `34.621 ms/frame`

说明:

- 当前概率场额外开销约为 `+1.755 ms/frame`

而在优化前，对应场景中:

- `spatial_off`: `32.204 ms/frame`
- `spatial_on_region`: `48.185 ms/frame`

说明:

- 优化前概率场额外开销约为 `+15.981 ms/frame`

结论:

- 概率场本身的额外负担已经从约 `16ms/frame` 压到约 `1.8ms/frame`

## 6. 全序列 SOMPT22 复用缓存评估

由于用户要求“不重新跑检测与 ReID，只复用旧 dets/embs”，本次采用缓存复用方式评估。

结果目录:

- `/root/autodl-tmp/boxmot/boxmot_demo2/runs_reusecache_20260315_112933`

结果文件:

- `/root/autodl-tmp/boxmot/boxmot_demo2/runs_reusecache_20260315_112933/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack/person_summary.txt`

### 6.1 指标对比

与既有基线相比，指标保持一致:

- `HOTA = 53.023`
- `MOTA = 65.710`
- `IDF1 = 66.179`
- `IDSW = 815`

说明:

- 本轮优化没有引入可见的关联质量回退

### 6.2 关联速度对比

历史基线来自此前完整 fresh eval:

- `Association = 52.81 ms/frame`
- `Association FPS = 18.9`

本轮复用缓存评估结果:

- `Association = 41.86 ms/frame`
- `Association FPS = 23.9`

提升幅度:

- 每帧减少 `10.95 ms`
- 相对耗时下降约 `20.7%`
- 关联 FPS 相对提升约 `26.5%`

## 7. 为什么优化后还远达不到 100 FPS

这轮结果说明:

- 概率场此前确实很慢
- 但现在它已经不再是主导瓶颈

当前 `Association = 41.86 ms/frame`，而概率场边际开销大约只有 `1.8 ms/frame` 左右，这意味着大部分时间已经不在概率场上。

下一轮更值得重点排查的部分:

- Step1 高分检测关联
- Step2 低分检测关联
- zombie rescue 的 cost matrix 构建
- Hungarian
- `joint_stracks / sub_stracks / remove_duplicate_stracks`
- `STrack` 对象创建和 Python 层循环

## 8. 本轮验证

执行过的关键回归测试:

```bash
uv run pytest tests/unit/test_trackers.py -k "zombie_reid_global_assignment_prefers_appearance or zombie_reid_gate_blocks_wrong_appearance_rescue or spatial_prior"
```

结果:

- `6 passed`

其中新增了一条 lazy decay 正确性测试，确保惰性衰减不会改变有效质量。

## 9. 后续建议

当前概率场实现已经从“主要瓶颈”降为“次要开销”，下一轮建议直接做细粒度 profiler，把 `tracker.update()` 进一步拆成:

1. Step1 高分关联
2. Step2 低分关联
3. zombie rescue cost build
4. Hungarian
5. spatial prior update
6. 列表去重与轨迹集合维护

只有把剩余 `41.86 ms/frame` 再拆开，才有机会继续往 `>100 FPS` 靠近。
