# 课程数据发布信息（已填入可确认部分）

核对日期：2026-09-24。已确认项来自[仓库首页及 README](https://github.com/AIE1903-DataCompetition-26AY/competition-data)；未确认项不能用代码示例代替。

| 字段 | 已确认内容 / 发布前待确认 |
| --- | --- |
| GitHub Organization | `AIE1903-DataCompetition-26AY` |
| 数据仓库 | `https://github.com/AIE1903-DataCompetition-26AY/competition-data` |
| 数据 clone 地址 | `https://github.com/AIE1903-DataCompetition-26AY/competition-data.git` |
| 仓库可见性 | 核对时为 Public；正式发布时复核 |
| 数据目录 | `cell_30`、`cell_217`、`grid_273` |
| 采样频率 | README 说明为 5 分钟 |
| 流量方向 | README 说明为 uplink traffic（上行流量） |
| 目录内容 | README 说明有训练集、空白测试模板、基线/示例提交、PDF；网格另有映射表 |
| 代码仓库 | 建议 `wireless-traffic-lab`；尚未创建/推送 |
| 学生原始数据目录 | 代码仓库旁的 `../competition-data/` |
| 学生课堂输入 | 接入脚本生成的 `data/course_traffic.csv` |
| 课堂输出列名 | `timestamp`、`traffic`；原始节点名见 metadata JSON |
| 真实数据版本 / 提交 | 待教师固定；脚本记录能取得的本地 Git 版本 |
| 实际训练文件相对路径 | 待本地 `--inspect` 与 PDF 核对 |
| 原始时间列 / 节点列 | 待确认；不要用 `cell_A` 代替真实节点 |
| 首次课堂指定节点 | 待教师确认并在 `--target` 中固定 |
| 时间范围 / 时区 | 待读取文件与 PDF 确认 |
| 流量单位 / 聚合定义 | 待 PDF 确认；代码不转换或猜测单位 |
| 缺失 / 零值含义 | 待 PDF 确认；代码不将空值填零 |
| 正式竞赛预测跨度 / 协议 | 待 PDF 确认；本地滚动一步回测不自动等于比赛 |
| 正式计分指标 / 提交格式 | 待 PDF 确认；本包没有生成正式提交 |
| 原始文件 SHA-256 | 本地脚本记录；教师发布时确认值 |
| 再分发数据 / 图片 / 结果权限 | 由教师确认；Public 可见性不替代授权说明 |
| 课堂电脑 / 服务器环境 | 待学生目标平台实测 |

数据已发布不等于模型接口已在真实文件上验收。请先完成学生手册第 7 节，再固定版本发给全班。
