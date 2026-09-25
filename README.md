# 无线流量预测入门实验：ARIMA 与 LSTM

面向第一次接触深度学习和时间序列预测的本科生。先获取课程训练数据中的一个节点，完成数据检查、训练、预测和结果比较，再修改参数。

**课程数据仓库已确定：[competition-data](https://github.com/AIE1903-DataCompetition-26AY/competition-data)。** 代码与数据分开存放；此代码包不重新分发真实数据。

## 两个入口，共用同一份数据

`lab/train_arima.py` 是传统统计模型入口；`lab/train_lstm.py` 是 LSTM 结构与主要训练流程。两者共享读取、时间切分、变换、测试时间点与评价指标。

本仓库做的是**公开训练数据内部的课堂回测**：按时间分 60% / 20% / 20%，对一个节点滚动预测下一 5 分钟点。LSTM 输入最近 12 点；ARIMA 保留更长的历史滤波状态。二者不是严格匹配历史长度的消融实验。

**竞赛里的空白测试模板不参加本地切分；本地 `test_predictions.csv` 不是正式竞赛提交文件。** 本包没有实现官方多节点、多步提交管线；预测范围和提交规则须查阅各数据目录的 PDF。

## 从哪里开始

新手从 [学生逐步操作手册](docs/STUDENT_GUIDE.md) 开始。

代码仓库使用 `AIE1903-DataCompetition-26AY/wireless-traffic-lab`。以下命令从本代码包的根目录执行，先按学生手册创建环境、安装依赖。

## 从真实的数据仓库开始

```bash
git clone https://github.com/AIE1903-DataCompetition-26AY/competition-data.git ../competition-data
python lab/prepare_course_data.py --data-root ../competition-data --level cell_30 --inspect
python lab/prepare_course_data.py --data-root ../competition-data --level cell_30
python lab/check_data.py --csv data/course_traffic.csv --target traffic
```

已经 clone 过数据的同学不必再执行第一条。`--inspect` 列出本地文件和 CSV 字段；它不是从网页猜文件名。

转换脚本只支持 UTF-8 CSV 宽表或 `.csv.gz`：默认要求唯一一个文件名含 `train` 的合适 CSV；不满足时会停下，要求根据数据 PDF 指定 `--file`。时间列识别有歧义时要求 `--time-column`。不会将 test、baseline、sample submission 或映射表选为训练数据。

默认提取 CSV 顺序中的第一个可用数值列用于单节点入门，**不表示这个节点被教师指定，也不表示它最好预测**。原始时间列、节点列、文件 SHA-256 及能读取的本地 Git 版本写入 `data/course_traffic.metadata.json`。教师正式发布前应固定节点与文件；学生也可通过 `--target` 指定原始列名。

输出统一为 `timestamp,traffic`；这里的 `traffic` 是脚本生成的列名，**不是对课程原文件字段的猜测**。原数据仓库不被改写；不猜单位、不归一化、不插值、不填零。训练期标准化仍由两种模型各自按相同规则完成。

## 运行两个模型并比较

```bash
python lab/train_arima.py --csv data/course_traffic.csv --target traffic --order 1 1 1 --out runs/course_arima
python lab/train_lstm.py --csv data/course_traffic.csv --target traffic --epochs 3 --device cpu --out runs/course_lstm
python lab/compare_results.py --lstm-dir runs/course_lstm --arima-dir runs/course_arima --out runs/course_comparison
```

3 个 epoch 只检查 LSTM 流程，不代表充分训练；正式课堂实验可以去掉 `--epochs 3`，使用原版最多 100 轮和早停。每次用新输出目录，不覆盖旧结果。比较图与指标保存到 `runs/course_comparison/`。

## 网络或数据尚未就绪时：人工数据调试

```bash
python lab/make_demo_data.py
python lab/train_arima.py --csv data/demo_traffic.csv --target cell_A --out runs/demo_arima
python lab/train_lstm.py --csv data/demo_traffic.csv --target cell_A --epochs 3 --device cpu --out runs/demo_lstm
python lab/compare_results.py --lstm-dir runs/demo_lstm --arima-dir runs/demo_arima --out runs/demo_comparison
```

这是人工数据，不是脱敏实测数据。只学 ARIMA 可安装 `requirements-core.txt`；两个模型都学则安装 `requirements.txt`。环境步骤、CPU PyTorch 安装和错误排查见学生手册。

## 仓库结构

```text
course-projects/
├── competition-data/             # 现有数据仓库，单独 clone
│   ├── cell_30/
│   ├── cell_217/
│   └── grid_273/
└── wireless-traffic-lab/         # 本代码包；远端待教师创建
    ├── README.md
    ├── requirements.txt
    ├── requirements-core.txt
    ├── lab/
    │   ├── prepare_course_data.py # 本地检查、提取单节点、来源记录
    │   ├── check_data.py
    │   ├── common.py
    │   ├── train_arima.py
    │   ├── train_lstm.py
    │   ├── compare_results.py
    │   └── make_demo_data.py
    ├──  data/                    # 本地产物，不提交 Git
    └──  docs/
```

## 进阶操作与测试

```bash
python lab/train_arima.py --csv data/course_traffic.csv --target traffic --search --engine fast --out runs/course_arima_search
python -m unittest discover -s tests -v
python tests/smoke_test.py
python tests/smoke_course_data.py
```

搜索只用验证集选阶。`--engine fast` 是同一滚动一步预测的批量单向滤波，不是一次性多步开环预测。

验证记录区分旧版测试、本次新增接入测试与未完成的真实数据实验。所有附带测试只用人工数据，不含真实课程流量。
