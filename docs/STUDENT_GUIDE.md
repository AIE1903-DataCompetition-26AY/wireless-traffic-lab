# 学生操作手册：从零运行无线流量预测实验

## 0. 先了解你要完成什么

你会完成一条完整流程：**获取代码 → 准备 Python 环境 → 获取并检查数据 → 运行 ARIMA 和 LSTM → 比较预测结果 → 保存自己的实验记录**。

你不需要先掌握神经网络全部理论，也不一定使用 GPU。本教学示例可以选择 CPU 运行。首次只研究一个节点，不要一开始就把全部节点同时跑一遍。

代码仓库和数据仓库不是同一件事：课程数据已经发布在 [AIE1903-DataCompetition-26AY/competition-data](https://github.com/AIE1903-DataCompetition-26AY/competition-data)，需要单独 clone；本代码包不重复打包真实数据。

本代码仓库发布在 [AIE1903-DataCompetition-26AY/wireless-traffic-lab](https://github.com/AIE1903-DataCompetition-26AY/wireless-traffic-lab)。`cell_A` 只出现在人工数据例子；真实数据会通过接入脚本生成统一的 `traffic` 列。

## 1. 准备 GitHub 账号

使用自己的 GitHub 账号登录。

本实验在本地运行，不要求把真实无线数据上传到任何 AI 编程工具、公共网站或公共仓库。未经许可，不要把数据行或包含真实信息的截图直接贴进公开 Issue。

## 2. 安装 Git 和 Python

### 2.1 Git

从 [Git 官方安装入口](https://git-scm.com/install/) 下载适合自己系统的安装包。Windows 按安装向导完成；macOS / Linux 按官方平台说明安装。

安装完成后关闭旧终端，再打开一个新终端，运行：

```bash
git --version
```

看到 `git version ...` 表示终端找到了 Git。若提示不是命令，先确认安装完成并重新打开终端，不要先修改实验代码。

### 2.2 Python

安装建议选择 **64 位 Python 3.13**。Python 3.12 也可作为环境备选。

从 Python 官方网站取得安装包。Windows 安装时注意启用将 Python 加入 PATH 的选项；具体界面可能随安装器版本变化。安装完成后重新打开终端。

Windows 可检查：

```bat
py -3.13 --version
```

macOS / Linux 可检查：

```bash
python3.13 --version
```

### 2.3 什么是“终端”

终端是输入命令并看到程序输出的窗口，不是网页搜索框，也不是 Python 的 `>>>` 交互窗口。

Windows 可以使用“命令提示符”或 PowerShell；macOS 使用 Terminal；Linux 使用终端。VS Code / Cursor 的内置终端也可以，但它们不是运行此实验的必需软件。

本手册代码块中没有 `$` 或 `>>>` 提示符，直接复制命令本身，每次按回车执行。终端显示输出后，再输入下一条命令。

## 3. clone 代码，并进入正确目录

先在文件管理器创建一个放课程项目的目录，例如 `course-projects`。在该目录打开终端，或用 `cd` 进入这个目录。

在教师公布的代码仓库页面点击 **Code → HTTPS**，复制地址。

```bash
git clone https://github.com/AIE1903-DataCompetition-26AY/wireless-traffic-lab.git
cd wireless-traffic-lab
```

第一条命令复制代码；第二条命令进入复制出的项目。**后文除特别说明外，都在这个仓库根目录运行。**

Windows 查看当前目录内容：

```bat
dir
```

macOS / Linux：

```bash
pwd
ls
```

你应能看见 `README.md`、`requirements.txt`、`lab`、`docs`、`data`、`tests`。不要停在它们上一级，也不要把当前目录切到 `lab/` 再照抄本手册命令。

### Download ZIP 是备用方式，不等同于 clone

GitHub 的 **Code → Download ZIP** 可以下载文件快照。解压后同样可以运行 Python 代码，但下载的文件夹不是完整的 Git clone，不能直接照用 `git pull` 等同步命令。课程要求练习 Git 时，优先使用 clone。

## 4. 创建独立环境

虚拟环境把本项目的软件依赖与其他项目隔离。你通常只创建一次，但每次打开新终端后需要重新激活。

### Windows 命令提示符（推荐给 Windows 初学者）

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python --version
python -c "import sys; print(sys.executable)"
```

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -c "import sys; print(sys.executable)"
```

若 PowerShell 提示不允许运行激活脚本，最简单的替代是换用“命令提示符”，再执行上一节的激活命令。也可以不激活，直接使用 `.\.venv\Scripts\python.exe` 替代后文的 `python`；不需要为本实验放宽整个系统的脚本执行策略。

### macOS / Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python --version
python -c "import sys; print(sys.executable)"
```

若解释器叫 `python3` 而不是 `python3.13`，先确认版本再替换创建环境的第一条命令。

正确激活后，提示符通常出现 `(.venv)`，解释器路径也应位于本项目的 `.venv` 内。**以解释器路径为准，不要只看提示符。**

## 5. 安装环境依赖

先升级这个环境里的 pip：

```bash
python -m pip install --upgrade pip
```

### 5.1 两种模型都要运行

Windows / Linux 先安装 GPU 版 PyTorch，再安装其余依赖：

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

可选：安装 CPU 版 PyTorch 将 url 替换为：https://download.pytorch.org/whl/cpu。

macOS 使用常规安装入口：

```bash
python -m pip install -r requirements.txt
```

### 5.2 目前只学 ARIMA

```bash
python -m pip install -r requirements-core.txt
```

这样不安装 PyTorch，仍可以生成演示数据、检查数据、运行 ARIMA。需要运行 LSTM 时再安装完整依赖。不要把“ARIMA 不能 import torch”当成必要安装问题：ARIMA 本身不依赖 torch。

### 5.3 检查是否装进了正确环境

全部模型环境：

```bash
python -c "import numpy, pandas, scipy, statsmodels, torch, matplotlib; print('Imports OK'); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

`Imports OK` 表示这些库能导入；`CUDA: False` 对 CPU 实验完全正常。

只安装 ARIMA 环境：

```bash
python -c "import numpy, pandas, scipy, statsmodels, matplotlib; print('ARIMA imports OK')"
```

下载失败时，先分清网络连接问题与版本不兼容问题。不要关闭 SSL 校验、安装来历不明的二进制包，或在不理解原因时给安装命令加管理员权限。

## 6. 第一次先用人工数据走通流程

这一步是可选的环境检查；课程真实数据已经有独立仓库。为先隔离安装和模型运行问题，可生成一份明确标记的人工演示 CSV：

```bash
python lab/make_demo_data.py
```

这会创建 `data/demo_traffic.csv` 和来源说明 `data/demo_traffic.metadata.json`。默认有 2016 个时间点，两个虚构节点 `cell_A`、`cell_B`，数据从人工周期与噪声生成，单位也是任意演示单位。

**它不是脱敏实测数据，也不是课程真实数据的一部分。不能用它证明 ARIMA 或 LSTM 在真实无线流量上的优劣。**

同一文件已存在时生成器会拒绝覆盖。第二次继续使用已有文件即可，不需要每次都重新生成。

### 6.1 先列出列名

```bash
python lab/check_data.py --csv data/demo_traffic.csv
```

看到 `timestamp`、`cell_A`、`cell_B`。指定预测列再检查：

```bash
python lab/check_data.py --csv data/demo_traffic.csv --target cell_A
```

默认完整人工数据会得到 1209 / 403 / 404 个训练 / 验证 / 测试时间点，以及 1197 / 391 / 392 个可计分窗口。每个 split 的前 12 点只能充当历史，不能成为带有同 split 内 12 点历史的目标。

### 6.2 先运行 ARIMA

```bash
python lab/train_arima.py --csv data/demo_traffic.csv --target cell_A --order 1 1 1 --out runs/demo_arima
```

参数含义：`--csv` 是文件路径；`--target` 是要预测的流量列；`--order 1 1 1` 是 `(p,d,q)`；`--out` 是本次结果目录。

ARIMA 的“训练”是估计统计模型参数，不会像 LSTM 一样逐轮打印 epoch。终端会显示选用的阶数、验证误差、测试 MAE/RMSE 和保存位置。

首次使用预先指定的 `(1,1,1)`，并不表示它最适合真实无线数据。选择更合适的阶数是后面的实验任务。

### 6.3 运行一个只用于检查流程的 LSTM

```bash
python lab/train_lstm.py --csv data/demo_traffic.csv --target cell_A --epochs 3 --device cuda --out runs/demo_lstm
```

`--epochs 3` 只训练 3 轮，用来检查程序完整运行；不足以据此比较模型能力。`--device cpu` 可以不用 GPU。

终端的 `train_mse` 和 `val_mse` 是变换后的空间里的训练/验证误差，不是 MB 等原始流量单位上的误差，也不是准确率。

### 6.4 生成对比表与图片

```bash
python lab/compare_results.py --lstm-dir runs/demo_lstm --arima-dir runs/demo_arima --out runs/demo_comparison
```

在文件管理器打开 `runs/demo_comparison/`，里面有：

| 文件                        | 应该怎样读                                                   |
| --------------------------- | ------------------------------------------------------------ |
| `metrics_comparison.csv`    | 在所有共同测试计分点上的 Persistence、ARIMA、LSTM 的 MAE/RMSE |
| `predictions_aligned.csv`   | 时间戳、真实值和三种方法的预测值                             |
| `prediction_comparison.png` | 默认展示前 288 个可计分测试点的曲线                          |
| `lstm_learning_curve.png`   | LSTM 训练/验证误差随 epoch 的变化                            |

预测图只显示测试集的一段，指标用全部可计分测试点。图如果没有弹窗，可以通过文件下载方式取得 PNG 后查看。

看到 Persistence 比复杂模型好不意味着代码必定出错；先确认训练是否充分、任务是否容易被最近观测解释，以及模型选择是否合适。不要通过反复看测试集来挑到一个“更好看”的参数。

## 7. 使用已经发布的课程真实无线数据

### 7.1 单独 clone 数据，不要再复制进代码 Git 仓库

回到本代码仓库根目录，运行：

```bash
git clone https://github.com/AIE1903-DataCompetition-26AY/competition-data.git ../competition-data
```

`../competition-data` 表示代码仓库的上一级目录下另建一个数据文件夹。预期结构是：

```text
course-projects/
├── competition-data/
│   ├── cell_30/
│   ├── cell_217/
│   └── grid_273/
└── wireless-traffic-lab/
    ├── lab/
    ├── data/
    └── .venv/
```

已经有完整的数据 clone 时不要再次 clone 到同名文件夹。也可在数据仓库网页使用 **Code → Download ZIP**，解压后将 `competition-data-main` 改名为 `competition-data`，放到同样的位置；ZIP 没有 Git 历史，不能 `git pull`。

仓库 README 将 `cell_30` 定位为入门、`cell_217` 为完整小区数据、`grid_273` 为更细粒度网格数据。第一次课堂实验用 `cell_30` 中一个节点，不代表一次训练所有 30 个节点。

### 7.2 查看实际文件

```bash
python lab/prepare_course_data.py --data-root ../competition-data --level cell_30 --inspect
```

脚本会列出本地实际文件名、大小和 CSV 列名；**此命令不写数据、不训练模型**。先打开同目录的数据说明 PDF，确认哪些是训练集、哪些是空白测试模板、哪些是示例提交。

### 7.3 从训练文件提取一个节点

```bash
python lab/prepare_course_data.py --data-root ../competition-data --level cell_30
```

默认规则：只在该目录内找唯一一个文件名含 `train` 的 CSV，排除 test、baseline、sample、submission、template、membership。只支持 UTF-8 逗号分隔的宽表及 `.csv.gz`；一行一个时间点、一列一个节点。

时间列只在常见名称匹配唯一时自动识别。默认取文件顺序中第一个有观测值的非负数值列，打印原始列名；它只是入门选择，不是已确认的教师指定节点，也不是根据效果选出的节点。

如果有多个训练 CSV、文件名不含 `train` 或时间列无法识别，程序会停止而不是悄悄挑一个。根据上一节列出的实际名称，使用下面的参数模板，**不要照抄大写占位词**：

```bash
python lab/prepare_course_data.py --data-root ../competition-data --level cell_30 --file "ACTUAL_TRAIN_FILENAME.csv" --time-column "ACTUAL_TIME_COLUMN" --target "ACTUAL_NODE_COLUMN"
```

`--file` 是相对于 `cell_30/` 的文件路径。不存在、越出该目录、指向测试或示例提交都会报错。长表、Excel、原始 MR、模糊整数时间戳或其他压缩格式不自动猜测转换，应使用教师核对过的预处理。

运行成功后生成：

```text
data/course_traffic.csv
data/course_traffic.metadata.json
```

CSV 只有 `timestamp` 和 `traffic` 两列。`traffic` 是**提取后的统一名称**，并不是声称原文件中就叫这个名字；JSON 保存原文件路径、原始节点列、原始时间列、文件哈希及能取得的本地 Git 提交。没有 Git 信息时记录 null，不伪造版本。

原始流量值和单位保持不变。只按时间排序、重建 5 分钟网格；缺失保持 NaN，不填零、不插值。两个训练脚本再仅用本地训练段拟合标准化。

```bash
python lab/check_data.py --csv data/course_traffic.csv --target traffic
```

请核对接入脚本打印的原始节点，教师正式上课前应固定一个具体节点。换节点时重新指定原始 `--target` 并使用新的 `--out`；已有 CSV 或来源 JSON 不会被覆盖。

### 7.4 分清两种“测试集”

竞赛仓库提供的训练文件有历史流量；空白测试模板提供需预测的时间戳和输出列。它们不能拼起来再按 60/20/20 切分，更不能将空白填零当标签。

本实验**只读取官方公开训练文件中的一个节点**，在其内部按时间分成：

```text
公开训练历史 → 本地训练 60% | 本地验证 20% | 本地回测 20%
官方空白测试模板 → 本次不读入训练/回测，不计算标签误差
```

本地最后 20% 有已知历史真值，用于评估方法；它不是官方隐藏测试集。这里的滚动一步预测允许时间推进后使用已经观测到的历史。正式竞赛预测时，哪些未来真值可用、预测多少步、输出多少列、按什么指标计分，必须服从 PDF；不能把本地协议直接当作竞赛规则。

### 7.5 运行两种模型

第一次检查真实数据路径与流程时：

```bash
python lab/train_arima.py --csv data/course_traffic.csv --target traffic --order 1 1 1 --out runs/course_arima
python lab/train_lstm.py --csv data/course_traffic.csv --target traffic --epochs 3 --device cpu --out runs/course_lstm
python lab/compare_results.py --lstm-dir runs/course_lstm --arima-dir runs/course_arima --out runs/course_comparison
```

3 轮只检查流程。之后在新目录充分训练 LSTM：

```bash
python lab/train_lstm.py --csv data/course_traffic.csv --target traffic --device cpu --out runs/course_lstm_full
python lab/compare_results.py --lstm-dir runs/course_lstm_full --arima-dir runs/course_arima --out runs/course_comparison_full
```

不写 `--epochs` 使用原示例最多 100 轮和早停。ARIMA 固定训练段估计的参数，只随已到达的历史更新状态；`--engine fast` 是可选的等价批量实现。

有 NVIDIA GPU 且已由教师验证对应环境时再考虑 `--device cuda`。第一节不要求 GPU。

### 7.6 结果不是官方提交

打开 `runs/course_comparison/` 查看预测图、学习曲线和指标表。`test_predictions.csv` 只有本地单节点回测结果，`next_forecast.json` 只有下一点预测，**都不是可以直接上传比赛平台的完整预测提交**。

## 8. 理解每种模型保存了什么

两种模型共同输出 `run_config.json`（参数、环境版本、所选序列指纹）、`split_summary.json`（时间边界和计分样本数）、`test_predictions.csv` 和 `metrics.json`。

LSTM 额外输出 `history.csv` 与 `best_lstm.pt`。训练结束后恢复验证 MSE 最低的模型再计算测试指标；不是随便取最后一轮。仅加载自己生成的模型文件，不能把陌生来源的 checkpoint 当作普通数据随意载入。

ARIMA 额外输出 `validation_search.csv`、`arima_parameters.json` 和 `model_summary.txt`。前者记录候选阶数、验证误差及失败原因；即使只有固定一个阶数也会记录。参数 JSON 不含可独立恢复的完整历史状态，不能单独把它当成部署完成的服务模型。

末尾 12 个观测完整时，两种模型还会写 `next_forecast.json`，给出 CSV 最后时刻之后一个 5 分钟点的预测。它不是已拥有真实答案的测试点；没有未来实测值时，不能为它计算实际预测误差。

MAE 与 RMSE 都越小越好，单位跟 CSV 流量列相同。RMSE 对较大误差更敏感。默认不报 MAPE，避免大量零值和小值造成误导。教师明确规定阈值后，才用 `--mape-threshold`，且不能看完测试结果再挑阈值。

## 9. 单点滚动预测

一次预测使用的信息要区分时间：预测 10:05 时可以用截至 10:00 的历史；10:05 真值到达后，预测 10:10 时可以把 10:05 放进历史。

这叫**滚动一步预测**，本实验在公开训练历史内部模拟这一过程。它不等于在 10:00 就把之后整天全部预测好。本代码没有后一种多步开环评估；没有未来标签的竞赛测试阶段不能照搬逐点读真值的实现。

两种模型都只在训练集估计参数。LSTM 用验证集决定何时早停和选哪一轮；ARIMA 开启搜索时用验证集选阶。测试目标不参与这些选择。

但两种模型的信息范围不完全一样：LSTM 每次只看最近 12 点；ARIMA 状态可以承载更久之前的历史，而且状态连续跨过 train / validation / test。它们拥有相同的测试计分点，不代表拥有完全相同的输入历史限制。

## 10. 跑通后再修改一个参数

### 10.1 比较预先指定的 ARIMA 阶数

```bash
python lab/train_arima.py --csv data/course_traffic.csv --target traffic --order 1 0 0 --out runs/course_arima_100
```

先看验证表现，再固定你的选择。不要反复根据测试集挑阶数。

### 10.2 自动执行一个小规模验证集搜索

```bash
python lab/train_arima.py --csv data/course_traffic.csv --target traffic --search --engine fast --out runs/course_arima_search
```

默认搜索 `p ∈ {0,1,2}`、`d ∈ {0,1}`、`q ∈ {0,1}`，共 12 个组合。`--search` 开启后忽略 `--order`。有的组合拟合失败会记录原因并跳过；全部失败会明确报错，不会偷偷改成另一个模型。

`--engine fast` 与默认逐点教学流程使用相同的一步预测信息集合，只是批量执行滤波。它不是把真实测试答案提前作为当前预测输入。

### 10.3 修改 LSTM 隐藏维度

```bash
python lab/train_lstm.py --csv data/course_traffic.csv --target traffic --hidden-size 32 --device cpu --out runs/course_lstm_h32
```

一次只改一个设置，记录原因和验证集结果。正式比较前固定数据版本、节点、时间划分、预处理和训练设置。

### 10.4 查看帮助

```bash
python lab/train_arima.py --help
python lab/train_lstm.py --help
python lab/compare_results.py --help
```

不要只把最后一层输出维度改成 12，就声称已实现“预测未来 12 步”；那还需要同时修改标签、边界检查、指标和推理流程。

## 11. 常见问题排查

| 现象                         | 优先检查                                                     |
| ---------------------------- | ------------------------------------------------------------ |
| `python` / `git` 不是命令    | 是否安装并重新打开终端；Python 环境是否激活                  |
| `can't open file lab/...`    | 当前目录是否是仓库根目录                                     |
| `ModuleNotFoundError`        | `python -c "import sys; print(sys.executable)"` 是否指向 `.venv`；用同一个 `python -m pip` 安装 |
| `Repository not found` / 403 | URL、登录账号、是否接受课程邀请、教师是否授予权限            |
| CSV 不存在                   | 数据仓库是否下载到同级目录；是否已成功执行 prepare_course_data.py；不要把 ZIP 当 CSV |
| 找不到唯一训练 CSV           | 先 `--inspect`，核对 PDF 后指定 `--file`；不能把测试模板改名充当训练数据 |
| 缺少 `traffic`               | 确认使用接入脚本生成的 `data/course_traffic.csv`；`cell_A` 仅用于人工数据 |
| 5 分钟网格错误               | 时间格式、真实采样频率、聚合方式；不要直接把记录顺序当时间   |
| 缺失窗口太多 / 没有完整窗口  | 缺失分布、数据长度和采样间隔；不要把空值随手填成零           |
| ARIMA 没有收敛               | 先检查数据，再尝试较低阶数或更大的 `--maxiter`；保留失败日志 |
| ARIMA 训练段恒定             | 使用 Persistence 或按教师要求换节点，不要给数据人为加噪声蒙混过去 |
| 输出目录不为空               | 换新的 `--out`，不要急着删除旧结果                           |
| `CUDA: False`                | CPU 实验正常，继续用 `--device cpu`                          |
| 没有匹配的 PyTorch 安装包    | Python 版本、操作系统和 CPU 架构；改用教师验证的环境         |
| 图片没有自动弹出             | 到结果目录打开 PNG；脚本设计为支持无桌面服务器               |
| 两种模型无法对齐             | 检查 CSV 版本、目标列、时间列和是否用了相同代码版本          |

求助时附上执行命令、Python 版本、报错最后几行和不含敏感数据的必要信息。不要只发送“跑不了”，也不要把整个真实数据文件上传到公开讨论区。

## 12. 保存与提交实验记录

先在本地记录：使用的数据版本、节点、单位、运行命令、选阶/早停依据、两种模型与 Persistence 的结果，以及你对误差的解释。只提交教师允许公开或在课程内共享的内容。

在小组自己的仓库中保存代码和报告；不要直接向教师基线仓库推送每个小组的结果，也不要无意中把真实数据、`.venv`、checkpoint 或密钥加进 Git。项目自带 `.gitignore`，但你仍要检查 `git status`。

教师更新了基线且你没有未保存的本地修改时，可在 clone 的仓库根目录运行：

```bash
git status
git pull --ff-only
```

有修改或冲突时先咨询教师或按小组 Git 流程处理，不要用强制重置删除自己的工作。更新后若依赖文件有变化，在已激活环境中重新执行 `python -m pip install -r requirements.txt`。

再次打开终端时，进入项目、激活已有 `.venv` 即可，不要重复 clone 或重复创建环境。

完成实验后可退出虚拟环境：

```bash
deactivate
```
