# 课程数据的位置与课堂输入格式

课程真实数据来自 [AIE1903-DataCompetition-26AY/competition-data](https://github.com/AIE1903-DataCompetition-26AY/competition-data)，独立于代码仓库。以该仓库 README 和各目录 PDF 为准；本次仅核对了首页/README，没有下载并验证具体数据文件。

从代码根目录执行：

```bash
git clone https://github.com/AIE1903-DataCompetition-26AY/competition-data.git ../competition-data
python lab/prepare_course_data.py --data-root ../competition-data --level cell_30 --inspect
python lab/prepare_course_data.py --data-root ../competition-data --level cell_30
python lab/check_data.py --csv data/course_traffic.csv --target traffic
```

数据已存在时不重复 clone。脚本支持 UTF-8 CSV 宽表及 `.csv.gz`，只处理训练文件的一列；文件或时间列有歧义时要求显式指定，不会选择测试模板。详情见 [课程接入说明](../docs/COURSE_DATA.md) 和 [学生手册第 7 节](../docs/STUDENT_GUIDE.md)。

本地生成 `course_traffic.csv`（`timestamp,traffic`）和 `course_traffic.metadata.json`。原始列名、文件哈希和可读取的 Git 版本记录在 JSON；`traffic` 是统一输出名，不是对原始字段的断言。

只按时间排序和重建 5 分钟时间轴。空值保留 NaN、不填零、不插值、不改变单位。两个模型会仅用本地训练段计算 `log1p` 后的标准化统计量。

本实验的本地训练/验证/回测三段全部来自公开训练文件；**不要拼入官方空白测试模板或示例提交文件**。

原始文件不必复制进此文件夹。`.gitignore` 忽略此目录中的数据与来源 JSON，只保留 Markdown 说明；它不是权限控制，发布前仍要看 `git status`。

人工调试数据用 `python lab/make_demo_data.py` 生成，包含虚构 `cell_A` / `cell_B`。它不来自课程仓库，也不能证明真实数据上的效果。

[课程发布信息](DATA_INFO_TEMPLATE.md) 已填入确定的数据仓库和目录；具体版本、字段、单位、授权与教师指定节点仍应课前确认。
