# 银行电话营销响应预测（scikit-learn 端到端项目）

从原始 CSV 到「打给谁」的完整链路：数据审计 → 特征工程 → 多模型对比 →
超参搜索 → 概率校准 → 阈值/容量决策 → 业务收益模拟 → 可解释性与报告。

**主结果**（按活动月分组留出，16,569 条测试样本）：

| 指标 | 数值 |
|---|---|
| 最终模型 | 逻辑回归 + 等渗校准（1-SE 规则选出） |
| ROC-AUC | **0.713**（95% CI 0.699–0.729） |
| Average Precision | **0.247**（基准 0.088） |
| top-20% 名单精确率 | **21.0%**（基准 8.8%，lift 2.4×） |
| 每千人利润 | **€2,393**，比全量拨打（€2,016）高 **€378** |

两个被量化出来的评估陷阱：

- `duration` 是通话结束后才知道的字段，加进模型 AP 从 0.292 涨到 0.497（AUC 0.747→0.921），上线拿不到；
- 随机切分让同一活动月同时出现在训练/测试，AP 从 0.247 虚增到 0.493，所以主评估改成按活动月分组。

概率在跨月时明显高估（平均预测 30.1% vs 实际 8.8%，Brier 0.120），因此业务决策
使用批次内 top-k 排序，而不是固定概率阈值；这一限制在报告第 8.3 节单独讨论。

完整分析见 [`reports/项目报告.md`](reports/项目报告.md)。

---

## 快速开始

```powershell
# 依赖：本机沙箱禁止 pip 写临时目录，脚本直接从 PyPI 下载 wheel 并解压
python scripts/bootstrap_env.py --only download
python scripts/bootstrap_env.py --only extract
$env:PYTHONPATH = "$PWD;$PWD\.pydeps"

# 全流程（训练 + 报告），CPU 单核约 5 分钟
python run_all.py

# 冒烟模式（小搜索空间，约 3 分钟）
python run_all.py --fast

# 单独跑测试 / 打分
python tests/test_smoke.py
python -m src.predict --input new_calls.csv --output scored.csv
```

数据会在首次运行时自动从 UCI 下载（`data/raw/`，约 5.8 MB）。

## 目录

```
bank-marketing-ml/
├─ run_all.py               # 一键复现
├─ src/
│  ├─ config.py             # 路径、业务假设（€80/€5、20% 容量）、随机种子
│  ├─ data.py               # 下载、审计、三种切分（随机/按月分组/时间外推）
│  ├─ features.py           # DomainFeatures + ColumnTransformer
│  ├─ models.py             # 模型库、搜索空间、GroupKFold 调参
│  ├─ train.py              # 主流程（CV → 调参 → 1-SE 选择 → 校准 → 评估 → 解释）
│  ├─ evaluate.py           # 指标、bootstrap、利润模拟、全部图表
│  ├─ report.py             # 从 artifacts 生成中文报告
│  └─ predict.py            # 批量打分（含字段校验）
├─ scripts/
│  ├─ bootstrap_env.py      # 无 pip 的依赖引导
│  └─ refresh_artifacts.py  # 不重训，刷新 EDA/PDP/利润图
├─ tests/test_smoke.py      # 泄漏、管线、指标的单测
├─ artifacts/               # 模型、metrics.json、各实验 CSV
└─ reports/                 # 项目报告 + figures/
```

## 关键设计

- **防泄漏**：`duration` 在 `DomainFeatures` 里强制删除，单测断言它永远不会进入特征矩阵；
  `pdays=999` 先转缺失再加 `contacted_before` 标记。
- **验证**：主评估按活动月整月留出；同时跑随机切分与时间外推做对照，
  三组结果都在 `artifacts/validation_strategy.csv`。
- **模型选择**：只看训练集 GroupKFold 的 AP，采用 1-SE 规则（简单模型不劣于最佳模型
  1 个标准误时优先选简单模型），留出集只用来做最终一次性汇报。
- **决策**：容量约束下按批次取 top-20% 分位；利润曲线与 €40–160 / €2–10 的敏感性
  网格说明结论对成本假设的依赖。
- **可复现**：固定种子、固定依赖版本（`requirements.txt`、`.pydeps/VERSIONS.txt`），
  报告中的每个数字都来自 `artifacts/`，不手写。

## 产物

| 文件 | 内容 |
|---|---|
| `artifacts/model.joblib` | 校准后的完整 pipeline（可直接 `predict_proba`） |
| `artifacts/metrics.json` | 所有指标、置信区间、选择与校准记录 |
| `artifacts/cv_model_zoo.csv` | 各模型逐折指标 |
| `artifacts/leakage_experiment.csv` | duration 泄漏对照 |
| `artifacts/validation_strategy.csv` | 三种切分对照 |
| `reports/figures/*.png` | 16 张图 |
| `reports/项目报告.md` | 完整中文报告（含面试讲解） |

## 环境说明

- Python 3.12，scikit-learn 1.9.1，pandas 3.0.1，matplotlib 3.11.2。
- 本机 DSH 沙箱会拦截 pip 的临时目录与 joblib 的进程/线程池（命名管道不可用），
  因此：依赖用 `bootstrap_env.py` 直接解压 wheel；训练全程 `n_jobs=1` 且
  `OMP_NUM_THREADS=1`（在 `src/__init__.py` 中设置）。换到普通环境时这两条限制可以去掉。
