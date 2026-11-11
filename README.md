# mahjong_AI： Transformer 麻将 AI

只为对局强度训练的日麻 AI，目标规则为**雀魂四人半庄、玉之间计分**。

- 监督学习：在约 50 万局高段位牌谱上训练。
- 强化学习：在监督模型的基础上做 PPO 自我对弈强化。

最终模型在 4 万局独立牌山上，对监督模型每半庄多拿 **+2.37 ±0.52 段位分**（z = 4.6）。

> 模型权重（151 MB）超过 GitHub 单文件 100 MB 的限制，放在 **Releases** 中，见下文"模型下载"。

---

## 目录

```
mjtf/            模型、训练（SL/RL）、评估、对局接口、复盘工具、测试
  model.py         TileTransformer 网络
  data.py          牌谱数据集（流式混洗、单局采样上限）
  train_sl.py      监督学习
  rl.py            PPO 自我对弈强化学习
  gen_worker.py    并行自我对弈 worker
  grpnet.py        顺位预测模型 GRP（RL 奖励用）
  eval.py          重复牌山 1 对 3 对战评估
  engine.py        libriichi 对局引擎适配器（贪心/采样，bf16 快路径）
  bot.py           mjai 协议对局接口（推理入口）
  review.py        牌谱复盘 / 何切工具
  cfg/             训练配置（sl_a.json、rl.json 等）
  tests/           回归与冒烟测试
server/          服务器脚本：牌谱预处理、发布、流水线、评估、RL 计划、性能剖析
engine/          规则引擎 libriichi（基于 Mortal，已打补丁）
  libriichi.so     Linux x86_64 / Python 3.11 编译产物
  libriichi_patched.diff, mortal_commit.txt, patch_libriichi.py, build.sh
  env.sh, pip_freeze.txt   训练环境
models/          grp_hanchan.pth（顺位预测，1 MB）；主模型见 Releases
records/         训练与评估记录（验证曲线、RL 训练曲线、全部对战评估结果）
docs/            PLAN.md（分步计划）、rules.md（规则与奖励）、TIPS.md（研究备忘）
```

---

## 模型下载

| 文件 | 位置 | 说明 |
|---|---|---|
| `final_rl_it40.pth` | Releases → tag `1` | **最终模型**，151 MB，SHA-256 `5b32372b29f884ccabd9778ec9f9c2d89eec15e4b07af01bb89cc682608d0a93` |
| `models/grp_hanchan.pth` | 仓库内 | 顺位预测模型，只有继续 RL 训练时需要 |

```bash
mkdir -p models
# 私有仓库需要登录后下载（网页下载或 gh release download）
gh release download 1 -R Mrbrub/mahjong_AI -p final_rl_it40.pth -D models
```

---

## 模型架构（`mjtf/model.py`，约 4,000 万参数）

思路是把 34 种牌各当作一个 token，用自注意力直接建模牌与牌之间的关系（筋、壁、宝牌关联、搭子等）。参考实现 Mortal 用的是 ResNet 卷积。

```
输入：libriichi 局面编码 obs v6，(1014 通道 × 34 种牌)
  │  转置 → 34 个牌 token，每个 token 是 1014 维特征
  ▼
输入投影：Linear(1014→512) → GELU → Linear(512→512)
  + 可学习的牌身份嵌入 (34×512)
  + 前置 CLS token
  ▼
12 × Transformer 编码层（Pre-LN；自注意力 8 头 d=512；FFN 512→2048→512，GELU；无 dropout）
  ▼
LayerNorm
  ├─ 牌 token → Linear(512→2)：34 个"打这张牌/杠这张牌" + 3 个"打赤宝牌"（5m/5p/5s）
  ├─ CLS → MLP → 9 个其他动作：立直、吃(低/中/高)、碰、杠、和、流局、跳过
  │     合计 46 个动作，非法动作屏蔽
  └─ CLS → 辅助头：最终顺位(4)、本局点数变化(1)、价值/期望段位分(1)
```

局面编码的 1014 个通道由规则引擎生成，包括：
- 自家手牌、副露；
- 四家牌河（顺序、手切/摸切、立直宣言牌、宝牌标记）；
- 立直状态、点数、场况（场风、局数、本场、供托、余牌）、宝牌；
- 向听数、待牌、可执行动作等。

---

## 训练方式

### 数据

- 雀魂王座之间 2023/2024 半庄约 48.9 万局，加上玉之间 2019 约 1.2 万局，来自 HF 数据集 `nikketryhard/tenhou-houou-mjai`。
- 预处理（`server/prep.py`）：
  - 修复雀魂牌谱的几处问题：亲家 14 张配牌、副露目标、杠标签、缺失的点数变化；
  - 用 libriichi 回放校验，剔除不合法的牌谱；
  - 判定东风/半庄，只保留半庄。
- 训练/验证集按文件名 crc32 划出 1% 作验证。
- 另处理了玉之间 2022–2025 半庄约 261 万局（`server/publish_hanchan.py` 做质量过滤），但最终模型没有使用。

### 阶段 1：监督学习（`mjtf/train_sl.py`，配置 `mjtf/cfg/sl_a.json`）

- 损失 = 动作交叉熵 + 0.2×本局点数变化 + 0.05×最终顺位 + 0.05×价值。顺位和价值两项在 9.8 万步时从 0.2 调低，原因是过拟合。
- AdamW，lr 3e-4，warmup 3,000 步，batch 1024，共 30 万步；bf16 + torch.compile。L40 上约 6,400 样本/秒，约 13 小时。
- 数据管线：
  - 每 worker 65,536 样本的跨文件流式混洗；
  - 每局每家最多随机取 64 个决策，降低同局样本之间的相关性。
- 结果：验证集动作准确率 79.2%（切牌 74.8%、立直 83.4%、吃碰 77.0%）。
- 对战：对 9.8 万步存档每局 +5.3，对 3.6 万步存档 +8.6，对规则 AI 一位率 99.9%。

### 阶段 2：顺位预测模型 GRP（`mjtf/grpnet.py`）

- 输入每局开局状态：局数 one-hot（东1–西4 共 12 维）、本场、供托、四家点数；输出四家最终顺位分布。
- 用约 50 万局半庄训练，验证 CE 1.125（瞎猜为 1.386）。
- 用途：RL 中每局结束时，计算"本局让我的期望最终段位分变化了多少"，作为按局奖励。

### 阶段 3：强化学习（`mjtf/rl.py`，配置 `records/rl_planA2.json`）

- 算法：PPO，clip 0.2，2 epoch，minibatch 1024，lr 1e-5。
  - 对监督模型加 KL 惩罚（系数 0.05），防止偏离太远；
  - 熵奖励 0.003；
  - 价值头作为基线。
- 奖励：GRP 按局奖励 r_k = φ(s_{k+1}) − φ(s_k)，其中 φ = GRP 期望顺位分 + 素点/1000。各局奖励之和等于终局段位分减去开局时的 φ，方差远小于直接使用终局分。
- 自我对弈：
  - libriichi 半庄（可西入），重复牌山 1 对 3；
  - 每轮 500 副牌山 × 4 座位 = 2,000 局；
  - 采样温度 0.5；
  - 对手一半是监督模型，一半是自身历史版本。
- 工程优化：
  - 4 个进程并行自我对弈；
  - bf16 权重走融合推理快路径；
  - SL 参考 log-prob 每轮只算一次；
  - 静态形状 torch.compile。
  - 每轮约 4.3 分钟。
- 关键经验：
  - 采样温度 1.0 时 RL 在独立牌山上没有提升；降到 0.5 后才见效。
  - 第 40 轮最好；继续训到第 90 轮反而退步，判断为对单一对手池过拟合。
  - 选优用过的牌山会带来乐观偏差，最终验收必须换一批独立牌山。

### 评估（`mjtf/eval.py`）

- 重复牌山：每副牌让挑战者依次坐四个座位，抵消牌运。
- 输出：
  - 完整玉之间段位分 = 顺位马 [110, 55, 0, −165] + 顺位得点 [15, 5, −5, −15] + (终局点数 − 25000)/1000；
  - 以 seed 为单位的聚类标准误；
  - 和了、自摸、放铳、立直、副露率。

最终验收（独立牌山 seed 20,000,000 起，1 万副 × 4 座位 = 4 万局，三家对手为监督模型）：

| 指标 | 最终模型 | 监督模型自对（对照） |
|---|---|---|
| 每半庄段位分 | **+2.37 ±0.52** | −0.21 ±0.24 |
| 一/二/三/四位率 | 23.9 / 26.0 / 26.9 / **23.2%** | ≈ 25% 各 |
| 放铳率 | **12.6%** | 13.6% |
| 立直率 / 副露率 | 15.4% / 33.2% | 19.1% / 28.6% |

---

## 调用方式

### 环境

- **Linux x86_64 + Python 3.11**。仓库里的 `engine/libriichi.so` 是这个平台的编译产物。
- PyTorch 2.x，有无 GPU 均可。CPU 推理每步几十毫秒。
- 其他平台（如 Windows）需要自行编译 libriichi：
  1. 取 Mortal 源码，检出 `engine/mortal_commit.txt` 中的 commit；
  2. 应用 `engine/libriichi_patched.diff`；
  3. 用 Rust + PyO3 编译（参考 `engine/build.sh`）。
- 训练脚本中写死了服务器根目录 `/root/gpufree-data/mj`。换路径时需要全局替换，推理（`bot.py --ckpt`）不受影响。

```bash
pip install torch numpy
export PYTHONPATH=$PWD/engine        # 让 Python 找到 libriichi.so
```

### 1. mjai 协议对局（推荐接口）

`bot.py` 从标准输入逐行读取 mjai 事件（JSON），在需要决策时向标准输出写回动作：

```bash
python mjtf/bot.py <座位号0-3> --ckpt models/final_rl_it40.pth [--cpu]
```

```text
输入（每行一个事件）：
{"type":"start_game","names":["a","b","c","d"]}
{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"dora_marker":"3p","scores":[25000,25000,25000,25000],"tehais":[[...13张...],[...],[...],[...]]}
{"type":"tsumo","actor":0,"pai":"5m"}
输出：
{"type":"dahai","actor":0,"pai":"W","tsumogiri":false,"meta":{...}}
```

可以接入任意 mjai 兼容的对局平台或模拟器，也可以写一个录入层，在真实对局中按发生顺序喂入事件。

### 2. 牌谱复盘 / 何切

```bash
python mjtf/review.py <牌谱.json[.gz]> <座位号> --ckpt models/final_rl_it40.pth --top 3
```

对指定玩家的每个决策，输出实际选择和模型前 3 个选择及其概率，并标出不一致之处。

### 3. Python 中直接调用

```python
import torch, json, sys
sys.path += ['mjtf', 'engine']
from model import TileTransformer
from libriichi.state import PlayerState

st = torch.load('models/final_rl_it40.pth', map_location='cpu', weights_only=False)
model = TileTransformer(**st['cfg']['model']); model.load_state_dict(st['model']); model.eval()

ps = PlayerState(0)                       # 自己的座位
for event in mjai_events:                 # 按发生顺序喂入整局事件
    cans = ps.update(json.dumps(event))
    if cans.can_act:
        obs, mask = ps.encode_obs(6, False)
        with torch.no_grad():
            out = model(torch.as_tensor(obs[None]), torch.as_tensor(mask[None]))
        action = int(out['logits'][0].argmax())   # 0-33 打牌，34-36 打赤宝，37 立直 ... 45 跳过
```

### 4. 对战评估

```bash
python mjtf/eval.py --challenger A.pth --champion B.pth --seeds 2500 --workers 4 --bf16
```

### 注意：只输入"当前手牌 + 宝牌"不够

模型的输入包含整局历史：四家牌河的顺序、摸切/手切、立直宣言牌、副露、点数等。只给手牌和宝牌时，其余通道全为 0，这是训练中从未出现过的局面，防守和读牌能力会失效。可行的接法：

- 按顺序喂入全部事件；
- 或者从局面快照（手牌、四家牌河、副露、宝牌、点数）合成一段 mjai 事件后再喂入。

详见 `docs/TIPS.md` 第 4 节。

---

## 继续训练

- 继续 RL：用 `final_rl_it40.pth` 作为起点（需要 `models/grp_hanchan.pth`），参考 `records/rl_planA2.json` 与 `mjtf/cfg/rl.json`。
- 建议改进方向：增加对手多样性，不要只延长训练轮数。
- 带优化器状态的续训存档（454 MB）和完整训练日志没有放进仓库。

## 致谢

规则引擎与局面编码基于 [Mortal](https://github.com/Equim-chan/Mortal) 的 libriichi（AGPL-3.0）。本仓库附带的补丁和编译产物用于个人学习研究。

---

## 早期工作（本仓库原有内容）

根目录下的 `mahjong_class.py`、`predict.py`、`test2.py` 是更早的实验：用 YOLO 识别牌面图像，再按规则打出不必要的牌。原说明：

> 目前还在研究中，没有很明确的执行步骤。
> 当前能实现基本的打牌，已知手牌的情况下，能呆瓜式打出不必要的牌
> 其他牌种还没有研究。参数等还没有研究
> 之后如何用机械学习来建立优化模型还有待探讨。

这部分的图像识别能力，可以作为上文"局面快照 → 合成事件"方案的输入端。
