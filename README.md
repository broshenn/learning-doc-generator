# Learning Doc Generator

> 一键生成高质量中文技术学习文档的 Claude Code Skill

## 这是什么

`learning-doc-writer` 是一个 Claude Code 全局 Skill。当你输入「写一份关于 XXX 的学习文档」时，它会自动生成结构清晰、数学严谨、代码可运行的完整中文技术教程。

## 核心特性

- 📖 **定义—要素—案例** 三段式概念引入，每个知识点的定义前必须有衔接段落
- 📐 **数学公式 + 符号逐一讲解**，绝无裸公式，每个符号有物理含义和典型取值
- 💻 **完整可运行代码**，无伪代码、无 `pass`/`TODO` 占位符，附带实际测试输出
- 🧮 **手算推演**，超小数据手动推一轮帮助理解
- 📊 **全景对比表** + 🎯 选择指南决策树
- 🌈 emoji 辅助导航，但不过度使用

## 安装

### 前置条件

- [Claude Code](https://claude.ai/code) 已安装
- Node.js（用于 `npx skills` 命令）

### 安装方式一：Skills CLI（推荐）

```bash
npx skills add broshenn/learning-doc-generator@learning-doc-writer -g
```

重启 Claude Code 后生效。

### 安装方式二：手动安装

```bash
mkdir -p ~/.claude/skills/learning-doc-writer
cp skill/SKILL.md ~/.claude/skills/learning-doc-writer/SKILL.md
```

重启 Claude Code 后生效。

## 使用

在 Claude Code 中直接说：

```
写一份关于 [主题] 的学习文档
```

或在 Claude Code 中输入 `/learning-doc-writer` 然后描述主题。

## 示例

[examples/optimizer_learning_guide.md](examples/optimizer_learning_guide.md) — 一份完整的优化器学习文档，覆盖 SGD → Momentum → RMSProp → Adam → AdamW 的进化链路，包含：

- 每个优化器的数学推导 + 手算推演
- 完整的可运行 Python 实现（含测试输出）
- 五者全景对比表
- 选择指南决策树

## 目录结构

```
learning-doc-generator/
├── README.md                           # 本文件
├── skill/
│   └── SKILL.md                        # Skill 定义文件
└── examples/
    └── optimizer_learning_guide.md     # 优化器学习文档示例
```

## 许可证

MIT
