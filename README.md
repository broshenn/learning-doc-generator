# Learning Doc Generator

> Claude Code Skill：一键生成高质量中文技术学习文档

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 这是什么

`learning-doc-writer` 是一个 Claude Code 全局 Skill。当你输入「写一份关于 XXX 的学习文档」时，它自动生成结构严谨、数学清晰、代码可运行的完整教程——就像一本精心打磨的技术教材。

格式规范融合了 DataWhale Hello-Agents 和 动手学强化学习 的写作风格。

## 核心特性

- 📖 **定义 → 要素 → 案例**三段式概念引入
- 📐 **数学公式逐一拆解**，每个符号标注含义和典型取值，绝无裸公式
- 💻 **完整可运行代码**，附带实际测试输出，零伪代码、零占位符
- 🧮 **手算推演**，超小数据手动过一轮，确保读者真正理解
- 📊 文末**全景对比表** + 🎯 **选择指南决策树**
- 🌈 emoji 辅助导航，不过度使用
- 🔗 每个方法的局限段落**承上启下**——明确指出下一节如何解决当前问题

## 快速开始

```bash
# 安装
npx skills add broshenn/learning-doc-generator@learning-doc-writer -g

# 重启 Claude Code，然后说：
写一份关于 [主题] 的学习文档
```

## 示例输出

仓库附带了一份完整示例：[优化器进化史：从 SGD 到 AdamW](examples/optimizer_learning_guide.md)

文档覆盖 SGD → Momentum → RMSProp → Adam → AdamW 的完整进化链，每个优化器包含：

- 定义 + 要素拆解 + 真实训练场景例子
- 数学公式 + 每个符号逐一讲解
- 手算推演（用同一个简单函数跑通所有方法）
- 完整的 PyTorch 实现 + 实际运行输出
- 文字形式的优缺分析（含承上启下过渡）

## 目录结构

```
learning-doc-generator/
├── README.md
├── skill/
│   └── SKILL.md                        # Skill 定义（写作规范编码）
└── examples/
    └── optimizer_learning_guide.md     # 完整示例文档
```

## 格式灵感

- [Datawhale Hello-Agents](https://hello-agents.datawhale.cc) — 章节结构、表格导航
- [动手学强化学习](https://hrl.boyuai.com) — 数学逐行解释、代码运行输出

## License

MIT
