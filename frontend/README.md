# Solver Agent Frontend

这是一个基于 React + Vite 的算法筛选页面，用于展示 `solver-agent/knowledge-base/algorithms` 中的算法条目。

## 目录结构

- `index.html`：页面入口
- `src/main.jsx`：React 入口文件
- `src/App.jsx`：算法筛选页面组件
- `src/styles.css`：页面样式
- `generate_index.py`：从 `knowledge-base` 中生成聚合 JSON 数据
- `data/algorithms.json`：生成后的算法数据，供页面读取
- `package.json`：前端依赖与运行脚本
- `vite.config.js`：Vite 配置

## 使用方式

1. 生成算法数据：

```bash
cd /Users/shihuimei/beihang/solver-agent/frontend
python3 generate_index.py
```

2. 安装依赖并启动开发服务器：

```bash
cd /Users/shihuimei/beihang/solver-agent
./run_frontend.sh
```

3. 在浏览器中打开：

`http://localhost:5173`

## 说明

该页面根据每个算法目录里的 `metadata.json` 属性生成筛选条件，并支持关键词搜索、问题类型、目标类型、变量类型和算法家族筛选。
