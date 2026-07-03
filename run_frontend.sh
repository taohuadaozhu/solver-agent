#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/frontend"

if [ ! -f package.json ]; then
  echo "错误：frontend/package.json 不存在。请检查路径。"
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "正在安装前端依赖..."
  npm install --legacy-peer-deps
fi

echo "正在生成算法数据..."
python3 generate_index.py

echo "启动 React 前端开发服务器..."
npm run dev
