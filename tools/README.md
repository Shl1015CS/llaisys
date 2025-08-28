# 🛠️ 开发工具 (Development Tools)

本目录包含用于开发和调试的实用工具脚本。

## 📁 目录结构

### 🔍 weights/ - 权重检查工具
- **`inspect_safetensors.py`** - 检查safetensors文件中的张量信息
- **`view_weights.py`** - 查看和分析模型权重

### 🤖 models/ - 模型分析工具  
- **`qwen_analysis.py`** - Qwen模型行为分析和对比

## 🚀 使用示例

### 检查模型权重
```bash
# 查看safetensors文件中的所有张量
python tools/weights/inspect_safetensors.py model.safetensors

# 查看特定权重张量
python tools/weights/view_weights.py model.safetensors tensor_name
```

### 分析模型行为
```bash
# 分析Qwen模型
python tools/models/qwen_analysis.py --model /path/to/model
```

## 📝 注意事项

- 这些工具主要用于开发调试，不是核心库的一部分
- 使用前请确保安装了必要的依赖包
- 部分脚本可能需要大量内存来加载模型
