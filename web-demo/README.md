# Python用户管理系统

这是一个基于Flask的用户管理系统，使用MySQL数据库存储用户信息。

## 功能特性

- 用户添加
- 用户编辑
- 用户删除
- 用户列表查看
- 数据验证（用户名和邮箱唯一性检查）

## 技术栈

- Python 3.9
- Flask 2.3.3
- Flask-SQLAlchemy 3.0.5
- PyMySQL 1.1.0
- MySQL数据库

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行应用
```bash
python app.py
```

### 3. 访问应用
打开浏览器访问：http://localhost:5000

## Docker部署

### 构建镜像
```bash
docker build -t web-demo:latest .
```

### 运行容器
```bash
docker run -p 5000:5000 web-demo:latest
```

## 数据库配置

应用默认连接到MySQL数据库：
- 主机：192.168.1.2
- 端口：3306
- 数据库：stellar
- 用户名：root
- 密码：4r5t6y7u@

## 项目结构

```
web-demo/
├── app.py              # 主应用文件
├── requirements.txt    # Python依赖
├── Dockerfile         # Docker配置
├── README.md          # 项目说明
├── templates/         # HTML模板
│   └── index.html
└── static/           # 静态文件
    └── style.css
``` 