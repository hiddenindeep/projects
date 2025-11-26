## ChatBI 接口文档

本文档描述当前 ChatBI 项目前后端交互使用到的 HTTP 接口。后端分为：

- **主业务服务**：`main_server.py`（`uvicorn main_server:app`）
- **股票数据子服务**：`api/autostock.py`，通过 `app.mount("/stock", stock_app)` 挂载到 `/stock` 路径

所有示例均省略统一响应包装字段，仅展示关键结构。

---

## 一、主服务通用信息

- **Base URL（本地开发）**：`http://127.0.0.1:8000`
- **文档（Swagger UI）**：`http://127.0.0.1:8000/docs#/`

### 1. 健康检查

- **Method**：`GET`
- **Path**：`/v1/healthy`
- **说明**：用于健康检查（当前实现为空 `pass`，可按需返回简单 JSON）。

---

## 二、用户模块（User APIs）

- **Router 前缀**：`/v1/users`
- **文件**：`routers/user.py`
- **统一响应模型**：`BasicResponse`

```json
{
  "code": 200,
  "message": "提示信息",
  "data": {}
}
```

### 1. 用户登录

- **Method**：`POST`
- **Path**：`/v1/users/login`
- **Request Body**：`RequestForUserLogin`

```json
{
  "user_name": "string",
  "password": "string"
}
```

- **Response**：
  - 成功：`code = 200, message = "用户登陆成功"`
  - 失败：`code = 400, message = "用户名或密码错误"`

---

### 2. 用户注册

- **Method**：`POST`
- **Path**：`/v1/users/register`
- **Request Body**：`RequestForUserRegister`

```json
{
  "user_name": "string",
  "password": "string",
  "user_role": "string"
}
```

- **Response**：
  - 成功：`code = 200, message = "用户注册成功"`
  - 失败：`code = 400, message = "用户名已存在"`

---

### 3. 重置密码

- **Method**：`POST`
- **Path**：`/v1/users/reset-password`
- **Request Body**：`RequestForUserResetPassword`

```json
{
  "user_name": "string",
  "password": "string",
  "new_password": "string"
}
```

- **Response**：
  - 成功：`code = 200, message = "密码重置成功"`
  - 失败：`code = 400, message = "用户名或密码错误"` 或 `"密码重置失败"`

---

### 4. 获取用户信息

- **Method**：`POST`
- **Path**：`/v1/users/info`
- **Request Params**：
  - `user_name: string`（query 或 form）

- **Response**：

```json
{
  "code": 200,
  "message": "获取用户信息成功",
  "data": {
    "user_id": 1,
    "user_name": "string",
    "user_role": "string",
    "register_time": "2025-01-01T00:00:00",
    "status": true
  }
}
```

---

### 5. 修改用户信息

- **Method**：`POST`
- **Path**：`/v1/users/reset-info`
- **Request Body**：`RequestForUserChangeInfo`

```json
{
  "user_name": "string",
  "user_role": "optional string",
  "status": "optional bool"
}
```

- **Response**：
  - 成功：`code = 200, message = "用户信息修改成功"`

---

### 6. 删除用户

- **Method**：`POST`
- **Path**：`/v1/users/delete`
- **Request Params**：
  - `user_name: string`

- **Response**：
  - 成功：`code = 200, message = "用户删除成功"`
  - 失败：`code = 400, message = "用户不存在"`

---

### 7. 用户列表

- **Method**：`POST`
- **Path**：`/v1/users/list`
- **Request Body**：无
- **Response**：

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "user_id": 1,
      "user_name": "string",
      "user_role": "string",
      "register_time": "2025-01-01T00:00:00",
      "status": true
    }
  ]
}
```

---

## 三、对话模块（Chat APIs）

- **Router 前缀**：`/v1/chat`
- **文件**：`routers/chat.py`

### 通用请求体：`RequestForChat`

```json
{
  "content": "用户提问",
  "user_name": "string",
  "session_id": "optional string",
  "task": "optional string",
  "tools": ["optional", "tool", "names"],
  "image_content": "optional string",
  "file_content": "optional string",
  "url_content": "optional string",
  "audio_content": "optional string",
  "video_content": "optional string",
  "vison_mode": false,
  "deepsearch_mode": false,
  "sql_interpreter": false,
  "code_interpreter": false
}
```

---

### 1. 流式对话接口

- **Method**：`POST`
- **Path**：`/v1/chat/`
- **Request Body**：`RequestForChat`
- **Response**：
  - `StreamingResponse`，`Content-Type: text/event-stream`
  - SSE 流式返回，对应每一段模型输出文本。

---

### 2. 初始化对话（获取 session_id）

- **Method**：`POST`
- **Path**：`/v1/chat/init`
- **Request Body**：无
- **Response**：

```json
{
  "code": 200,
  "message": "ok",
  "data": {
    "session_id": "生成的会话ID"
  }
}
```

---

### 3. 获取会话详情

- **Method**：`POST`
- **Path**：`/v1/chat/get`
- **Request Params**：
  - `session_id: string`

- **Response**：

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    // 该会话对应的消息记录
  ]
}
```

---

### 4. 删除会话

- **Method**：`POST`
- **Path**：`/v1/chat/delete`
- **Request Params**：
  - `session_id: string`

- **Response**：
  - 成功：`code = 200, message = "ok"`

---

### 5. 会话列表

- **Method**：`POST`
- **Path**：`/v1/chat/list`
- **Request Params**：
  - `user_name: string`

- **Response**：

```json
{
  "code": 200,
  "message": "ok",
  "data": [
    {
      "user_id": 1,
      "session_id": "string",
      "title": "string",
      "start_time": "2025-01-01T00:00:00",
      "feedback": true,
      "feedback_time": "2025-01-01T00:10:00"
    }
  ]
}
```

---

### 6. 对话反馈（点赞/踩）

- **Method**：`POST`
- **Path**：`/v1/chat/feedback`
- **Request Params**：
  - `session_id: string`
  - `message_id: int`
  - `feedback: bool`

- **Response**：
  - 成功：`code = 200, message = "ok"`

---

## 四、数据模块（Data APIs）

- **Router 前缀**：`/v1/data`
- **文件**：`routers/data.py`
- 目前为预留接口，函数体尚未实现，仅列出占位定义。

1. **下载数据**
   - Method：`POST`
   - Path：`/v1/data/download`

2. **创建数据**
   - Method：`POST`
   - Path：`/v1/data/create`

3. **上传数据**
   - Method：`POST`
   - Path：`/v1/data/upload`

4. **删除数据**
   - Method：`POST`
   - Path：`/v1/data/delete`

---

## 五、自选股模块（User Favorite Stock APIs）

- **Router 前缀**：`/v1/stock`
- **文件**：`routers/stock.py`
- **统一响应模型**：`BasicResponse`

### 1. 获取用户所有自选股

- **Method**：`POST`
- **Path**：`/v1/stock/list_fav_stock`
- **Request Params**：
  - `user_name: string`

- **Response 示例**：

```json
{
  "code": 200,
  "message": "获取用户所有股票成功",
  "data": [
    {
      "stock_code": "600000.SH",
      "create_time": "2025-01-01T00:00:00"
    }
  ]
}
```

---

### 2. 删除某个自选股

- **Method**：`POST`
- **Path**：`/v1/stock/del_fav_stock`
- **Request Params**：
  - `user_name: string`
  - `stock_code: string`

- **Response**：
  - 成功：`code = 200, message = "删除成功"`

---

### 3. 新增自选股

- **Method**：`POST`
- **Path**：`/v1/stock/add_fav_stock`
- **Request Params**：
  - `user_name: string`
  - `stock_code: string`

- **Response**：
  - 成功：`code = 200, message = "添加成功"`

---

### 4. 清空用户自选股

- **Method**：`POST`
- **Path**：`/v1/stock/clear_fav_stock`
- **Request Params**：
  - `user_name: string`

- **Response**：
  - 成功：`code = 200, message = "删除成功"`

---

## 六、股票数据子服务（Autostock APIs）

- **Mount 前缀**：`/stock`
- **文件**：`api/autostock.py`
- **Base URL（本地开发）**：`http://127.0.0.1:8000/stock`
- **说明**：对接第三方 `autostock` 服务，返回原始 JSON 数据。

> 所有接口 Method 为 `GET`。

### 1. 股票列表（模糊查询）

- **Path**：`/stock/get_stock_code`
- **Query Params**：
  - `keyword: Optional[str]`，支持代码和名称模糊查询。

---

### 2. 指数列表

- **Path**：`/stock/get_index_code`
- **说明**：获取所有指数信息。

---

### 3. 行业排行

- **Path**：`/stock/get_industry_code`
- **说明**：获取行业/板块排行数据。

---

### 4. 大盘信息

- **Path**：`/stock/get_board_info`
- **说明**：获取大盘指数简要行情。

---

### 5. 股票排行

- **Path**：`/stock/get_stock_rank`
- **Query Params**：
  - `node: str` — 市场/板块代码，如 `a, b, ash, asz, bsh, bsz` 等。
  - `industryCode: Optional[str]` — 行业代码，非必填。
  - `pageIndex: int = 1`
  - `pageSize: int = 100`
  - `sort: str = "price"` — 排序字段，如 `priceChange, pricePercent, volume` 等。
  - `asc: int = 0` — `0` 降序（默认），`1` 升序。

---

### 6. 月 K 线

- **Path**：`/stock/get_month_line`
- **Query Params**：
  - `code: str` — 股票代码
  - `startDate: Optional[str]`
  - `endDate: Optional[str]`
  - `type: int = 0` — `0` 不复权，`1` 前复权，`2` 后复权

---

### 7. 周 K 线

- **Path**：`/stock/get_week_line`
- **Query Params**：同上。

---

### 8. 日 K 线

- **Path**：`/stock/get_day_line`
- **Query Params**：同上。

---

### 9. 股票基础信息

- **Path**：`/stock/get_stock_info`
- **Query Params**：
  - `code: str` — 股票代码

---

### 10. 股票分时数据

- **Path**：`/stock/get_stock_minute_data`
- **Query Params**：
  - `code: str` — 股票代码

---

## 七、接口总览表

| 模块       | 前缀/挂载路径 | 接口数量 | 说明                           |
| ---------- | ------------- | -------- | ------------------------------ |
| 通用       | `/v1`         | 1        | 健康检查                       |
| 用户       | `/v1/users`   | 7        | 登录 / 注册 / 信息 / 列表等   |
| 对话       | `/v1/chat`    | 6        | 流式对话 / 历史 / 反馈        |
| 数据       | `/v1/data`    | 4        | 预留，待实现                   |
| 自选股     | `/v1/stock`   | 4        | 用户自选股增删改查            |
| 股票数据   | `/stock`      | 10       | 对接 autostock 股票行情接口   |

> 实际可用接口数量会随后续开发（特别是 `/v1/data`）而增加，请以 `http://127.0.0.1:8000/docs#/` 实时文档为准。


