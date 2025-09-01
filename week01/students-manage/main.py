from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="学生管理API",
    description="一个简单的 FastAPI 应用，用于管理学生信息。",
    version="1.0.0"
)

class Student(BaseModel):
    name : str
    age : int
    email : str

#初始化学生列表
students : Dict[int, Student] = {}
student_id = 1

@app.get("/students",description='获取所有学生列表')
def get_students():
    return students

@app.post("/add",description='添加学生')
async def add_student(student:Student):
    global student_id
    students[student_id] = student
    student_id += 1
    return {"id":student_id,**student.dict()}