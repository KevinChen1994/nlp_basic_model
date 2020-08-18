# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/8/18 15:04

import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title='Fastapi Demo',
              description='A simple fastapi demo, including api demo and templates',
              version='0.1.0')

# 挂载静态文件
app.mount('/static', StaticFiles(directory='static'), name='static')
# 模板目录
templates = Jinja2Templates(directory='templates')


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


# get请求 http://127.0.0.1:8080
@app.get("/", tags=['hello world'], summary='hello world summary', description='hello world description')
async def read_root():
    return {"Hello": "World"}


# get请求  http://127.0.0.1:8080/items/1?q=1
@app.get("/items/{item_id}", tags=['search data'])
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


# post请求 http://127.0.0.1:8080/items/1?item_color=red   在加上一个request body
@app.post('/items/{item_id}', tags=['upload data'])
async def update_item(item_id: int, item_color: str, item: Item):
    if item.is_offer:
        return {'item_id': item_id, 'item_name': item.name}
    else:
        return {'item_color': item_color, 'item_price': item.price}


@app.post('/async_test1', tags=['async test'])
async def async_test1(item: Item):
    time.sleep(2)
    return {'async1': item.name}


@app.post('/async_test2', tags=['async test'])
async def async_test2(item: Item):
    time.sleep(1)
    return {'async2': item.name}


@app.get('/data/{data}', tags=['demo html'], summary='fastapi配合Jinja2渲染HTML模板',
         description='输入网址：http://127.0.0.1:8080/data/test，即可访问到demo页面')
async def read_data(request: Request, data: str):
    return templates.TemplateResponse('index.html', {'request': request, 'data': data})


if __name__ == '__main__':
    uvicorn.run(app='fastapi_demo:app', host='127.0.0.1', port=8080, reload=True)
