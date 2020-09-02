# !usr/bin/env python
# -*- coding:utf-8 _*-
# author:chenmeng
# datetime:2020/8/18 15:04

import time
from typing import Optional
import json
from typing import Optional
import uvicorn
from fastapi import FastAPI, Request, Header, Body, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI(title='Fastapi Demo',
              description='A simple fastapi demo, including api demo and templates',
              version='0.1.0')

# 挂载静态文件
app.mount('/static', StaticFiles(directory='static'), name='static')
# 模板目录
templates = Jinja2Templates(directory='templates')


class Item(BaseModel):
    name: str
    # price: float
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
    return {'item_color': item_color, 'item_price': item.price}


@app.post('/async_test1', tags=['async test'])
async def async_test1(item: Item):
    time.sleep(1)
    data = json.dumps({'async1': item.name})
    return JSONResponse(content=data)


@app.post('/async_test2', tags=['async test'])
async def async_test2(item: Item):
    time.sleep(1)
    return {'async2': item.name}

@app.post('/test')
def test(item: Item = Body(None, media_type='text/plain;charset=UTF-8')):
    return {'content': item.name.encode('utf-8')}


@app.get('/data/{data}', tags=['demo html'], summary='fastapi配合Jinja2渲染HTML模板',
         description='输入网址：http://127.0.0.1:8080/data/test，即可访问到demo页面')
async def read_data(request: Request, data: str):
    return templates.TemplateResponse('index.html', {'request': request, 'data': data})


if __name__ == '__main__':
    uvicorn.run(app='fastapi_demo:app', host='0.0.0.0', port=8080, reload=True)
